"""Runtime translation for the editor's user-facing strings.

The Chinese source text doubles as the lookup key, the way gettext uses a
msgid. Two properties follow from that and both matter here:

* A missing or incomplete catalog degrades to the original Chinese string
  rather than to a blank label, so a half-translated build still runs.
* Call sites stay readable -- ``t("笔刷库")`` says what it renders.

Adding a language means dropping ``<code>.json`` into ``locales/``; no code
change is required. ``zh`` is the source language and therefore has no file.

Interpolation uses named placeholders so translators can reorder them:

    t("已选择 {columns}×{rows}，共 {count} 张", columns=4, rows=3, count=12)
"""

from __future__ import annotations

import json
import locale
import logging
import os
import threading
from pathlib import Path
from typing import Callable, Iterable

LOGGER = logging.getLogger(__name__)

LOCALE_DIR = Path(__file__).resolve().parent / "locales"
SOURCE_LANGUAGE = "zh"

# Display names are intentionally written in their own language.
LANGUAGE_NAMES: dict[str, str] = {
    "zh": "中文",
    "en": "English",
    "ja": "日本語",
}

_ENV_OVERRIDE = "KISEKI_LANGUAGE"

# The chosen language lives beside the other runtime state, outside version
# control. It is read once at import so that module-level ``t(...)`` constants
# are already rendered in the right language.
PREFERENCE_FILE = Path(__file__).resolve().parent.parent / "runtime" / "language.txt"


def _normalize(tag: str | None) -> str | None:
    """Map a locale tag such as ``ja_JP`` or ``zh-Hans-CN`` onto a language code."""
    if not tag:
        return None
    code = tag.replace("-", "_").split("_", 1)[0].strip().lower()
    return code or None


def detect_system_language(available: Iterable[str]) -> str:
    """Pick the best available language for this machine, falling back to English."""
    options = set(available)
    candidates: list[str | None] = [_normalize(os.environ.get(_ENV_OVERRIDE))]
    try:
        candidates.append(_normalize(locale.getlocale()[0]))
    except (ValueError, TypeError):
        pass
    try:
        candidates.append(_normalize(locale.getdefaultlocale()[0]))  # noqa: DEP004
    except Exception:  # pragma: no cover - platform dependent
        pass
    for name in ("LC_ALL", "LC_MESSAGES", "LANG"):
        candidates.append(_normalize(os.environ.get(name)))

    for candidate in candidates:
        if candidate and candidate in options:
            return candidate
    return "en" if "en" in options else SOURCE_LANGUAGE


class Translator:
    """Holds the active language and its catalog.

    Instances are cheap; the module keeps one shared instance that the UI uses.
    """

    def __init__(self, locale_dir: Path | None = None) -> None:
        self._locale_dir = Path(locale_dir) if locale_dir is not None else LOCALE_DIR
        self._lock = threading.RLock()
        self._catalog: dict[str, str] = {}
        self._listeners: list[Callable[[str], None]] = []
        self._language = SOURCE_LANGUAGE
        self._available = self._scan_languages()

    # ---------------------------------------------------------------- catalog

    def _scan_languages(self) -> tuple[str, ...]:
        found = {SOURCE_LANGUAGE}
        if self._locale_dir.is_dir():
            for path in self._locale_dir.glob("*.json"):
                code = _normalize(path.stem)
                if code:
                    found.add(code)
        ordered = [code for code in ("zh", "en", "ja") if code in found]
        ordered.extend(sorted(found.difference(ordered)))
        return tuple(ordered)

    def _load_catalog(self, language: str) -> dict[str, str]:
        if language == SOURCE_LANGUAGE:
            return {}
        path = self._locale_dir / f"{language}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            LOGGER.warning("Locale file missing: %s", path)
            return {}
        except (OSError, ValueError) as exc:
            LOGGER.warning("Cannot read locale %s: %s", path, exc)
            return {}
        if not isinstance(payload, dict):
            LOGGER.warning("Locale %s is not an object", path)
            return {}
        return {str(key): str(value) for key, value in payload.items() if value}

    # --------------------------------------------------------------- language

    @property
    def available_languages(self) -> tuple[str, ...]:
        return self._available

    @property
    def language(self) -> str:
        with self._lock:
            return self._language

    def set_language(self, language: str) -> bool:
        """Activate ``language``. Returns ``False`` when it is not available."""
        code = _normalize(language)
        if not code or code not in self._available:
            return False
        with self._lock:
            if code == self._language:
                return True
            self._catalog = self._load_catalog(code)
            self._language = code
            listeners = tuple(self._listeners)
        for listener in listeners:
            try:
                listener(code)
            except Exception:  # pragma: no cover - a listener must not break switching
                LOGGER.exception("Language listener failed")
        return True

    def use_system_language(self) -> str:
        self.set_language(detect_system_language(self._available))
        return self.language

    # ------------------------------------------------------------ preference

    def load_preference(self, path: Path | None = None) -> str:
        """Activate the saved language, or the system one when nothing is saved."""
        target = Path(path) if path is not None else PREFERENCE_FILE
        saved: str | None = None
        try:
            saved = _normalize(target.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            saved = None
        if saved and saved in self._available:
            self.set_language(saved)
        else:
            self.use_system_language()
        return self.language

    def save_preference(self, path: Path | None = None) -> bool:
        target = Path(path) if path is not None else PREFERENCE_FILE
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(self.language, encoding="utf-8")
        except OSError as exc:
            LOGGER.warning("Cannot save language preference to %s: %s", target, exc)
            return False
        return True

    def add_listener(self, callback: Callable[[str], None]) -> None:
        with self._lock:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[str], None]) -> None:
        with self._lock:
            if callback in self._listeners:
                self._listeners.remove(callback)

    # ------------------------------------------------------------- translate

    def translate(self, text: str, /, **values: object) -> str:
        with self._lock:
            rendered = self._catalog.get(text, text)
        if not values:
            return rendered
        try:
            return rendered.format(**values)
        except (KeyError, IndexError, ValueError):
            # A broken translation must never take the UI down: fall back to the
            # source string, and only then give up and return it unformatted.
            try:
                return text.format(**values)
            except (KeyError, IndexError, ValueError):
                LOGGER.warning("Cannot interpolate string: %r", text)
                return text


_TRANSLATOR = Translator()
_TRANSLATOR.load_preference()


def get_translator() -> Translator:
    return _TRANSLATOR


def t(text: str, /, **values: object) -> str:
    """Translate ``text`` into the active language."""
    return _TRANSLATOR.translate(text, **values)


def language_display_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


def available_languages() -> tuple[str, ...]:
    return _TRANSLATOR.available_languages


def current_language() -> str:
    return _TRANSLATOR.language


def set_language(language: str, *, persist: bool = True) -> bool:
    """Switch language and, by default, remember the choice for the next run."""
    if not _TRANSLATOR.set_language(language):
        return False
    if persist:
        _TRANSLATOR.save_preference()
    return True
