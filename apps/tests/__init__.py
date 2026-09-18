"""Test package setup.

A few tests compare against tier and mode names that are also user-facing
strings, so the suite has to run in a known language. Without this the result
would depend on the machine's locale: the same tests pass on a Chinese desktop
and fail on an English one.

The variable is set before any application module is imported, because
``app.i18n`` resolves the active language once, at import time. ``setdefault``
keeps an explicit ``KISEKI_LANGUAGE`` from the environment, so the suite can
still be run against another language on purpose.
"""

from __future__ import annotations

import os

os.environ.setdefault("KISEKI_LANGUAGE", "zh")
