from __future__ import annotations

import ast
import builtins
import unittest
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1] / "app"


def _collect_defined_names(tree: ast.Module) -> set[str]:
    names: set[str] = set(dir(builtins))
    names.update({"__file__", "__name__", "__doc__", "__package__", "__spec__"})
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                for sub in ast.walk(target):
                    if isinstance(sub, ast.Name):
                        names.add(sub.id)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            for sub in ast.walk(node.target):
                if isinstance(sub, ast.Name):
                    names.add(sub.id)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    for sub in ast.walk(item.optional_vars):
                        if isinstance(sub, ast.Name):
                            names.add(sub.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.Lambda,)):
            pass
        elif isinstance(node, ast.comprehension):
            for sub in ast.walk(node.target):
                if isinstance(sub, ast.Name):
                    names.add(sub.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Global):
            names.update(node.names)
    return names


class ModuleNameHygieneTest(unittest.TestCase):
    """Catch module-global names used without an import.

    ``compileall`` does not catch these: a missing ``import logging`` inside a
    worker thread produced NameError on every GPU view request, which silently
    froze the viewport at LOD4 and surfaced only in the window title. This scan
    is deliberately conservative - it only flags loads of names that appear
    nowhere as an import, definition, or binding in the module.
    """

    def test_every_loaded_global_name_is_defined_somewhere(self):
        problems: list[str] = []
        for path in sorted(APP_ROOT.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            defined = _collect_defined_names(tree)
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    if node.id not in defined:
                        problems.append(f"{path.name}:{node.lineno} uses undefined name {node.id!r}")
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
