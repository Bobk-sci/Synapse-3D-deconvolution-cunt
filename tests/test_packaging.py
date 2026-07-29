"""Every third-party module the package imports must be a declared dependency.

Most imports here are deliberately made inside functions, to keep start-up fast
and to avoid pulling matplotlib into a headless batch that never draws. The cost
is that an undeclared dependency is invisible until the exact code path runs on
a machine that does not happen to have it installed -- which is how
scikit-image, used only by the punctum segmentation, shipped undeclared.

This test reads the imports out of the source instead of executing it, so a
lazily imported package is checked like any other.
"""

from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "synapse_deconv"
PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

#: Import name -> distribution name, where the two differ.
_DISTRIBUTION_NAMES = {
    "skimage": "scikit-image",
    "yaml": "pyyaml",
    "cupy": "cupy",
    "cupyx": "cupy",
    "PIL": "pillow",
}

#: Optional at runtime: the code must degrade gracefully without them.
_OPTIONAL = {"cupy", "cupyx"}


def _imported_top_level_modules() -> dict[str, set[str]]:
    """Top-level module names imported by each file of the package."""
    found: dict[str, set[str]] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:      # skip relative imports
                    names.add(node.module.split(".")[0])
        found[path.name] = names
    return found


def _declared_distributions() -> set[str]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = data["project"]
    requirements = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        requirements.extend(extra)
    names = set()
    for requirement in requirements:
        name = requirement.split(">")[0].split("<")[0].split("=")[0].split("[")[0]
        names.add(name.strip().lower().replace("_", "-"))
    return names


def test_every_third_party_import_is_declared():
    declared = _declared_distributions()
    stdlib = set(sys.stdlib_module_names)
    undeclared: list[str] = []

    for filename, modules in _imported_top_level_modules().items():
        for module in modules:
            if module in stdlib or module == "synapse_deconv":
                continue
            distribution = _DISTRIBUTION_NAMES.get(module, module).lower()
            if distribution in _OPTIONAL:
                continue
            if distribution not in declared:
                undeclared.append(f"{filename} imports {module!r} -> {distribution}")

    assert not undeclared, (
        "undeclared dependencies (add them to pyproject.toml):\n  "
        + "\n  ".join(sorted(set(undeclared)))
    )


def test_scikit_image_is_declared():
    """It is imported lazily inside detect_puncta, which hid it from view."""
    assert "scikit-image" in _declared_distributions()


@pytest.mark.parametrize("module", ["numpy", "scipy", "tifffile", "yaml", "oiffile"])
def test_core_dependencies_are_declared(module):
    distribution = _DISTRIBUTION_NAMES.get(module, module).lower()
    assert distribution in _declared_distributions()


def test_optional_backend_is_not_a_hard_requirement():
    """CuPy must stay optional: the NumPy path is the documented default."""
    assert "cupy" not in _declared_distributions()
