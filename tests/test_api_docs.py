"""`docs/api.md` is generated, so it cannot quietly stop being true.

The page is rendered from `fz.__all__` by introspection. This test renders it
again under pytest and compares byte for byte, which means a renamed parameter,
a reordered signature, a new export or a rewritten summary line all fail here
until the page is regenerated (`make docs`).
"""

import pathlib
import re
import sys

import fz

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))

from gen_api_docs import OUT, render  # noqa: E402


def test_committed_api_page_is_what_the_generator_produces():
    assert OUT.exists(), "docs/api.md is missing; run `make docs`"
    committed = OUT.read_text(encoding="utf-8")
    assert committed == render(), (
        "docs/api.md has drifted from the code; run `python3 tools/gen_api_docs.py`"
    )


def test_every_export_appears_on_the_page():
    """Belt to the generator's braces: the generator refuses to render an
    export it cannot place in a section, and this asserts the placed name
    really reached the page. Dataclasses head their section as
    `class Name(...)`, so the name may be preceded by the `class` keyword.
    """
    page = OUT.read_text(encoding="utf-8")
    for name in fz.__all__:
        if name.startswith("__"):
            continue
        assert re.search(rf"`(?:class )?{re.escape(name)}\b", page), (
            f"{name} is exported but not documented"
        )
