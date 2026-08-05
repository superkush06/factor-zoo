"""README's ``pycon`` blocks have to keep printing what they print.

The tour block in `README.md` quotes a live pipeline -- the recovered
coefficients, their t-stats, the number of cross-sections that survive warm-up
-- and the prose beside it claims each of those numbers is pinned by a test.
This is that test. It extracts every ``pycon`` block from the README and runs
it as a doctest, so no printed number can drift without CI going red.

Why not `--doctest-glob=*.md` in `pyproject.toml`: doctest swallows the closing
fence of a fenced block into the last example's expected output, so a whole-file
run fails on both blocks (expected ``1247\\n```'', got ``1247``). Extracting the
block first is what makes the fences invisible.
"""

import doctest
import pathlib
import re

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

_BLOCK = re.compile(r"^```pycon$\n(.*?)^```$", re.MULTILINE | re.DOTALL)

# The README quotes what numpy 2 prints (`np.True_` for a `np.bool_`); numpy 1
# printed `True` for the same expression. `pip install -e .` gets numpy 2.
_NUMPY2 = int(np.__version__.split(".")[0]) >= 2


def _pycon_blocks() -> list[tuple[int, str]]:
    text = README.read_text(encoding="utf-8")
    return [(text[:m.start()].count("\n") + 1, m.group(1)) for m in _BLOCK.finditer(text)]


BLOCKS = _pycon_blocks()


def test_both_pycon_blocks_are_found_and_every_example_is_covered():
    """If a block is added or a fence renamed, this is what notices."""
    assert len(BLOCKS) == 2, f"expected 2 pycon blocks in README.md, found {len(BLOCKS)}"
    parser = doctest.DocTestParser()
    counts = [len(parser.get_examples(src)) for _, src in BLOCKS]
    assert counts == [14, 6], f"README pycon examples moved: {counts}"


@pytest.mark.skipif(not _NUMPY2, reason="README quotes numpy 2 scalar reprs (np.True_)")
@pytest.mark.parametrize("lineno, source", BLOCKS,
                         ids=[f"README.md:{ln}" for ln, _ in BLOCKS])
def test_pycon_block_reproduces_line_for_line(lineno, source, monkeypatch):
    # The second block reads data/sample_prices.csv by relative path.
    monkeypatch.chdir(ROOT)
    test = doctest.DocTestParser().get_doctest(
        source, {}, f"README.md:{lineno}", str(README), lineno)
    report: list[str] = []
    result = doctest.DocTestRunner(verbose=False).run(test, out=report.append)
    assert result.attempted > 0, "block ran no examples"
    assert result.failed == 0, "".join(report)
