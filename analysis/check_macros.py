"""Fail the build when the manuscript references a number that does not exist.

This exists because of a specific, embarrassing failure. The first version of
this paper shipped a PDF containing the sentence "over-salts by x at the
parallelism we tested" -- the figure was missing because a macro had been
retired from the generated numbers file while the prose still referenced it.
LaTeX does not stop for that. An undefined control sequence in a well-formed
document produces a warning buried in a log nobody reads, and renders as
nothing at all.

An automated check is the only reliable defence: a human proofreading for
*absent* text is a human looking for something that leaves no trace.

The check runs in both directions. Undefined macros are errors, because they
silently corrupt the text. Defined-but-unused macros are reported as warnings,
because they usually mean a claim was cut and its supporting number left
behind -- harmless, but a signal that the prose and the data have drifted.

Usage:
    python analysis/check_macros.py paper
    python analysis/check_macros.py paper --strict   # unused macros also fail
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# A generated macro: capitalised, letters only. This deliberately does not match
# LaTeX built-ins (\section, \textbf) or the class's own commands, which are
# lower-case or mixed and are not our concern.
# Digits must be allowed inside the name. An earlier version used
# [A-Z][A-Za-z]* with a trailing word boundary, which could not match a macro
# containing a digit at all -- so \BaseCoalWall2dpThin was invisible to the
# very check whose purpose is to catch it. A checker with a blind spot is worse
# than no checker, because it is trusted.
MACRO_USE = re.compile(r"\\([A-Z][A-Za-z0-9]*)")
MACRO_DEF = re.compile(r"\\newcommand\{\\([A-Za-z0-9]+)\}")
PROVIDE_DEF = re.compile(r"\\providecommand\{\\([A-Za-z0-9]+)\}")

# Commands defined by IEEEtran, amsmath, hyperref and friends that legitimately
# begin with a capital and are not ours to generate.
KNOWN_EXTERNAL: Set[str] = {
    "IEEEauthorblockN", "IEEEauthorblockA", "IEEEoverridecommandlockouts",
    "IEEEkeywords", "BibTeX", "LaTeX", "TeX", "Roman", "Alph", "Large",
    "LARGE", "Huge", "REQUIRE", "ENSURE", "RETURN", "IF", "ELSE", "ELSIF",
    "ENDIF", "FOR", "ENDFOR", "WHILE", "ENDWHILE", "STATE", "COMMENT",
    "AND", "OR", "NOT", "TRUE", "FALSE", "PRINT", "REPEAT", "UNTIL",
    "LOOP", "ENDLOOP", "Delta", "Theta", "Gamma", "Lambda", "Sigma", "Omega",
    "Pi", "Phi", "Psi", "Xi", "Upsilon", "AA", "S", "P", "L", "O",
    # amsmath / LaTeX maths delimiters and operators.
    "Bigl", "Bigr", "Big", "Bigg", "Biggl", "Biggr", "Longrightarrow",
    "Longleftarrow", "Rightarrow", "Leftarrow", "Leftrightarrow", "Pr",
    "Re", "Im", "Vert", "Arrowvert", "Downarrow", "Uparrow", "Updownarrow",
}


def collect(paper_dir: Path) -> Tuple[Dict[str, Set[str]], Set[str]]:
    """Return (macro -> files that use it, macros defined)."""
    numbers = paper_dir / "numbers.tex"
    defined: Set[str] = set()
    if numbers.exists():
        text = numbers.read_text()
        defined |= set(MACRO_DEF.findall(text))
        defined |= set(PROVIDE_DEF.findall(text))

    sources = sorted(paper_dir.glob("*.tex")) + sorted(paper_dir.glob("sections/*.tex"))
    used: Dict[str, Set[str]] = {}
    for path in sources:
        if path.name == "numbers.tex":
            continue
        content = path.read_text()
        # Strip comments: a macro inside a comment is not rendered.
        content = re.sub(r"(?<!\\)%.*", "", content)
        # Macros defined inline in this file are self-satisfying.
        defined |= set(MACRO_DEF.findall(content))
        for name in MACRO_USE.findall(content):
            if name in KNOWN_EXTERNAL:
                continue
            used.setdefault(name, set()).add(path.name)
    return used, defined


def main(paper_dir: str = "paper", strict: bool = False) -> int:
    directory = Path(paper_dir)
    used, defined = collect(directory)

    undefined = {name: files for name, files in used.items() if name not in defined}
    unused = sorted(defined - set(used))

    if undefined:
        print("FAIL: the manuscript references macros that are not defined.")
        print("      These render as NOTHING in the PDF -- silent text corruption.\n")
        for name in sorted(undefined):
            print(f"  \\{name:<22} used in: {', '.join(sorted(undefined[name]))}")
        print(f"\n{len(undefined)} undefined macro(s). "
              "Regenerate numbers.tex (make analyze) or fix the prose.")
        return 1

    print(f"OK: all {len(used)} referenced macros are defined.")
    if unused:
        print(f"\nNote: {len(unused)} macro(s) defined but never used. Usually this "
              "means a claim was cut and its number left behind:")
        print("  " + ", ".join(f"\\{name}" for name in unused[:25]))
        if len(unused) > 25:
            print(f"  ... and {len(unused) - 25} more")
        if strict:
            return 1
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    raise SystemExit(main(args[0] if args else "paper", "--strict" in sys.argv))
