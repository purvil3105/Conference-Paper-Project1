"""Pre-compile checks for main.tex. Run: python paper/check_paper.py

Catches the faults the reviewers flagged (broken references, placeholders, missing
figures) plus IEEE style rules. It does not replace compiling and reading the PDF.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
BANNED = ["optimis", "behaviour", "modelling", "signalling", "summaris", "realis",
          "lightweight", "standards-compliant,", "deployment-ready", "real-time",
          "energy-efficient", "outperform", "novel", "ACB factor", "Error!", "[Rxx]",
          "Figure "]


def main():
    raw = (HERE / "main.tex").read_text(encoding="utf-8")
    tex = re.sub(r"(?<!\\)%.*", "", raw)
    body, _, bib = tex.partition(r"\begin{thebibliography}")
    problems = []

    if tex.count("{") != tex.count("}"):
        problems.append(f"unbalanced braces: {tex.count('{')} vs {tex.count('}')}")
    begins = re.findall(r"\\begin\{(\w+\*?)\}", tex)
    ends = re.findall(r"\\end\{(\w+\*?)\}", tex)
    for env in sorted(set(begins + ends)):
        if begins.count(env) != ends.count(env):
            problems.append(f"unmatched environment: {env}")

    labels = re.findall(r"\\label\{([^}]+)\}", tex)
    refs = re.findall(r"\\(?:eq)?ref\{([^}]+)\}", tex)
    problems += [f"undefined reference: {r}" for r in sorted(set(refs) - set(labels))]
    problems += [f"duplicate label: {r}" for r in sorted(set(labels))
                 if labels.count(r) > 1]

    cites = []
    for group in re.findall(r"\\cite\{([^}]+)\}", body):
        for key in group.split(","):
            if key.strip() not in cites:
                cites.append(key.strip())
    items = re.findall(r"\\bibitem\{([^}]+)\}", bib)
    problems += [f"undefined citation: {c}" for c in cites if c not in items]
    problems += [f"reference never cited: {i}" for i in items if i not in cites]
    if [c for c in cites if c in items] != [i for i in items if i in cites]:
        problems.append("references are not in order of first citation")

    for path in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex):
        if not (HERE / path).exists():
            problems.append(f"missing figure file: {path}")

    prose = re.sub(r"\\(emph|textbf|texttt)\{", "{", body)
    for word in BANNED:
        if word in prose:
            problems.append(f"check wording: {word!r}")
    if re.search(r"Fig\.\s*\\ref", body):
        problems.append("use Fig.~\\ref so the number cannot split or lose its space")
    if re.search(r"\[\d+\]\s*[-\u2013]\s*\[\d+\]", body):
        problems.append("IEEE style writes reference ranges out in full")
    if not raw.isascii():
        problems.append("non-ASCII character in main.tex")

    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S).group(1)
    abstract = re.sub(r"\\pending\{[^}]*\}", "", abstract)
    n_abstract = len(abstract.split())
    if n_abstract > 250:
        problems.append(f"abstract has {n_abstract} words; the limit is 250")

    print(f"abstract words (before the pending sentence): {n_abstract}")
    print(f"body words: {len(prose.split())}; references: {len(items)}; "
          f"figures: {tex.count('{figure}') // 2}; tables: {tex.count('{table}') // 2}")
    pending = [" ".join(p.split()) for p in re.findall(r"\\pending\{(.*?)\}\s", body, re.S)]
    print("\nPENDING text (table and figure placeholders are not listed):")
    for item in pending:
        print(f"  - {item[:90]}")
    print(f"\nproblems: {len(problems)}")
    for problem in problems:
        print(f"  - {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
