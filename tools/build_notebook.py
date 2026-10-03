"""Convert the percent-format source (src/shrinkage_frontier.py) into a Colab notebook.

Usage: python tools/build_notebook.py
Cells are delimited by '# %%' (code) and '# %% [markdown]' (markdown). Lines starting with
'# COLAB: ' are uncommented so shell/magic commands run in Colab but not in plain Python.
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "shrinkage_frontier.py"
DST = ROOT / "notebooks" / "shrinkage_frontier.ipynb"


def parse_cells(text: str):
    cells, kind, buf = [], None, []

    def flush():
        body = "\n".join(buf).strip("\n")
        if kind and body:
            cells.append((kind, body))

    for line in text.splitlines():
        if line.startswith("# %% [markdown]"):
            flush(); kind, buf = "markdown", []
        elif line.startswith("# %%"):
            flush(); kind, buf = "code", []
        elif kind == "markdown":
            buf.append(line[2:] if line.startswith("# ") else line.lstrip("#"))
        else:
            buf.append(line.replace("# COLAB: ", "", 1) if line.startswith("# COLAB: ") else line)
    flush()
    return cells


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata.update({
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True},
    })
    for kind, body in parse_cells(SRC.read_text(encoding="utf-8")):
        nb.cells.append(nbf.v4.new_markdown_cell(body) if kind == "markdown" else nbf.v4.new_code_cell(body))
    DST.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, DST)
    print(f"Wrote {DST} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
