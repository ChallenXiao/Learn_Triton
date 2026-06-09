#!/usr/bin/env python3
"""把 Markdown 教程源文件转换为 Jupyter Notebook (.ipynb)。

规则:
- ```python 围栏代码块 -> code cell(会被执行)
- 其余内容            -> markdown cell
- markdown 中如需展示不执行的代码,用 ```py / ```text 等其他围栏语言标记

用法: python3 tools/md2nb.py sources/01_xxx.md notebooks/01_xxx.ipynb
"""
import re
import sys

import nbformat as nbf

CODE_FENCE = re.compile(r"^```python\s*$")
FENCE_END = re.compile(r"^```\s*$")


def md_to_cells(text: str):
    cells = []
    lines = text.splitlines()
    buf, in_code = [], False

    def flush(kind):
        content = "\n".join(buf).strip("\n")
        if content.strip():
            if kind == "code":
                cells.append(nbf.v4.new_code_cell(content))
            else:
                cells.append(nbf.v4.new_markdown_cell(content))
        buf.clear()

    for line in lines:
        if not in_code and CODE_FENCE.match(line):
            flush("markdown")
            in_code = True
        elif in_code and FENCE_END.match(line):
            flush("code")
            in_code = False
        else:
            buf.append(line)
    flush("code" if in_code else "markdown")
    return cells


def main():
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, encoding="utf-8") as f:
        text = f.read()
    nb = nbf.v4.new_notebook()
    nb.cells = md_to_cells(text)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    # Colab 徽章需要的元数据
    nb.metadata["colab"] = {"provenance": []}
    nbf.validate(nb)
    with open(dst, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"OK: {src} -> {dst} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
