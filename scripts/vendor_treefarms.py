#!/usr/bin/env python3
"""One-shot vendoring helper: copy treeFARMS C++ sources into
src/shinrin/_spotset/cpp/engine and wrap them in `namespace spotset`.

Rules:
- insert `namespace spotset {` after the last top-of-file #include line
- before each top-level `namespace std {`, close spotset first
- at EOF, close spotset if still open
- skip main.* / python_extension.* (CLI + pybind layer we replace)
"""

import re
import sys
from pathlib import Path

SRC = Path("/var/folders/qd/y59cv0457csc9nm628g7pjv80000gn/T/opencode/treeFarms/src")
DST = Path("src/shinrin/_spotset/cpp/engine")

SKIP = {"main.cpp", "main.hpp", "python_extension.cpp", "python_extension.hpp"}

INCLUDE_RE = re.compile(r"^#include\s")


def wrap(text: str, rel: str) -> str:
    lines = text.split("\n")
    # find last consecutive top-of-file include line (skipping guard/comments)
    last_include = -1
    for i, line in enumerate(lines):
        if INCLUDE_RE.match(line):
            last_include = i
        elif i > 0 and line.startswith(("#ifndef", "#define", "#pragma", "/*", " *", " */", "//")):
            continue
        elif not line.strip():
            continue
    if last_include < 0:
        if rel in ("version.hpp",):
            return text  # macro-only header, nothing to wrap
        print(f"  ** {rel}: no includes, wrapping whole file", file=sys.stderr)
        body = lines
        tail_close = True
    else:
        body = None
        tail_close = False

    out = []
    open_spotset = False
    for i, line in enumerate(lines):
        if line.startswith("namespace std"):
            if open_spotset:
                out.append("}  // namespace spotset")
                out.append("")
                open_spotset = False
            out.append(line)
            continue
        out.append(line)
        if i == last_include:
            out.append("")
            out.append("namespace spotset {")
            open_spotset = True
    if open_spotset or tail_close:
        if not open_spotset:
            out.insert(0, "namespace spotset {")
        out.append("")
        out.append("}  // namespace spotset")
    return "\n".join(out)


def main() -> None:
    files = sorted(SRC.rglob("*"))
    for f in files:
        if f.suffix not in (".hpp", ".cpp"):
            continue
        rel = f.relative_to(SRC)
        if rel.name in SKIP:
            continue
        text = f.read_text()
        text = text.replace("#include <json/json.hpp>", "#include <nlohmann/json.hpp>")
        text = text.replace("#include <tbb/concurrent_unordered_set.h>\n", "")
        wrapped = wrap(text, str(rel))
        dest = DST / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(wrapped)
        print(f"  vendored {rel}")


if __name__ == "__main__":
    main()
