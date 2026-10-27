"""Recursively check paper sources without invoking a LaTeX compiler.

This conservative check distinguishes math, literal arguments and verbatim
content from prose. It does not replace the final Overleaf compile/layout check.
All dependencies use paths relative to the compilation root, as in main.tex.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCY = re.compile(r"\\(input|include|includegraphics|bibliography|addbibresource)\*?(?:\s*\[[^\]]*\])*\s*\{([^{}]+)\}")
MATH_ENVS = {"math", "displaymath", "equation", "align", "alignat", "gather", "multline", "eqnarray", "flalign"}
LITERAL_COMMANDS = {
    "input", "include", "includegraphics", "bibliography", "bibliographystyle",
    "addbibresource", "label", "ref", "eqref", "autoref", "pageref", "cref", "Cref",
    "cite", "citep", "citet", "nocite", "url", "path", "nolinkurl", "href",
    "detokenize", "documentclass", "usepackage",
}


def blank(text: str) -> str:
    return "".join("\n" if char == "\n" else " " for char in text)


def clean_source(text: str) -> str:
    # Preserve offsets, including escaped percent signs and line-ending \\\\.
    text = re.sub(r"\\begin\{(verbatim\*?|lstlisting|minted|CCSXML|comment)\}[\s\S]*?\\end\{\1\}", lambda m: blank(m[0]), text)
    text = re.sub(r"\\verb\*?([^\w\s])[^\n]*?\1", lambda m: blank(m[0]), text)
    return re.sub(r"\\[\s\S]|%[^\n]*", lambda m: blank(m[0]) if m[0][0] == "%" else m[0], text)


def group_end(text: str, start: int) -> int | None:
    if start >= len(text) or text[start] != "{":
        return None
    depth, index = 0, start
    while index < len(text):
        if text[index] == "\\":
            index += 2
            continue
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return None


def mask_literal_arguments(text: str) -> str:
    chars = list(text)
    for match in re.finditer(r"\\([A-Za-z]+)\*?(?:\s*\[[^\]]*\])*\s*", text):
        if match[1] in LITERAL_COMMANDS:
            end = group_end(text, match.end())
            if end is not None:
                # href hides only the URL; its display text remains checked.
                chars[match.end():end] = blank(text[match.end():end])
    return "".join(chars)


def brace_errors(text: str, name: str) -> list[str]:
    errors, stack = [], []
    for match in re.finditer(r"\\[\s\S]|[{}]", text):
        line = text.count("\n", 0, match.start()) + 1
        if match[0] == "{":
            stack.append(line)
        elif match[0] == "}":
            if stack:
                stack.pop()
            else:
                errors.append(f"{name}:{line}: unmatched closing brace")
    if stack:
        errors.append(f"{name}:{stack[0]}: unclosed brace(s): {len(stack)}")
    return errors


def syntax_errors(text: str, name: str) -> list[str]:
    text = clean_source(text)
    errors = brace_errors(text, name)
    text = mask_literal_arguments(text)
    argument_modes = []
    for command in re.finditer(
        r"\\(ensuremath|text|textrm|textbf|textit|textnormal|mbox|"
        r"newcommand|renewcommand|providecommand|DeclareMathOperator)\s*\{",
        text,
    ):
        end = group_end(text, command.end() - 1)
        if end is not None:
            math_like = command[1] in {
                "ensuremath",
                "newcommand",
                "renewcommand",
                "providecommand",
                "DeclareMathOperator",
            }
            argument_modes.append((command.end(), end - 1, math_like))
            # \newcommand and friends take a second {body} argument whose
            # contents are math when the macro expands; skip it past the check.
            if command[1] in {"newcommand", "renewcommand", "providecommand"}:
                body_end = group_end(text, end)
                if body_end is not None:
                    argument_modes.append((end, body_end - 1, True))
    environments, math_delimiters = [], {}
    tokens = re.finditer(r"\\(?:begin|end)\s*\{[^{}]+\}|\\[A-Za-z]+|\\[\s\S]|\$\$|\$|[_^]", text)
    for match in tokens:
        token, line = match[0], text.count("\n", 0, match.start()) + 1
        active_arguments = [(start, is_math) for start, end, is_math in argument_modes if start <= match.start() < end]
        argument = active_arguments[-1] if active_arguments else None
        context = argument[0] if argument else None
        math_delimiter = math_delimiters.get(context)
        env = re.fullmatch(r"\\(begin|end)\s*\{([^{}]+)\}", token)
        if env:
            kind, value = env.groups()
            if kind == "begin":
                environments.append((value, line))
            elif not environments:
                errors.append(f"{name}:{line}: end of {value} without begin")
            else:
                previous, start_line = environments.pop()
                if previous != value:
                    errors.append(f"{name}:{line}: end of {value} mismatches {previous} at {start_line}")
        elif token in {"$", "$$", r"\(", r"\["}:
            if math_delimiter is None:
                math_delimiters[context] = (token, line)
            elif token == math_delimiter[0] and token in {"$", "$$"}:
                math_delimiters.pop(context)
            else:
                errors.append(f"{name}:{line}: mismatched math delimiter {token}")
        elif token in {r"\)", r"\]"}:
            opening = r"\(" if token == r"\)" else r"\["
            if math_delimiter is None or math_delimiter[0] != opening:
                errors.append(f"{name}:{line}: unmatched math delimiter {token}")
            math_delimiters.pop(context, None)
        elif token in {"_", "^"}:
            in_math = math_delimiter is not None or any(e.rstrip("*") in MATH_ENVS for e, _ in environments)
            if argument:
                in_math = argument[1] or math_delimiter is not None
            if not in_math:
                errors.append(f"{name}:{line}: unescaped '{token}' outside math mode")
    for math_delimiter in math_delimiters.values():
        errors.append(f"{name}:{math_delimiter[1]}: unclosed math delimiter {math_delimiter[0]}")
    if environments:
        errors.append(f"{name}: unclosed environments {environments}")
    return errors


@dataclass
class PaperSources:
    tex: dict[Path, str] = field(default_factory=dict)
    bib: dict[Path, str] = field(default_factory=dict)
    assets: set[Path] = field(default_factory=set)
    errors: list[str] = field(default_factory=list)


def collect_sources(root: Path, main: Path) -> PaperSources:
    """Find the recursive input/graphics/bibliography closure, without Git."""
    root, sources, visiting = root.resolve(), PaperSources(), set()

    def visit(path: Path) -> None:
        path = path.resolve()
        if not path.is_relative_to(root):
            sources.errors.append(f"dependency escapes compilation root: {path.name}")
            return
        if path in visiting:
            sources.errors.append(f"cyclic input: {path.relative_to(root).as_posix()}")
            return
        if path in sources.tex:
            return
        if not path.is_file():
            sources.errors.append(f"missing input: {path.relative_to(root).as_posix()}")
            return
        visiting.add(path)
        text = path.read_text(encoding="utf-8")
        sources.tex[path] = text
        for match in DEPENDENCY.finditer(clean_source(text)):
            kind, argument = match.groups()
            names = argument.split(",") if kind == "bibliography" else [argument]
            for name in names:
                suffixes = [".tex"] if kind in {"input", "include"} else [".bib"] if kind in {"bibliography", "addbibresource"} else [".pdf", ".png", ".jpg", ".jpeg"]
                target = root / name.strip()
                if not target.suffix:
                    target = next((target.with_suffix(s) for s in suffixes if target.with_suffix(s).is_file()), target.with_suffix(suffixes[0]))
                target = target.resolve()
                if not target.is_relative_to(root):
                    sources.errors.append(f"dependency escapes compilation root: {name}")
                elif kind in {"input", "include"}:
                    visit(target)
                elif not target.is_file():
                    sources.errors.append(f"missing {kind}: {name}")
                elif kind in {"bibliography", "addbibresource"}:
                    sources.bib[target] = target.read_text(encoding="utf-8")
                else:
                    sources.assets.add(target)
        visiting.remove(path)

    visit(main if main.is_absolute() else root / main)
    return sources


def validate(root: Path = ROOT, main: Path = Path("paper/main.tex")) -> dict[str, object]:
    root = root.resolve()
    sources, errors = collect_sources(root, main), []
    errors.extend(sources.errors)
    for path, content in sources.tex.items():
        errors.extend(syntax_errors(content, path.relative_to(root).as_posix()))
    for path, content in sources.bib.items():
        name = path.relative_to(root).as_posix()
        errors.extend(brace_errors(clean_source(content), name))
        if any(ord(char) > 127 for char in content):
            errors.append(f"{name}: non-ASCII bibliography characters (use LaTeX escapes)")
    tex = "\n".join(clean_source(text) for text in sources.tex.values())
    bib = "\n".join(sources.bib.values())
    citations = {key.strip() for m in re.finditer(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]*)\}", tex) for key in m[1].split(",")}
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib))
    labels = set(re.findall(r"\\label\{([^}]*)\}", tex))
    refs = set(re.findall(r"\\(?:ref|eqref|autoref|pageref)\{([^}]*)\}", tex))
    if citations - bib_keys:
        errors.append(f"unresolved citations: {sorted(citations - bib_keys)}")
    if refs - labels:
        errors.append(f"unresolved references: {sorted(refs - labels)}")
    return {"tex_file_count": len(sources.tex), "bib_file_count": len(sources.bib),
            "asset_count": len(sources.assets), "citation_count": len(citations),
            "bibliography_entries": len(bib_keys), "reference_count": len(refs),
            "unused_bibliography_entries": sorted(bib_keys - citations), "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--main", type=Path, default=Path("paper/main.tex"))
    args = parser.parse_args()
    report = validate(args.root, args.main)
    print(json.dumps(report, indent=2))
    print("VALIDATION FAILED" if report["errors"] else "VALIDATION PASSED (final Overleaf compile still required)")
    return int(bool(report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
