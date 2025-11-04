#!/usr/bin/env python3
"""
No-Defaults Linter with Structural Whitelist Support

Enforces the "no default variables unless whitelisted" rule.
Detects defaults in function parameters, assignments, and common framework calls.

Whitelist Format (supports both structural and line-based):

    Structural (recommended - stable across refactors):
        <filepath>::<class>::<function>::<variable>:<rule_id>

    Examples:
        # Whitelist entire module
        src/townlet/recording/**:*

        # Whitelist entire file
        src/townlet/demo/runner.py:*

        # Whitelist all defaults in a class
        src/townlet/agent/networks.py::SimpleQNetwork:*

        # Whitelist specific function
        src/townlet/environment/cascade_engine.py::apply_base_depletions:DEF001

        # Whitelist specific field in class
        src/townlet/training/state.py::BatchedAgentState::description:CALL003

    Line-based (legacy - fragile, use only when necessary):
        <filepath>:<lineno>:<rule_id>

Wildcards:
    * matches anything
    ** matches path components (for modules)
"""
import argparse
import ast
import fnmatch
import pathlib
import sys
from dataclasses import dataclass

Rule = tuple[str, int, int, str]  # (rule_id, lineno, col, message)

DEFAULT_KEYWORDS = {"default", "default_factory"}


@dataclass
class ASTContext:
    """Track AST context (class, function, variable) for structural matching."""

    module: str  # filepath
    class_name: str | None = None
    function_name: str | None = None
    variable_name: str | None = None

    def to_path(self) -> str:
        """Convert to structural path: module::class::function::variable"""
        parts = [self.module]
        if self.class_name:
            parts.append(self.class_name)
        if self.function_name:
            parts.append(self.function_name)
        if self.variable_name:
            parts.append(self.variable_name)
        return "::".join(parts)


@dataclass
class WhitelistPattern:
    """Represents a whitelist pattern (structural or line-based)."""

    pattern: str  # Original pattern string
    rule_id: str | None = None  # None means all rules

    # For structural patterns
    filepath_pattern: str | None = None
    class_pattern: str | None = None
    function_pattern: str | None = None
    variable_pattern: str | None = None

    # For line-based patterns
    lineno: int | None = None

    def is_structural(self) -> bool:
        """Check if this is a structural pattern (not line-based)."""
        return self.lineno is None

    def matches_structural(self, filepath: str, context: ASTContext, rule_id: str) -> bool:
        """Check if structural pattern matches the violation."""
        if not self.is_structural():
            return False

        # Check rule_id
        if self.rule_id and self.rule_id != "*" and self.rule_id != rule_id:
            return False

        # Check filepath
        if self.filepath_pattern:
            # Support ** for directory wildcards
            if "**" in self.filepath_pattern:
                pattern = self.filepath_pattern.replace("**", "*")
                if not fnmatch.fnmatch(filepath, pattern):
                    return False
            elif not fnmatch.fnmatch(filepath, self.filepath_pattern):
                return False

        # Check class
        if self.class_pattern and self.class_pattern != "*":
            if not context.class_name:
                return False
            if not fnmatch.fnmatch(context.class_name, self.class_pattern):
                return False

        # Check function
        if self.function_pattern and self.function_pattern != "*":
            if not context.function_name:
                return False
            if not fnmatch.fnmatch(context.function_name, self.function_pattern):
                return False

        # Check variable
        if self.variable_pattern and self.variable_pattern != "*":
            if not context.variable_name:
                return False
            if not fnmatch.fnmatch(context.variable_name, self.variable_pattern):
                return False

        return True

    def matches_line(self, filepath: str, lineno: int, rule_id: str) -> bool:
        """Check if line-based pattern matches the violation."""
        if self.is_structural():
            return False

        if filepath != self.filepath_pattern:
            return False
        if lineno != self.lineno:
            return False
        if self.rule_id and self.rule_id != rule_id:
            return False

        return True


class NoDefaultsVisitor(ast.NodeVisitor):
    def __init__(self, filename: str):
        self.filename = filename
        self.violations: list[tuple[Rule, ASTContext]] = []

        # Track AST context
        self.class_stack: list[str] = []
        self.function_stack: list[str] = []

    def _current_context(self, variable_name: str | None = None) -> ASTContext:
        """Get current AST context."""
        return ASTContext(
            module=self.filename,
            class_name="::".join(self.class_stack) if self.class_stack else None,
            function_name=self.function_stack[-1] if self.function_stack else None,
            variable_name=variable_name,
        )

    def _report(self, node: ast.AST, rule_id: str, msg: str, variable_name: str | None = None):
        lineno = getattr(node, "lineno", 1)
        col = getattr(node, "col_offset", 0)
        context = self._current_context(variable_name)
        self.violations.append(((rule_id, lineno, col, msg), context))

    def _has_default_kwargs(self, keywords: list[ast.keyword]) -> str | None:
        for kw in keywords:
            if kw.arg in DEFAULT_KEYWORDS:
                return kw.arg
        return None

    def _is_os_getenv(self, node: ast.Call) -> bool:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "getenv":
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "os":
                return True
        return False

    # Track class context
    def visit_ClassDef(self, node: ast.ClassDef):
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    # Track function context
    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.function_stack.append(node.name)
        self._check_args_defaults(node, "DEF001")
        self.generic_visit(node)
        self.function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.function_stack.append(node.name)
        self._check_args_defaults(node, "DEF001")
        self.generic_visit(node)
        self.function_stack.pop()

    def visit_Lambda(self, node: ast.Lambda):
        args = node.args
        if (args.defaults and any(d is not None for d in args.defaults)) or (
            args.kw_defaults and any(d is not None for d in args.kw_defaults)
        ):
            self._report(node, "DEF002", "Lambda has parameter default(s)")
        self.generic_visit(node)

    def _check_args_defaults(self, node, rule_id: str):
        args = node.args
        has_pos_kw_defaults = (args.defaults and any(d is not None for d in args.defaults)) or (
            args.kw_defaults and any(d is not None for d in args.kw_defaults)
        )
        if has_pos_kw_defaults:
            self._report(node, rule_id, f"Function '{node.name}' has parameter default(s)")

    # Track assignment targets for variable names
    def visit_Assign(self, node: ast.Assign):
        var_name = self._extract_target_name(node.targets[0]) if node.targets else None
        self._check_assignment_value(node.value, var_name)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        var_name = self._extract_target_name(node.target)
        if node.value is not None:
            self._check_assignment_value(node.value, var_name)
        self.generic_visit(node)

    def _extract_target_name(self, target: ast.AST) -> str | None:
        """Extract variable name from assignment target."""
        if isinstance(target, ast.Name):
            return target.id
        elif isinstance(target, ast.Attribute):
            return target.attr
        elif isinstance(target, (ast.Tuple, ast.List)):
            # Handle tuple/list unpacking: a, b = foo()
            names = []
            for elt in target.elts:
                name = self._extract_target_name(elt)
                if name:
                    names.append(name)
            return ",".join(names) if names else None
        elif isinstance(target, ast.Subscript):
            # Handle subscript assignment: x[0] = foo()
            return self._extract_target_name(target.value)
        return None

    def _check_assignment_value(self, value: ast.AST, var_name: str | None = None):
        if isinstance(value, ast.BoolOp) and isinstance(value.op, ast.Or):
            self._report(value, "ASG001", "Logical OR used as a default (x = a or b)", var_name)
        if isinstance(value, ast.IfExp):
            self._report(value, "ASG002", "Ternary expression used as a default (x = a if cond else b)", var_name)

    def visit_Call(self, node: ast.Call):
        # dict.get(key, default) / dict.setdefault(key, default)
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"get", "setdefault"}:
            if len(node.args) >= 2:
                self._report(node, "CALL001", f"'{node.func.attr}(..., default)' provides a default")

        # os.getenv(NAME, default)
        if self._is_os_getenv(node) and len(node.args) >= 2:
            self._report(node, "CALL002", "os.getenv with default value")

        # Any call using default= or default_factory=
        kw = self._has_default_kwargs(node.keywords)
        if kw:
            self._report(node, "CALL003", f"Call provides '{kw}='")

        # Framework-specific hints
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "add_argument":
                if any(k.arg == "default" for k in node.keywords):
                    self._report(node, "ARGP001", "argparse add_argument(default=...)")
            if node.func.attr == "option":
                if any(k.arg == "default" for k in node.keywords):
                    self._report(node, "CLICK001", "click.option(default=...)")

        self.generic_visit(node)


def parse_whitelist_pattern(line: str) -> WhitelistPattern | None:
    """
    Parse whitelist pattern (structural or line-based).

    Structural: filepath[::class][::function][::variable]:rule_id
    Line-based: filepath:lineno:rule_id
    """
    parts = line.split(":")
    if len(parts) < 2:
        return None

    # Check if line-based (second part is a number)
    try:
        lineno = int(parts[-2])
        # Line-based: filepath:lineno:rule_id
        filepath = ":".join(parts[:-2])
        rule_id = parts[-1] if parts[-1] else None
        return WhitelistPattern(
            pattern=line,
            rule_id=rule_id,
            filepath_pattern=filepath,
            lineno=lineno,
        )
    except (ValueError, IndexError):
        pass

    # Structural pattern
    rule_id = parts[-1] if parts[-1] and parts[-1] != "*" else None
    path_part = ":".join(parts[:-1])

    # Split by ::
    components = path_part.split("::")

    pattern = WhitelistPattern(
        pattern=line,
        rule_id=rule_id,
        filepath_pattern=components[0] if len(components) >= 1 else None,
        class_pattern=components[1] if len(components) >= 2 else None,
        function_pattern=components[2] if len(components) >= 3 else None,
        variable_pattern=components[3] if len(components) >= 4 else None,
    )

    return pattern


def load_whitelist(whitelist_path: pathlib.Path) -> list[WhitelistPattern]:
    """Load whitelist patterns from file."""
    patterns: list[WhitelistPattern] = []

    if not whitelist_path.exists():
        print(f"Warning: Whitelist file not found: {whitelist_path}", file=sys.stderr)
        return patterns

    try:
        with whitelist_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()

                # Skip comments and blank lines
                if not line or line.startswith("#"):
                    continue

                pattern = parse_whitelist_pattern(line)
                if pattern:
                    patterns.append(pattern)
                else:
                    print(f"Warning: Invalid whitelist entry at line {line_num}: {line}", file=sys.stderr)

    except Exception as e:
        print(f"Error reading whitelist: {e}", file=sys.stderr)
        return []

    return patterns


def scan_file(path: pathlib.Path) -> list[tuple[Rule, ASTContext]]:
    try:
        source = path.read_text(encoding="utf-8")
    except Exception as e:
        context = ASTContext(module=str(path))
        return [(("IOERR", 1, 0, f"Failed to read: {e}"), context)]
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        context = ASTContext(module=str(path))
        return [(("PARSE", e.lineno or 1, e.offset or 0, f"SyntaxError: {e.msg}"), context)]
    v = NoDefaultsVisitor(str(path))
    v.visit(tree)
    return v.violations


def iter_py_files(root: pathlib.Path, exclude_patterns: list[str] | None = None):
    """Iterate over Python files, optionally excluding patterns."""
    exclude_patterns = exclude_patterns or []

    def is_excluded(path: pathlib.Path) -> bool:
        """Check if path matches any exclude pattern."""
        path_str = str(path)
        for pattern in exclude_patterns:
            # Support both glob-style and path-based patterns
            if fnmatch.fnmatch(path_str, pattern):
                return True
            if fnmatch.fnmatch(path_str, f"*/{pattern}"):
                return True
            if fnmatch.fnmatch(path_str, f"**/{pattern}"):
                return True
        return False

    if root.is_file() and root.suffix == ".py":
        if not is_excluded(root):
            yield root
        return

    for p in root.rglob("*.py"):
        # Skip common vendor dirs
        parts = {part.lower() for part in p.parts}
        if {"venv", ".venv", ".tox", "site-packages"} & parts:
            continue

        # Skip excluded patterns
        if is_excluded(p):
            continue

        yield p


def is_whitelisted(
    filepath: str,
    lineno: int,
    rule_id: str,
    context: ASTContext,
    patterns: list[WhitelistPattern],
) -> bool:
    """Check if violation is whitelisted."""
    for pattern in patterns:
        if pattern.is_structural():
            if pattern.matches_structural(filepath, context, rule_id):
                return True
        else:
            if pattern.matches_line(filepath, lineno, rule_id):
                return True
    return False


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Lint Python code for default values (unless whitelisted)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Whitelist Format (supports both structural and line-based):

  Structural (recommended - stable across refactors):
    <filepath>::<class>::<function>::<variable>:<rule_id>

  Examples:
    # Whitelist entire module
    src/townlet/recording/**:*

    # Whitelist entire file
    src/townlet/demo/runner.py:*

    # Whitelist all defaults in a class
    src/townlet/agent/networks.py::SimpleQNetwork:*

    # Whitelist specific function
    src/townlet/environment/cascade_engine.py::apply_base_depletions:DEF001

    # Whitelist specific field in class
    src/townlet/training/state.py::BatchedAgentState::description:CALL003

  Line-based (legacy - fragile):
    <filepath>:<lineno>:<rule_id>

Rule IDs:
  DEF001  - Function parameter defaults
  DEF002  - Lambda parameter defaults
  ASG001  - Logical OR as default (x = a or b)
  ASG002  - Ternary as default (x = a if cond else b)
  CALL001 - dict.get/setdefault with default
  CALL002 - os.getenv with default
  CALL003 - Call with default= or default_factory=
  ARGP001 - argparse add_argument(default=...)
  CLICK001- click.option(default=...)
        """,
    )
    parser.add_argument("paths", nargs="+", help="Python files or directories to scan")
    parser.add_argument("--whitelist", type=pathlib.Path, help="Path to whitelist file")
    parser.add_argument("--show-whitelisted", action="store_true", help="Show whitelisted violations (for debugging)")
    parser.add_argument("--show-context", action="store_true", help="Show AST context (class/function) for violations")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument("--exclude", action="append", help="Exclude files matching pattern (can be used multiple times)")

    args = parser.parse_args(argv[1:])

    # Load whitelist
    patterns: list[WhitelistPattern] = []
    if args.whitelist:
        patterns = load_whitelist(args.whitelist)
        if patterns:
            structural_count = sum(1 for p in patterns if p.is_structural())
            line_count = len(patterns) - structural_count
            print(f"Loaded {len(patterns)} whitelist pattern(s) from {args.whitelist}")
            if structural_count:
                print(f"  - {structural_count} structural pattern(s)")
            if line_count:
                print(f"  - {line_count} line-based pattern(s)")

    # Scan files
    all_violations: list[tuple[str, str, int, int, str, ASTContext]] = []
    whitelisted_violations: list[tuple[str, str, int, int, str, ASTContext]] = []

    # Collect all files first to show progress
    all_files = []
    exclude_patterns = args.exclude or []
    for path_arg in args.paths:
        root = pathlib.Path(path_arg)
        if not root.exists():
            print(f"Not found: {root}", file=sys.stderr)
            return 2
        all_files.extend(iter_py_files(root, exclude_patterns))

    total_files = len(all_files)
    if not args.quiet and total_files > 0:
        print(f"Scanning {total_files} Python file(s)...", file=sys.stderr)

    for idx, file in enumerate(all_files, start=1):
        if not args.quiet and total_files > 10:
            # Show progress for large scans
            if idx == 1 or idx % 10 == 0 or idx == total_files:
                print(f"Progress: {idx}/{total_files} files scanned", file=sys.stderr, end="\r")

        for (rule_id, lineno, col, msg), context in scan_file(file):
            filepath = str(file)
            violation = (filepath, rule_id, lineno, col, msg, context)

            # Check if whitelisted
            if is_whitelisted(filepath, lineno, rule_id, context, patterns):
                whitelisted_violations.append(violation)
            else:
                all_violations.append(violation)

    if not args.quiet and total_files > 10:
        print(file=sys.stderr)  # New line after progress

    # Report results
    if args.show_whitelisted and whitelisted_violations:
        print("\n=== Whitelisted Violations (not counted as errors) ===")
        for fname, rule_id, lineno, col, msg, context in sorted(whitelisted_violations):
            if args.show_context:
                print(f"{fname}:{lineno}:{col}: {rule_id}: {msg} [WHITELISTED]")
                print(f"  Context: {context.to_path()}")
            else:
                print(f"{fname}:{lineno}:{col}: {rule_id}: {msg} [WHITELISTED]")

    if all_violations:
        print("\n=== Non-Whitelisted Violations ===")
        for fname, rule_id, lineno, col, msg, context in sorted(all_violations):
            if args.show_context:
                print(f"{fname}:{lineno}:{col}: {rule_id}: {msg}")
                print(f"  Context: {context.to_path()}")
            else:
                print(f"{fname}:{lineno}:{col}: {rule_id}: {msg}")
        print(f"\n{len(all_violations)} violation(s) found.")

        if patterns:
            print(f"({len(whitelisted_violations)} violation(s) whitelisted)")

        return 1
    else:
        print("No defaults detected. Clean.")
        if whitelisted_violations:
            print(f"({len(whitelisted_violations)} violation(s) whitelisted)")
        return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
