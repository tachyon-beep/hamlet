#!/usr/bin/env python3
"""
Unit tests for scripts/no_defaults_lint.py

Tests all violation detection rules, whitelist patterns, and edge cases.
"""

import ast

# Import the linter module
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import no_defaults_lint


class TestASTContext:
    """Test ASTContext dataclass."""

    def test_simple_path(self):
        ctx = no_defaults_lint.ASTContext(module="foo.py", class_name="MyClass", function_name="my_func", variable_name="x")
        assert ctx.to_path() == "foo.py::MyClass::my_func::x"

    def test_nested_class_path(self):
        ctx = no_defaults_lint.ASTContext(module="foo.py", class_name="Outer::Inner", function_name="my_func")
        assert ctx.to_path() == "foo.py::Outer::Inner::my_func"

    def test_module_only(self):
        ctx = no_defaults_lint.ASTContext(module="foo.py")
        assert ctx.to_path() == "foo.py"

    def test_module_and_function(self):
        ctx = no_defaults_lint.ASTContext(module="foo.py", function_name="my_func")
        assert ctx.to_path() == "foo.py::my_func"


class TestWhitelistPattern:
    """Test WhitelistPattern parsing and matching."""

    def test_parse_structural_module_wildcard(self):
        pattern = no_defaults_lint.parse_whitelist_pattern("src/townlet/demo/**:*")
        assert pattern is not None
        assert pattern.is_structural()
        assert pattern.filepath_pattern == "src/townlet/demo/**"
        assert pattern.rule_id is None

    def test_parse_structural_file_wildcard(self):
        pattern = no_defaults_lint.parse_whitelist_pattern("src/townlet/demo/runner.py:*")
        assert pattern is not None
        assert pattern.is_structural()
        assert pattern.filepath_pattern == "src/townlet/demo/runner.py"
        assert pattern.rule_id is None

    def test_parse_structural_class_wildcard(self):
        pattern = no_defaults_lint.parse_whitelist_pattern("src/townlet/agent/networks.py::SimpleQNetwork:*")
        assert pattern is not None
        assert pattern.is_structural()
        assert pattern.filepath_pattern == "src/townlet/agent/networks.py"
        assert pattern.class_pattern == "SimpleQNetwork"
        assert pattern.rule_id is None

    def test_parse_line_based(self):
        pattern = no_defaults_lint.parse_whitelist_pattern("src/townlet/config.py:42:DEF001")
        assert pattern is not None
        assert not pattern.is_structural()
        assert pattern.filepath_pattern == "src/townlet/config.py"
        assert pattern.lineno == 42
        assert pattern.rule_id == "DEF001"

    def test_matches_structural_module_wildcard(self):
        pattern = no_defaults_lint.parse_whitelist_pattern("src/townlet/demo/**:*")
        ctx = no_defaults_lint.ASTContext(module="src/townlet/demo/runner.py", class_name="Runner", function_name="run")
        assert pattern.matches_structural("src/townlet/demo/runner.py", ctx, "DEF001")

    def test_matches_structural_class(self):
        pattern = no_defaults_lint.parse_whitelist_pattern("src/agent.py::MyClass:*")
        ctx = no_defaults_lint.ASTContext(module="src/agent.py", class_name="MyClass", function_name="foo")
        assert pattern.matches_structural("src/agent.py", ctx, "DEF001")

        # Should not match different class
        ctx2 = no_defaults_lint.ASTContext(module="src/agent.py", class_name="OtherClass", function_name="foo")
        assert not pattern.matches_structural("src/agent.py", ctx2, "DEF001")

    def test_matches_line_based(self):
        pattern = no_defaults_lint.parse_whitelist_pattern("src/config.py:42:DEF001")
        assert pattern.matches_line("src/config.py", 42, "DEF001")
        assert not pattern.matches_line("src/config.py", 43, "DEF001")
        assert not pattern.matches_line("src/other.py", 42, "DEF001")
        assert not pattern.matches_line("src/config.py", 42, "DEF002")


class TestViolationDetection:
    """Test detection of various default value patterns."""

    def _scan_code(self, code: str) -> list[tuple[str, int, int, str]]:
        """Helper to scan code and return violations (without context)."""
        tree = ast.parse(code)
        visitor = no_defaults_lint.NoDefaultsVisitor("test.py")
        visitor.visit(tree)
        return [(rule_id, lineno, col, msg) for (rule_id, lineno, col, msg), _ in visitor.violations]

    def test_detect_function_defaults(self):
        code = "def foo(x=10): pass"
        violations = self._scan_code(code)
        assert len(violations) == 1
        assert violations[0][0] == "DEF001"

    def test_detect_multiple_function_defaults(self):
        code = "def foo(x=10, y=20): pass"
        violations = self._scan_code(code)
        assert len(violations) == 1
        assert violations[0][0] == "DEF001"

    def test_no_violation_no_defaults(self):
        code = "def foo(x, y): pass"
        violations = self._scan_code(code)
        assert len(violations) == 0

    def test_detect_lambda_defaults(self):
        code = "f = lambda x=10: x"
        violations = self._scan_code(code)
        assert len(violations) == 1
        assert violations[0][0] == "DEF002"

    def test_detect_logical_or_default(self):
        code = "x = a or b"
        violations = self._scan_code(code)
        assert len(violations) == 1
        assert violations[0][0] == "ASG001"

    def test_detect_ternary_default(self):
        code = "x = a if condition else b"
        violations = self._scan_code(code)
        assert len(violations) == 1
        assert violations[0][0] == "ASG002"

    def test_detect_dict_get_default(self):
        code = "x = config.get('key', default_value)"
        violations = self._scan_code(code)
        assert len(violations) == 1
        assert violations[0][0] == "CALL001"

    def test_detect_dict_setdefault(self):
        code = "x = config.setdefault('key', default_value)"
        violations = self._scan_code(code)
        assert len(violations) == 1
        assert violations[0][0] == "CALL001"

    def test_detect_os_getenv_default(self):
        code = "import os\nx = os.getenv('VAR', 'default')"
        violations = self._scan_code(code)
        assert len(violations) == 1
        assert violations[0][0] == "CALL002"

    def test_detect_call_with_default_kwarg(self):
        code = "field = Field(default='value')"
        violations = self._scan_code(code)
        assert len(violations) == 1
        assert violations[0][0] == "CALL003"

    def test_detect_call_with_default_factory(self):
        code = "field = Field(default_factory=list)"
        violations = self._scan_code(code)
        assert len(violations) == 1
        assert violations[0][0] == "CALL003"

    def test_detect_argparse_default(self):
        code = "parser.add_argument('--foo', default=10)"
        violations = self._scan_code(code)
        # Should detect both CALL003 (general) and ARGP001 (specific)
        assert len(violations) == 2
        rule_ids = {v[0] for v in violations}
        assert "CALL003" in rule_ids
        assert "ARGP001" in rule_ids

    def test_detect_click_default(self):
        code = "@click.option('--foo', default=10)\ndef cmd(): pass"
        violations = self._scan_code(code)
        # Should detect both CALL003 (general) and CLICK001 (specific)
        assert len(violations) == 2
        rule_ids = {v[0] for v in violations}
        assert "CALL003" in rule_ids
        assert "CLICK001" in rule_ids

    def test_nested_class_context(self):
        code = """
class Outer:
    class Inner:
        def foo(self, x=10):
            pass
"""
        tree = ast.parse(code)
        visitor = no_defaults_lint.NoDefaultsVisitor("test.py")
        visitor.visit(tree)
        violations = visitor.violations
        assert len(violations) == 1
        _, context = violations[0]
        # Should capture nested class hierarchy
        assert context.class_name == "Outer::Inner"
        assert context.function_name == "foo"

    def test_tuple_unpacking_variable_extraction(self):
        code = "a, b = foo() or (1, 2)"
        tree = ast.parse(code)
        visitor = no_defaults_lint.NoDefaultsVisitor("test.py")
        visitor.visit(tree)
        violations = visitor.violations
        assert len(violations) == 1
        _, context = violations[0]
        assert context.variable_name == "a,b"

    def test_list_unpacking_variable_extraction(self):
        code = "[x, y] = foo() or [1, 2]"
        tree = ast.parse(code)
        visitor = no_defaults_lint.NoDefaultsVisitor("test.py")
        visitor.visit(tree)
        violations = visitor.violations
        assert len(violations) == 1
        _, context = violations[0]
        assert context.variable_name == "x,y"

    def test_subscript_assignment_variable_extraction(self):
        code = "config['key'] = foo() or 'default'"
        tree = ast.parse(code)
        visitor = no_defaults_lint.NoDefaultsVisitor("test.py")
        visitor.visit(tree)
        violations = visitor.violations
        assert len(violations) == 1
        _, context = violations[0]
        # Should extract base variable name
        assert context.variable_name == "config"


class TestFileScanning:
    """Test file scanning and integration."""

    def test_scan_file_success(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def foo(x=10): pass")
            f.flush()
            path = Path(f.name)

        try:
            violations = no_defaults_lint.scan_file(path)
            assert len(violations) == 1
            rule, context = violations[0]
            assert rule[0] == "DEF001"
        finally:
            path.unlink()

    def test_scan_file_syntax_error(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def foo(x=10")  # Syntax error
            f.flush()
            path = Path(f.name)

        try:
            violations = no_defaults_lint.scan_file(path)
            assert len(violations) == 1
            rule, _ = violations[0]
            assert rule[0] == "PARSE"
        finally:
            path.unlink()

    def test_iter_py_files_with_exclude(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create some test files
            (tmpdir_path / "foo.py").write_text("pass")
            (tmpdir_path / "test_foo.py").write_text("pass")
            (tmpdir_path / "bar.py").write_text("pass")

            # Without exclude
            files = list(no_defaults_lint.iter_py_files(tmpdir_path))
            assert len(files) == 3

            # With exclude
            files = list(no_defaults_lint.iter_py_files(tmpdir_path, exclude_patterns=["test_*.py"]))
            assert len(files) == 2
            assert not any("test_" in str(f) for f in files)


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_main_no_violations(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def foo(x): pass")
            f.flush()
            path = Path(f.name)

        try:
            exit_code = no_defaults_lint.main(["test", str(path)])
            assert exit_code == 0
        finally:
            path.unlink()

    def test_main_with_violations(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def foo(x=10): pass")
            f.flush()
            path = Path(f.name)

        try:
            exit_code = no_defaults_lint.main(["test", str(path)])
            assert exit_code == 1
        finally:
            path.unlink()

    def test_main_with_whitelist(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as code_file:
            code_file.write("def foo(x=10): pass")
            code_file.flush()
            code_path = Path(code_file.name)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as whitelist_file:
            # Structural whitelist matching entire file
            whitelist_file.write(f"{code_path}:*\n")
            whitelist_file.flush()
            whitelist_path = Path(whitelist_file.name)

        try:
            exit_code = no_defaults_lint.main(["test", str(code_path), "--whitelist", str(whitelist_path)])
            assert exit_code == 0  # All violations whitelisted
        finally:
            code_path.unlink()
            whitelist_path.unlink()

    def test_main_file_not_found(self):
        exit_code = no_defaults_lint.main(["test", "/nonexistent/file.py"])
        assert exit_code == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
