# triton_to_gluon.py
import ast
import inspect
from typing import Union, Optional

GLUON_IMPORT_LINES = (
    "from triton.experimental import gluon\n"
    "from triton.experimental.gluon import language as ttgl"
)


def get_source(func_or_src: Union[str, object]) -> str:
    if isinstance(func_or_src, str):
        return func_or_src
    return inspect.getsource(func_or_src)


def unparse(tree: ast.AST) -> str:
    # Python 3.9+ required
    return ast.unparse(tree)


def _flatten_attr(node: ast.AST) -> Optional[list]:
    # Converts nested attributes into ["base", "a", "b"] for base.a.b
    parts = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.insert(0, cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.insert(0, cur.id)
        return parts
    return None


def _reconstruct_attr(parts: list) -> ast.AST:
    # Rebuilds AST from ["base", "a", "b"] -> base.a.b
    assert len(parts) >= 1
    node: ast.AST = ast.Name(id=parts[0], ctx=ast.Load())
    for attr in parts[1:]:
        node = ast.Attribute(value=node, attr=attr, ctx=ast.Load())
    return node


def _is_tl_constexpr_annotation(node: ast.expr) -> bool:
    parts = _flatten_attr(node)
    if not parts:
        return False
    # Matches tl.constexpr or triton.language.constexpr
    return (
        parts == ["tl", "constexpr"]
        or parts == ["triton", "language", "constexpr"]
    )


def _is_triton_jit_decorator(node: ast.expr) -> bool:
    parts = _flatten_attr(node)
    if not parts:
        return False
    # Matches @triton.jit or @jit (if imported from triton)
    return parts == ["triton", "jit"] or parts == ["jit"]


class TritonToGluonTransformer(ast.NodeTransformer):
    def __init__(self) -> None:
        super().__init__()
        self.insert_gluon_import = True  # Add "import gluon as gl" unless already present

    def visit_Import(self, node: ast.Import) -> Optional[ast.AST]:
        # Preserve imports; mark if gluon import is already present
        for alias in node.names:
            if alias.name.startswith("triton.experimental.gluon"):
                self.insert_gluon_import = False
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Optional[ast.AST]:
        # Detect existing gluon import
        if node.module and node.module.startswith("triton.experimental.gluon"):
            self.insert_gluon_import = False
        return node

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self.generic_visit(node)
        if self.insert_gluon_import:
            gl_import_mod = ast.parse(GLUON_IMPORT_LINES).body
            for stmt in reversed(gl_import_mod):
                node.body.insert(0, stmt)
        return node

    def _rewrite_attr_chain_to_gl(self, node: ast.AST) -> ast.AST:
        # tl.* and triton.language.* -> ttgl.* (language namespace)
        # Preserve tl.constexpr as-is
        parts = _flatten_attr(node)
        if not parts:
            return node
        # Map tl.constexpr / triton.language.constexpr -> ttgl.constexpr
        if parts == ["tl", "constexpr"] or parts == ["triton", "language", "constexpr"]:
            return _reconstruct_attr(["ttgl", "constexpr"])
        if parts[0] == "tl":
            new_parts = ["ttgl"] + parts[1:]
            return _reconstruct_attr(new_parts)
        if parts[0] == "triton":
            if len(parts) >= 2 and parts[1] == "language":
                new_parts = ["ttgl"] + parts[2:]
                return _reconstruct_attr(new_parts)
            # other triton.* are not language calls; leave unchanged
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        node = self.generic_visit(node)
        return self._rewrite_attr_chain_to_gl(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        # Do not replace bare names; attribute visitor handles tl.* rewriting
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        # Decorators: @triton.jit -> @gluon.jit; retain other decorators
        new_decorators = []
        for dec in node.decorator_list:
            if _is_triton_jit_decorator(dec):
                new_decorators.append(
                    ast.Attribute(value=ast.Name(id="gluon", ctx=ast.Load()),
                                  attr="jit", ctx=ast.Load())
                )
            else:
                new_decorators.append(self.visit(dec))
        node.decorator_list = new_decorators

        self.generic_visit(node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        # Handle special Triton calls if signatures differ in Gluon (naive mapping kept)
        node = self.generic_visit(node)

        # Normalize callee attributes to ttgl.*
        target = self._rewrite_attr_chain_to_gl(node.func)
        node.func = target

        # Example: If you need to reorder tl.load/store args for Gluon, do it here
        # e.g., gl.load(ptr, mask) from tl.load(ptr, mask=mask)
        # This prototype leaves keywords intact for a naive 1:1 mapping.

        return node


def convert_triton_to_gluon(func_or_src: Union[str, object]) -> str:
    src = get_source(func_or_src)
    tree = ast.parse(src)

    transformer = TritonToGluonTransformer()
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)

    out = unparse(new_tree)

    # Ensure we have the gluon import lines at top (simple guard)
    first_lines = "\n".join(out.splitlines()[0:5])
    if "triton.experimental.gluon" not in first_lines:
        out = GLUON_IMPORT_LINES + "\n\n" + out

    return out


# Example/demo usage
if __name__ == "__main__":
    example = """
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
"""

    print(convert_triton_to_gluon(example))