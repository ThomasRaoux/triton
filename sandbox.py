# triton_to_gluon.py
import ast
import inspect
from typing import Union, Optional

GLUON_IMPORT_LINE = "import gluon as gl"


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
        # Preserve non-triton imports; detect if gl already present
        for alias in node.names:
            if alias.name in ("gluon",):
                self.insert_gluon_import = False
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Optional[ast.AST]:
        # Detect existing gl import
        if node.module == "gluon":
            self.insert_gluon_import = False
        return node

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self.generic_visit(node)
        if self.insert_gluon_import:
            gl_import = ast.parse(GLUON_IMPORT_LINE).body[0]
            node.body.insert(0, gl_import)
        return node

    def _rewrite_attr_chain_to_gl(self, node: ast.AST) -> ast.AST:
        # tl.*         -> gl.*
        # triton.*     -> gl.*
        # triton.language.* -> gl.*
        parts = _flatten_attr(node)
        if not parts:
            return node
        if parts[0] == "tl":
            new_parts = ["gl"] + parts[1:]
            return _reconstruct_attr(new_parts)
        if parts[0] == "triton":
            if len(parts) >= 2 and parts[1] == "language":
                new_parts = ["gl"] + parts[2:]
            else:
                new_parts = ["gl"] + parts[1:]
            return _reconstruct_attr(new_parts)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        node = self.generic_visit(node)
        return self._rewrite_attr_chain_to_gl(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        # Rare cases referencing module name directly
        if node.id in ("tl", "triton"):
            return ast.copy_location(ast.Name(id="gl", ctx=ast.Load()), node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        # Decorators: @triton.jit -> @gl.kernel; retain other decorators
        new_decorators = []
        for dec in node.decorator_list:
            if _is_triton_jit_decorator(dec):
                new_decorators.append(
                    ast.Attribute(value=ast.Name(id="gl", ctx=ast.Load()),
                                  attr="kernel", ctx=ast.Load())
                )
            else:
                new_decorators.append(self.visit(dec))
        node.decorator_list = new_decorators

        self.generic_visit(node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        # Handle special Triton calls if signatures differ in Gluon (naive mapping kept)
        node = self.generic_visit(node)

        # Normalize callee attributes to gl.*
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

    # Ensure we have the gl import line at top (simple guard)
    if GLUON_IMPORT_LINE not in out.splitlines()[0:3]:
        out = GLUON_IMPORT_LINE + "\n\n" + out

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