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
    def __init__(self, globals_map: Optional[dict] = None) -> None:
        super().__init__()
        self.insert_gluon_import = True  # Add "import gluon as gl" unless already present
        # Track import aliases found in the parsed source (best-effort)
        self._alias_to_module: dict[str, str] = {}
        self._symbol_to_module: dict[str, str] = {}
        # Globals from the original function (if provided) to resolve real bindings
        self._globals: dict = globals_map or {}
        # Temp counter for generating unique variable names
        self._temp_counter: int = 0
        # Track if we need the dot helper import
        self._need_dot_helper: bool = False

    def visit_Import(self, node: ast.Import) -> Optional[ast.AST]:
        # Preserve imports; mark if gluon import is already present
        for alias in node.names:
            full = alias.name
            asname = alias.asname or full.split(".")[-1]
            self._alias_to_module[asname] = full
            if full.startswith("triton.experimental.gluon"):
                self.insert_gluon_import = False
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Optional[ast.AST]:
        # Detect existing gluon import and record imported symbols
        if node.module and node.module.startswith("triton.experimental.gluon"):
            self.insert_gluon_import = False
        if node.module:
            for alias in node.names:
                name = alias.asname or alias.name
                self._symbol_to_module[name] = f"{node.module}.{alias.name}"
        return node

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self.generic_visit(node)
        if self.insert_gluon_import:
            gl_import_mod = ast.parse(GLUON_IMPORT_LINES).body
            for stmt in reversed(gl_import_mod):
                node.body.insert(0, stmt)
        # Import dot helper if it is needed and not yet imported
        if self._need_dot_helper:
            already = False
            for stmt in node.body:
                if isinstance(stmt, ast.ImportFrom) and stmt.module == "helpers":
                    for alias in stmt.names:
                        if alias.name == "dot_accumulate":
                            already = True
                            break
                if already:
                    break
            if not already:
                helper_import = ast.ImportFrom(module="helpers", names=[ast.alias(name="dot_accumulate", asname=None)], level=0)
                node.body.insert(0, helper_import)
        return node

    def _parts(self, node: ast.AST) -> Optional[list]:
        return _flatten_attr(node)

    def _module_name_of_global(self, name: str) -> str:
        if name not in self._globals:
            return ""
        obj = self._globals[name]
        # If it's a module, __name__ is defined; otherwise, use __module__
        mod = getattr(obj, "__name__", None) or getattr(obj, "__module__", "")
        return mod or ""

    def _is_name_from_module_prefix(self, name: str, module_prefix: str) -> bool:
        mod = self._module_name_of_global(name)
        if mod:
            return mod.startswith(module_prefix)
        # Fallback: use parsed import aliases/symbols
        if name in self._alias_to_module:
            return self._alias_to_module[name].startswith(module_prefix)
        if name in self._symbol_to_module:
            return self._symbol_to_module[name].startswith(module_prefix)
        return False

    def _is_tl_call(self, func: ast.expr, symbol: str) -> bool:
        # Accept forms: tl.symbol(...), triton.language.symbol(...), symbol(...) when imported from triton.language
        if isinstance(func, ast.Attribute):
            parts = self._parts(func)
            if not parts:
                return False
            # alias.module symbol: tl.symbol
            if len(parts) == 2 and self._is_name_from_module_prefix(parts[0], "triton.language") and parts[1] == symbol:
                return True
            # triton.language.symbol
            if (
                len(parts) == 3
                and self._is_name_from_module_prefix(parts[0], "triton")
                and parts[1] == "language"
                and parts[2] == symbol
            ):
                return True
            return False
        if isinstance(func, ast.Name):
            # from triton.language import symbol as name
            if self._is_name_from_module_prefix(func.id, f"triton.language"):
                # name may be the symbol or a re-export; best-effort check that original endswith symbol
                target = self._symbol_to_module.get(func.id, "")
                if not target:
                    # if from globals, module will be like triton.language.core
                    return True
                return target.split(".")[-1] == symbol
        return False

    def _ttgl_attr(self, name: str) -> ast.AST:
        return ast.Attribute(value=ast.Name(id="ttgl", ctx=ast.Load()), attr=name, ctx=ast.Load())

    def _new_temp_name(self, base: str) -> str:
        self._temp_counter += 1
        return f"__{base}_{self._temp_counter}"

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        node = self.generic_visit(node)
        parts = self._parts(node)
        # Map tl.constexpr / triton.language.constexpr -> ttgl.constexpr
        if parts and (
            (len(parts) == 2 and self._is_name_from_module_prefix(parts[0], "triton.language") and parts[1] == "constexpr")
            or (
                len(parts) == 3
                and self._is_name_from_module_prefix(parts[0], "triton")
                and parts[1] == "language"
                and parts[2] == "constexpr"
            )
        ):
            return _reconstruct_attr(["ttgl", "constexpr"])
        # Map common dtypes: tl.float32 -> ttgl.float32, etc.
        DTYPE_ATTRS = {
            "float16", "bfloat16", "float32", "float64",
            "int1", "int8", "int16", "int32", "int64",
            "uint8", "uint16", "uint32", "uint64",
        }
        if parts and len(parts) == 2 and self._is_name_from_module_prefix(parts[0], "triton.language") and parts[1] in DTYPE_ATTRS:
            return _reconstruct_attr(["ttgl", parts[1]])
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        # Do not replace bare names; attribute visitor handles tl.* rewriting
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        # Decorators: @triton.jit -> @gluon.jit; retain other decorators
        new_decorators = []
        for dec in node.decorator_list:
            # Match @triton.jit or @jit imported from triton
            is_jit = False
            if isinstance(dec, ast.Attribute):
                parts = self._parts(dec)
                if parts and len(parts) == 2 and self._is_name_from_module_prefix(parts[0], "triton") and parts[1] == "jit":
                    is_jit = True
                # Fallback string-y check
                if not is_jit and parts == ["triton", "jit"]:
                    is_jit = True
            elif isinstance(dec, ast.Name):
                if self._is_name_from_module_prefix(dec.id, "triton") and (dec.id == "jit" or dec.id in self._symbol_to_module):
                    target = self._symbol_to_module.get(dec.id, "")
                    is_jit = dec.id == "jit" or target.endswith(".jit")
                # Fallback: plain 'jit'
                if not is_jit and dec.id == "jit":
                    is_jit = True
            if is_jit:
                new_decorators.append(
                    ast.Attribute(value=ast.Name(id="gluon", ctx=ast.Load()), attr="jit", ctx=ast.Load())
                )
            else:
                new_decorators.append(self.visit(dec))
        node.decorator_list = new_decorators

        self.generic_visit(node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        if self._is_tl_call(node.func, "arange"):
            return self._handle_tl_arange(node)
        if self._is_tl_call(node.func, "program_id"):
            return self._handle_tl_program_id(node)
        if self._is_tl_call(node.func, "load"):
            return self._handle_tl_load(node)
        if self._is_tl_call(node.func, "store"):
            return self._handle_tl_store(node)
        if self._is_tl_call(node.func, "zeros"):
            return self._handle_tl_zeros(node)
        return node

    def _default_auto_layout_kw(self) -> ast.keyword:
        # layout=ttgl.AutoLayout()
        layout_call = ast.Call(func=self._ttgl_attr("AutoLayout"), args=[], keywords=[])
        return ast.keyword(arg="layout", value=layout_call)

    def _handle_tl_arange(self, node: ast.Call) -> ast.Call:
        new_call = ast.Call(func=self._ttgl_attr("arange"), args=list(node.args), keywords=list(node.keywords))
        has_layout_kw = any(isinstance(kw, ast.keyword) and kw.arg == "layout" for kw in new_call.keywords)
        has_layout_pos = len(new_call.args) >= 3
        if not has_layout_kw and not has_layout_pos:
            new_call.keywords.append(self._default_auto_layout_kw())
        return new_call

    def _handle_tl_program_id(self, node: ast.Call) -> ast.Call:
        return ast.Call(func=self._ttgl_attr("program_id"), args=list(node.args), keywords=list(node.keywords))

    def _handle_tl_load(self, node: ast.Call) -> ast.Call:
        return ast.Call(func=self._ttgl_attr("load"), args=list(node.args), keywords=list(node.keywords))

    def _handle_tl_store(self, node: ast.Call) -> ast.Call:
        return ast.Call(func=self._ttgl_attr("store"), args=list(node.args), keywords=list(node.keywords))

    def _handle_tl_zeros(self, node: ast.Call) -> ast.Call:
        # tl.zeros((M,N), dtype=tl.float32) -> ttgl.zeros([M,N], ttgl.float32, layout=AutoLayout())
        args = list(node.args)
        kwds = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}
        # shape can be first positional or keyword 'shape' in Triton
        shape_expr = None
        if args:
            shape_expr = args[0]
        if kwds.get("shape") is not None:
            shape_expr = kwds["shape"]
        dtype_expr = kwds.get("dtype", None)
        # Convert shape tuple to list for Gluon
        if isinstance(shape_expr, ast.Tuple):
            shape_expr = ast.List(elts=list(shape_expr.elts), ctx=ast.Load())
        elif not isinstance(shape_expr, ast.List):
            # Wrap single dim into list
            shape_expr = ast.List(elts=[shape_expr], ctx=ast.Load())
        layout_kw = ast.keyword(arg="layout", value=ast.Call(func=self._ttgl_attr("AutoLayout"), args=[], keywords=[]))
        new_args = [shape_expr]
        if dtype_expr is not None:
            new_args.append(dtype_expr)
        return ast.Call(func=self._ttgl_attr("zeros"), args=new_args, keywords=[layout_kw])

    def _wrap_dot_operand(self, expr: ast.expr, operand_index: int) -> ast.Call:
        # ttgl.convert_layout(expr, ttgl.DotOperandLayout(operand_index, ttgl.AutoLayout(), 0))
        dot_layout = ast.Call(
            func=self._ttgl_attr("DotOperandLayout"),
            args=[ast.Constant(value=operand_index), ast.Call(func=self._ttgl_attr("AutoLayout"), args=[], keywords=[]), ast.Constant(value=0)],
            keywords=[],
        )
        return ast.Call(func=self._ttgl_attr("convert_layout"), args=[expr, dot_layout], keywords=[])

    def _make_dot_layout_call(self, operand_index: int) -> ast.Call:
        return ast.Call(
            func=self._ttgl_attr("DotOperandLayout"),
            args=[
                ast.Constant(value=operand_index),
                ast.Call(func=self._ttgl_attr("AutoLayout"), args=[], keywords=[]),
                ast.Constant(value=0),
            ],
            keywords=[],
        )

    def _make_dot_layout_annassign(self, name: str, operand_index: int) -> ast.AST:
        return ast.AnnAssign(
            target=ast.Name(id=name, ctx=ast.Store()),
            annotation=self._ttgl_attr("constexpr"),
            value=self._make_dot_layout_call(operand_index),
            simple=1,
        )

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        node = self.generic_visit(node)
        if isinstance(node.op, ast.Add) and isinstance(node.value, ast.Call) and self._is_tl_call(node.value.func, "dot"):
            # acc += tl.dot(a, b) -> acc = dot_accumulate(a, b, acc)
            if isinstance(node.target, ast.Name) and len(node.value.args) >= 2:
                self._need_dot_helper = True
                acc_name = ast.Name(id=node.target.id, ctx=ast.Load())
                helper_call = ast.Call(
                    func=ast.Name(id="dot_accumulate", ctx=ast.Load()),
                    args=[node.value.args[0], node.value.args[1], acc_name],
                    keywords=[],
                )
                final_assign = ast.Assign(targets=[node.target], value=helper_call)
                return final_assign
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        node = self.generic_visit(node)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Add):
            target = node.targets[0].id
            left = node.value.left
            right = node.value.right
            # acc = acc + tl.dot(a,b)
            if isinstance(left, ast.Name) and left.id == target and isinstance(right, ast.Call) and self._is_tl_call(right.func, "dot"):
                if len(right.args) >= 2:
                    self._need_dot_helper = True
                    acc_name = ast.Name(id=target, ctx=ast.Load())
                    helper_call = ast.Call(func=ast.Name(id="dot_accumulate", ctx=ast.Load()), args=[right.args[0], right.args[1], acc_name], keywords=[])
                    return ast.Assign(targets=node.targets, value=helper_call)
            # acc = tl.dot(a,b) + acc
            if isinstance(right, ast.Name) and right.id == target and isinstance(left, ast.Call) and self._is_tl_call(left.func, "dot"):
                if len(left.args) >= 2:
                    self._need_dot_helper = True
                    acc_name = ast.Name(id=target, ctx=ast.Load())
                    helper_call = ast.Call(func=ast.Name(id="dot_accumulate", ctx=ast.Load()), args=[left.args[0], left.args[1], acc_name], keywords=[])
                    return ast.Assign(targets=node.targets, value=helper_call)
        return node


def convert_triton_to_gluon(func_or_src: Union[str, object]) -> str:
    src = get_source(func_or_src)
    tree = ast.parse(src)

    # Provide globals map if we were given a live function
    globals_map = None
    if not isinstance(func_or_src, str):
        globals_map = getattr(func_or_src, "__globals__", None)

    transformer = TritonToGluonTransformer(globals_map)
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