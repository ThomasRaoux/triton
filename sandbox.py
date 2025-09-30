# triton_to_gluon.py
import ast
import inspect
from typing import Union, Optional

GLUON_IMPORT_LINES = ("from triton.experimental import gluon\n"
                      "from triton.experimental.gluon import language as ttgl")


def get_source(func_or_src: Union[str, object]) -> str:
    if isinstance(func_or_src, str):
        return func_or_src
    return inspect.getsource(func_or_src)


def unparse(tree: ast.AST) -> str:
    # Python 3.9+ required
    return ast.unparse(tree)


def _is_supported_external(obj: object) -> bool:
    # Only allow external functions that are @triton.jit or @gluon.constexpr_function
    # Inspect source decorators to decide.
    try:
        src = inspect.getsource(obj)
        tree = ast.parse(src)
        for n in tree.body:
            if isinstance(n, ast.FunctionDef):
                for dec in n.decorator_list:
                    # triton.jit or jit
                    if isinstance(dec, ast.Attribute) and isinstance(dec.value, ast.Name):
                        if dec.value.id == "triton" and dec.attr == "jit":
                            return True
                        if dec.value.id == "gluon" and dec.attr == "constexpr_function":
                            return True
                    if isinstance(dec, ast.Name) and dec.id in ("jit", "constexpr_function"):
                        return True
        return False
    except Exception:
        # Best-effort: detect Triton JIT wrapper objects exposing .fn
        if hasattr(obj, "fn") and inspect.isfunction(getattr(obj, "fn")):
            return True
        return False


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
    return (parts == ["tl", "constexpr"] or parts == ["triton", "language", "constexpr"])


def _is_triton_jit_decorator(node: ast.expr) -> bool:
    parts = _flatten_attr(node)
    if not parts:
        return False
    # Matches @triton.jit or @jit (if imported from triton)
    return parts == ["triton", "jit"] or parts == ["jit"]


class TritonToGluonTransformer(ast.NodeTransformer):

    def __init__(self, globals_map: Optional[dict] = None, convert_only_names: Optional[set[str]] = None,
                 external_attr_to_local: Optional[set[tuple[str, str]]] = None) -> None:
        super().__init__()
        self.insert_gluon_import = True  # Add "import gluon as gl" unless already present
        # Track import aliases found in the parsed source (best-effort)
        self._alias_to_module: dict[str, str] = {}
        self._symbol_to_module: dict[str, str] = {}
        # Globals from the original function (if provided) to resolve real bindings
        self._globals: dict = globals_map or {}
        # Temp counter for generating unique variable names
        self._temp_counter: int = 0
        # Function conversion scoping
        self._convert_only: Optional[set[str]] = convert_only_names
        self._current_function: Optional[str] = None
        # External calls referenced as alias.attr that should be rewritten to local attr
        self._external_attr_to_local: set[tuple[str, str]] = external_attr_to_local or set()

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
        # Always import helpers unconditionally
        node.body.insert(0, ast.ImportFrom(module="helpers", names=[ast.alias(name="tl_arange", asname=None)], level=0))
        node.body.insert(0, ast.ImportFrom(module="helpers", names=[ast.alias(name="tl_zeros", asname=None)], level=0))
        node.body.insert(0, ast.ImportFrom(module="helpers", names=[ast.alias(name="tl_full", asname=None)], level=0))
        node.body.insert(0, ast.ImportFrom(module="helpers", names=[ast.alias(name="descriptor_load", asname=None)], level=0))
        node.body.insert(0, ast.ImportFrom(module="helpers", names=[ast.alias(name="descriptor_store", asname=None)], level=0))
        node.body.insert(0, ast.ImportFrom(module="helpers", names=[ast.alias(name="default_blocked_layout", asname=None)], level=0))
        node.body.insert(0, ast.ImportFrom(module="helpers", names=[ast.alias(name="dot_accumulate", asname=None)], level=0))
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
            # General rule: if the root name resolves to a module from triton.language,
            # and the last attribute matches the target symbol, consider it a TL call.
            # This covers tl.symbol, tl.submodule.symbol, triton.language.symbol, t.language.sub.symbol, etc.
            if len(parts) >= 2 and parts[-1] == symbol:
                if self._is_name_from_module_prefix(parts[0], "triton.language"):
                    return True
                # Also accept alias to triton itself with explicit .language prefix
                if len(parts) >= 3 and self._is_name_from_module_prefix(parts[0], "triton") and parts[1] == "language":
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
        # Only rewrite inside selected functions (if scoping is enabled)
        if self._convert_only is not None and self._current_function not in self._convert_only:
            return node
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
        # Track current function for scoping
        prev_fn = self._current_function
        self._current_function = node.name
        try:
            should_convert = self._convert_only is None or node.name in self._convert_only
            # Decorators: @triton.jit -> @gluon.jit; retain other decorators
            if should_convert:
                new_decorators = []
                for dec in node.decorator_list:
                    # Drop any @triton.language._tensor_member_fn or aliases to it
                    drop = False
                    if isinstance(dec, ast.Attribute):
                        parts = self._parts(dec)
                        if parts and len(parts) >= 2 and parts[-1] == "_tensor_member_fn":
                            # Resolve root against triton.language or triton . language
                            if self._is_name_from_module_prefix(parts[0], "triton.language"):
                                drop = True
                            if len(parts) >= 3 and self._is_name_from_module_prefix(parts[0], "triton") and parts[1] == "language":
                                drop = True
                    elif isinstance(dec, ast.Name):
                        target = self._symbol_to_module.get(dec.id, "")
                        if target.endswith("._tensor_member_fn"):
                            drop = True
                    if drop:
                        continue
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
            # Visit body
            self.generic_visit(node)
            return node
        finally:
            self._current_function = prev_fn

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        # Rewrite alias.func(...) to func(...) when marked as external callee
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            pair = (node.func.value.id, node.func.attr)
            if pair in self._external_attr_to_local:
                return ast.Call(func=ast.Name(id=node.func.attr, ctx=ast.Load()), args=list(node.args), keywords=list(node.keywords))
        # Only rewrite inside selected functions (if scoping is enabled)
        if self._convert_only is not None and self._current_function not in self._convert_only:
            return node
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
        if self._is_tl_call(node.func, "full"):
            return self._handle_tl_full(node)
        if self._is_tl_call(node.func, "cdiv"):
            return self._handle_cdiv(node)
        if self._is_tl_call(node.func, "dot"):
            return self._handle_tl_dot(node)
        # tl.make_tensor_descriptor -> make_descriptor helper
        if self._is_tl_call(node.func, "make_tensor_descriptor"):
            return ast.Call(func=ast.Name(id="make_descriptor", ctx=ast.Load()), args=list(node.args), keywords=list(node.keywords))
        # desc.load(...) / desc.store(...)
        if isinstance(node.func, ast.Attribute) and node.func.attr in ("load", "store"):
            base = node.func.value
            if node.func.attr == "load":
                # descriptor_load(desc, offsets)
                return ast.Call(func=ast.Name(id="descriptor_load", ctx=ast.Load()), args=[base] + list(node.args), keywords=list(node.keywords))
            else:
                # descriptor_store(desc, offsets, value)
                return ast.Call(func=ast.Name(id="descriptor_store", ctx=ast.Load()), args=[base] + list(node.args), keywords=list(node.keywords))
        return node

    def _default_helper_layout_kw(self, shape_expr: ast.expr) -> ast.keyword:
        # layout=default_blocked_layout(shape, ttgl.num_warps())
        layout_call = ast.Call(
            func=ast.Name(id="default_blocked_layout", ctx=ast.Load()),
            args=[shape_expr, ast.Call(func=self._ttgl_attr("num_warps"), args=[], keywords=[])],
            keywords=[],
        )
        return ast.keyword(arg="layout", value=layout_call)

    def _make_slice_layout_value(self, dim_value: int, shape_expr: ast.expr) -> ast.Call:
        # ttgl.SliceLayout(dim, default_blocked_layout(shape, ttgl.num_warps()))
        return ast.Call(
            func=self._ttgl_attr("SliceLayout"),
            args=[
                ast.Constant(value=dim_value),
                ast.Call(func=ast.Name(id="default_blocked_layout", ctx=ast.Load()),
                         args=[shape_expr,
                               ast.Call(func=self._ttgl_attr("num_warps"), args=[], keywords=[])], keywords=[]),
            ],
            keywords=[],
        )

    def _handle_tl_arange(self, node: ast.Call) -> ast.Call:
        # Forward to helper that mirrors tl.arange signature and adds default layout
        return ast.Call(func=ast.Name(id="tl_arange", ctx=ast.Load()), args=list(node.args), keywords=list(node.keywords))

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        node = self.generic_visit(node)
        # Only rewrite inside selected functions (if scoping is enabled)
        if self._convert_only is not None and self._current_function not in self._convert_only:
            return node
        # For patterns like x[None, :] or x[:, None], ensure x has a SliceLayout along the expanded dim
        dim = None
        if isinstance(node.slice, ast.Tuple) and len(node.slice.elts) == 2:
            first, second = node.slice.elts
            if isinstance(first, ast.Constant) and first.value is None:
                dim = 0
            elif isinstance(second, ast.Constant) and second.value is None:
                dim = 1
        if dim is not None:
            value_expr = node.value
            # Construct a 2D parent shape with a dummy dimension of size 1 at the expanded dim
            # Use value.type.shape[0] as the vector length
            type_attr = ast.Attribute(value=value_expr, attr="type", ctx=ast.Load())
            shape_attr = ast.Attribute(value=type_attr, attr="shape", ctx=ast.Load())
            len_expr = ast.Subscript(value=shape_attr, slice=ast.Constant(value=0), ctx=ast.Load())
            if dim == 0:
                parent_shape = ast.List(elts=[len_expr, ast.Constant(value=1)], ctx=ast.Load())
            else:
                parent_shape = ast.List(elts=[ast.Constant(value=1), len_expr], ctx=ast.Load())
            # Build SliceLayout(dim, default_blocked_layout(parent_shape, ttgl.num_warps()))
            slice_layout = ast.Call(
                func=self._ttgl_attr("SliceLayout"),
                args=[
                    ast.Constant(value=dim),
                    ast.Call(
                        func=ast.Name(id="default_blocked_layout", ctx=ast.Load()),
                        args=[parent_shape, ast.Call(func=self._ttgl_attr("num_warps"), args=[], keywords=[])],
                        keywords=[],
                    ),
                ],
                keywords=[],
            )
            converted_value = ast.Call(
                func=self._ttgl_attr("convert_layout"),
                args=[value_expr, slice_layout],
                keywords=[],
            )
            return ast.Subscript(value=converted_value, slice=node.slice, ctx=node.ctx)
        return node

    def _handle_tl_program_id(self, node: ast.Call) -> ast.Call:
        return ast.Call(func=self._ttgl_attr("program_id"), args=list(node.args), keywords=list(node.keywords))

    def _handle_tl_load(self, node: ast.Call) -> ast.Call:
        return ast.Call(func=self._ttgl_attr("load"), args=list(node.args), keywords=list(node.keywords))

    def _handle_tl_store(self, node: ast.Call) -> ast.Call:
        return ast.Call(func=self._ttgl_attr("store"), args=list(node.args), keywords=list(node.keywords))

    def _handle_tl_zeros(self, node: ast.Call) -> ast.Call:
        # Forward to helper with same signature as tl.zeros
        return ast.Call(func=ast.Name(id="tl_zeros", ctx=ast.Load()), args=list(node.args), keywords=list(node.keywords))

    def _handle_tl_full(self, node: ast.Call) -> ast.Call:
        # Forward to helper with same signature as tl.full
        return ast.Call(func=ast.Name(id="tl_full", ctx=ast.Load()), args=list(node.args), keywords=list(node.keywords))

    def _handle_cdiv(self, node: ast.Call) -> ast.Call:
        # tl.cdiv(x, y) or triton.cdiv(x, y) -> ttgl.cdiv(x, y)
        return ast.Call(func=self._ttgl_attr("cdiv"), args=list(node.args), keywords=list(node.keywords))

    def _handle_tl_dot(self, node: ast.Call) -> ast.AST:
        # Rewrite tl.dot(a, b, acc=..., out_dtype=..., input_precision=...) -> dot_accumulate(a, b, acc, out_dtype=..., input_precision=...)
        # Only transform when an explicit acc kwarg is provided; otherwise leave unchanged
        return ast.Call(
            func=ast.Name(id="dot_accumulate", ctx=ast.Load()),
            args=list(node.args),
            keywords=list(node.keywords),
        )

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        return self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        return self.generic_visit(node)


def convert_triton_function_only(func: object) -> str:
    # Convert a single Triton JIT function definition without pulling the whole module
    # Unwrap Triton JIT wrapper objects
    if hasattr(func, "fn") and inspect.isfunction(getattr(func, "fn")):
        func = getattr(func, "fn")
    src = get_source(func)
    globals_map = getattr(func, "__globals__", None)
    entry_name = getattr(func, "__name__", None)

    tree = ast.parse(src)
    convert_only_names: Optional[set[str]] = {entry_name} if entry_name else None
    transformer = TritonToGluonTransformer(globals_map, convert_only_names)
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    # Prune to only the entry function
    if convert_only_names:
        pruned_body = []
        for stmt in new_tree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                pruned_body.append(stmt)
            elif isinstance(stmt, ast.FunctionDef) and stmt.name in convert_only_names:
                pruned_body.append(stmt)
        new_tree.body = pruned_body
    return unparse(new_tree)


def convert_triton_to_gluon(func_or_src: Union[str, object]) -> str:
    # If given a live function, prefer converting the entire defining module
    if not isinstance(func_or_src, str):
        try:
            mod = inspect.getmodule(func_or_src)
            if mod is not None:
                src = inspect.getsource(mod)
                globals_map = getattr(mod, "__dict__", None)
                entry_name = getattr(func_or_src, "__name__", None)
            else:
                src = get_source(func_or_src)
                globals_map = getattr(func_or_src, "__globals__", None)
                entry_name = getattr(func_or_src, "__name__", None)
        except OSError:
            src = get_source(func_or_src)
            globals_map = getattr(func_or_src, "__globals__", None)
            entry_name = getattr(func_or_src, "__name__", None)
    else:
        src = func_or_src
        globals_map = None
        entry_name = None

    tree = ast.parse(src)

    # Compute reachable function names from entry (only top-level, by simple name calls)
    convert_only_names: Optional[set[str]] = None
    external_callees: list[object] = []
    external_attr_to_local: set[tuple[str, str]] = set()
    if entry_name is not None:
        name_to_def: dict[str, ast.FunctionDef] = {}
        jit_names: set[str] = set()
        for n in tree.body:
            if isinstance(n, ast.FunctionDef):
                name_to_def[n.name] = n
                # detect original triton.jit decoration
                is_jit_orig = False
                for dec in n.decorator_list:
                    if isinstance(dec, ast.Attribute):
                        parts = [dec.value.id, dec.attr] if isinstance(dec.value, ast.Name) else None
                        if parts == ["triton", "jit"]:
                            is_jit_orig = True
                    elif isinstance(dec, ast.Name):
                        if dec.id == "jit":
                            is_jit_orig = True
                if is_jit_orig:
                    jit_names.add(n.name)
        reachable: set[str] = set()
        work: list[str] = []
        if entry_name in jit_names:
            reachable.add(entry_name)
            work.append(entry_name)
        # BFS on simple name calls, but only traverse into jit-decorated functions
        while work:
            fn = work.pop()
            fn_node = name_to_def.get(fn)
            if not fn_node:
                continue
            for sub in ast.walk(fn_node):
                if isinstance(sub, ast.Call):
                    if isinstance(sub.func, ast.Name):
                        callee = sub.func.id
                        if callee in jit_names and callee not in reachable:
                            reachable.add(callee)
                            work.append(callee)
                        # Track external jit callees referenced by name via globals
                        if callee not in name_to_def and globals_map is not None and callee in globals_map:
                            obj = globals_map[callee]
                            try:
                                if _is_supported_external(obj) and obj not in external_callees:
                                    external_callees.append(obj)
                            except Exception:
                                pass
                    # Handle module alias calls: alias.func(...)
                    if isinstance(sub.func, ast.Attribute) and isinstance(sub.func.value, ast.Name) and globals_map is not None:
                        alias = sub.func.value.id
                        attr = sub.func.attr
                        mod_obj = globals_map.get(alias)
                        try:
                            if mod_obj is not None and hasattr(mod_obj, attr):
                                obj = getattr(mod_obj, attr)
                                if _is_supported_external(obj):
                                    if obj not in external_callees:
                                        external_callees.append(obj)
                                        external_attr_to_local.add((alias, attr))
                        except Exception:
                            pass
        convert_only_names = reachable if reachable else (jit_names if entry_name in jit_names else None)

    transformer = TritonToGluonTransformer(globals_map, convert_only_names, external_attr_to_local)
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)

    # If scoping is enabled, prune to imports and reachable functions only
    if convert_only_names:
        pruned_body = []
        for stmt in new_tree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                pruned_body.append(stmt)
            elif isinstance(stmt, ast.FunctionDef) and stmt.name in convert_only_names:
                pruned_body.append(stmt)
        new_tree.body = pruned_body

    out = unparse(new_tree)

    # Ensure we have the gluon import lines at top (simple guard)
    first_lines = "\n".join(out.splitlines()[0:5])
    if "triton.experimental.gluon" not in first_lines:
        out = GLUON_IMPORT_LINES + "\n\n" + out

    # Append converted definitions for external callees (from other modules)
    if external_callees:
        appended_defs: list[str] = []
        # Avoid duplicating functions already present
        present_names = set()
        for stmt in new_tree.body:
            if isinstance(stmt, ast.FunctionDef):
                present_names.add(stmt.name)
        for ext_fn in external_callees:
            try:
                ext_src = convert_triton_function_only(ext_fn)
                ext_tree = ast.parse(ext_src)
                for stmt in ext_tree.body:
                    if isinstance(stmt, ast.FunctionDef) and stmt.name not in present_names:
                        appended_defs.append(unparse(ast.Module(body=[stmt], type_ignores=[])))
                        present_names.add(stmt.name)
            except Exception:
                continue
        if appended_defs:
            out = out.rstrip() + "\n\n" + "\n\n".join(s.strip() for s in appended_defs) + "\n"

    return out
