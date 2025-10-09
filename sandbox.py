# triton_to_gluon.py
import ast
import inspect
from typing import Union, Optional
import triton
import triton.language.core as tlc
from triton.experimental.gluon import language as ttgl_mod
import sys
import importlib
import importlib.util
import copy


GLUON_IMPORT_LINES = ("from triton.experimental import gluon\n"
                      "from triton.experimental.gluon import language as ttgl\n"
                      "from helpers import *\n")


class TritonToGluonTransformer(ast.NodeTransformer):

    def __init__(self, globals_map: dict,
                 shared_jit_set: set,
                 shared_queue: list,
                 is_jit,
                 constexpr_globals: dict):
        super().__init__()
        # Resolution scope (globals ∪ nonlocals)
        self._scope: dict = globals_map or {}
        # Track discovered JIT functions to inline/append later
        self._jit_functions: set = shared_jit_set
        self.queue: list = shared_queue
        self._is_jit = is_jit
        # Maps module_file -> {name: value}
        self._constexpr_globals: dict = constexpr_globals

    def _is_triton_constexpr_annotation(self, ann: ast.expr) -> bool:
        # Resolve the annotation to a Python object and compare by identity
        obj = self._resolve_value(ann)
        return obj is tlc.constexpr

    def _as_ttgl_constexpr(self) -> ast.expr:
        # Build ttgl.constexpr
        return ast.Attribute(value=ast.Name(id="ttgl", ctx=ast.Load()), attr="constexpr", ctx=ast.Load())

    def _maybe_rewrite_constexpr_annotation(self, ann: Optional[ast.expr]) -> Optional[ast.expr]:
        if ann is None:
            return None
        if self._is_triton_constexpr_annotation(ann):
            return self._as_ttgl_constexpr()
        return ann

    def _ttgl_attr(self, name: str) -> ast.AST:
        return ast.Attribute(value=ast.Name(id="ttgl", ctx=ast.Load()), attr=name, ctx=ast.Load())

    def _flatten_attr(self, node: ast.AST) -> list[str] | None:
      parts, cur = [], node
      while isinstance(cur, ast.Attribute):
          parts.insert(0, cur.attr)
          cur = cur.value
      if isinstance(cur, ast.Name):
          parts.insert(0, cur.id)
          return parts
      return None

    def _resolve_value(self, expr: ast.expr):
      # Resolve Name or dotted Attribute from recorded globals
      if isinstance(expr, ast.Name):
          # Try scope first, then loaded modules, then importable module
          val = self._scope.get(expr.id)
          if val is None:
              val = sys.modules.get(expr.id)
          if val is None:
              try:
                  val = importlib.import_module(expr.id)
              except Exception:
                  val = None
          return val
      if isinstance(expr, ast.Attribute):
          parts = self._flatten_attr(expr)
          if not parts:
              return None
          # Try from most-specific module/object root down to least
          for i in range(len(parts), 0, -1):
              root = ".".join(parts[:i])
              # 1) scope can contain fully-qualified names
              obj = self._scope.get(root)
              # 2) already-loaded modules
              if obj is None:
                  obj = sys.modules.get(root)
              # 3) importable modules
              if obj is None:
                  try:
                      obj = importlib.import_module(root)
                  except Exception:
                      obj = None
              if obj is None:
                  continue
              # Walk remaining attributes
              rem = parts[i:]
              for attr in rem:
                  obj = getattr(obj, attr, None)
                  if obj is None:
                      break
              if obj is not None:
                  return obj
          return None
      return None


    def _forward_call(self, node: ast.Call, target_func: ast.expr) -> ast.Call:
        new_keywords = [kw for kw in node.keywords if kw.arg not in {"can_reorder"}]
        return ast.Call(func=target_func, args=list(node.args), keywords=list(new_keywords))

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        fn_obj = self._resolve_value(node.func)
        if fn_obj is not None:
            fn_obj = triton.language.core._unwrap_if_constexpr(fn_obj)
            base = getattr(fn_obj, "fn", fn_obj)
            name = getattr(base, "__qualname__", getattr(base, "__name__", str(base)))
            if triton.language.core.is_builtin(fn_obj):
                simple = name.split(".")[-1]
                mapping: dict[str, ast.expr] = {
                    "arange": ast.Name(id="tl_arange", ctx=ast.Load()),
                    "program_id": self._ttgl_attr("program_id"),
                    "load": self._ttgl_attr("load"),
                    "store": self._ttgl_attr("store"),
                    "cdiv": self._ttgl_attr("cdiv"),
                    "static_print": self._ttgl_attr("static_print"),
                    "static_assert": self._ttgl_attr("static_assert"),
                    "device_assert": self._ttgl_attr("device_assert"),
                    "max_contiguous": self._ttgl_attr("max_contiguous"),
                    "multiple_of": self._ttgl_attr("multiple_of"),
                    "assume": self._ttgl_attr("assume"),
                    "minimum": self._ttgl_attr("minimum"),
                    "maximum": self._ttgl_attr("maximum"),
                    "fma": self._ttgl_attr("fma"),
                    "where": self._ttgl_attr("where"),
                    "cast": self._ttgl_attr("cast"),
                    "reshape": self._ttgl_attr("reshape"),
                    "trans": self._ttgl_attr("trans"),
                    "split": self._ttgl_attr("split"),
                    "inline_asm_elementwise": self._ttgl_attr("inline_asm_elementwise"),
                    "join": self._ttgl_attr("join"),
                    "atomic_max": self._ttgl_attr("atomic_max"),
                    "atomic_min": self._ttgl_attr("atomic_min"),
                    "atomic_or": self._ttgl_attr("atomic_or"),
                    "atomic_xchg": self._ttgl_attr("atomic_xchg"),
                    "atomic_xor": self._ttgl_attr("atomic_xor"),
                    "atomic_add": self._ttgl_attr("atomic_add"),
                    "atomic_and": self._ttgl_attr("atomic_and"),
                    "atomic_cas": self._ttgl_attr("atomic_cas"),
                    "num_warps": self._ttgl_attr("num_warps"),
                    "reduce": self._ttgl_attr("reduce"),
                    "full": ast.Name(id="tl_full", ctx=ast.Load()),
                    "dot": ast.Name(id="tl_dot", ctx=ast.Load()),
                    "dot_scaled": ast.Name(id="tl_dot_scaled", ctx=ast.Load()),
                    "make_tensor_descriptor": ast.Name(id="tl_make_tensor_descriptor", ctx=ast.Load()),
                    "load_tensor_descriptor": ast.Name(id="tl_load_tensor_descriptor", ctx=ast.Load()),
                    "store_tensor_descriptor": ast.Name(id="tl_store_tensor_descriptor", ctx=ast.Load()),
                    "num_threads": ast.Name(id="get_num_threads_per_warp", ctx=ast.Load()),
                }
                target = mapping.get(simple)
                if target is not None:
                    node = self._forward_call(node, target)
                    # For split, apply on the source argument rather than wrapping destination
                    if simple == "split":
                        src = node.args[0]
                        wrapped_src = ast.Call(func=ast.Name(id="set_split_src_layout", ctx=ast.Load()), args=[src], keywords=[])
                        node.args[0] = ast.copy_location(wrapped_src, src)
                    # For shape/layout changing ops, wrap to reset layout
                    if simple in {"reshape", "trans", "join", "reduce", "split"}:
                        forwarded = self._forward_call(node, target)
                        wrapped = ast.Call(func=ast.Name(id="reset_to_default_layout", ctx=ast.Load()), args=[forwarded], keywords=[])
                        node = ast.copy_location(wrapped, node)
                    return node
            # Track JITFunction callees
            if isinstance(fn_obj, triton.runtime.jit.JITCallable):
                if fn_obj not in self._jit_functions:
                    self._jit_functions.add(fn_obj)
                    self.queue.append(fn_obj)
                # Strip namespace: rewrite to local function name
                return self._forward_call(node, ast.Name(id=getattr(base, "__name__", ""), ctx=ast.Load()))
            if fn_obj is triton.language.core.range:
                # skip all keywords except arg1, arg2, and step and replace with range.
                allowed = {"arg1", "arg2", "step"}
                new_keywords = [kw for kw in node.keywords if kw.arg in allowed]
                new_args = list(node.args[:3])
                return ast.copy_location(ast.Call(func=ast.Name(id="range", ctx=ast.Load()), args=new_args, keywords=new_keywords),node,)
            if fn_obj is triton.language.core.static_range:
                return self._forward_call(node, ast.Name(id="ttgl.static_range", ctx=ast.Load()))
        else:
            if isinstance(node.func, ast.Attribute) and node.func.attr in ["store", "load", "gather"]:
                target = "tl_obj_" + node.func.attr
                return ast.Call(
                    func=ast.Name(id=target, ctx=ast.Load()),
                    args=[node.func.value] + list(node.args),
                    keywords=list(node.keywords),
                )                
            if isinstance(node.func, ast.Attribute) and node.func.attr in ["reshape", "trans", "split", "join", "reduce"]:
                if node.func.attr == "split":
                    recv = node.func.value
                    wrapped_recv = ast.Call(func=ast.Name(id="set_split_src_layout", ctx=ast.Load()), args=[recv], keywords=[])
                    new_func = ast.Attribute(value=ast.copy_location(wrapped_recv, recv), attr=node.func.attr, ctx=ast.Load())
                    node = ast.copy_location(ast.Call(func=new_func, args=list(node.args), keywords=list(node.keywords)), node)
                wrapped = ast.Call(
                    func=ast.Name(id="reset_to_default_layout", ctx=ast.Load()),
                    args=[node],
                    keywords=[],
                )
                return ast.copy_location(wrapped, node)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        node = self.generic_visit(node)
        parts = self._flatten_attr(node)
        if parts:
            last = parts[-1]
            # Only rewrite dtypes when the resolved object is a tl.dtype instance
            # or the tl.dtype class itself (e.g., tl.float16 or tl.dtype.float16 / tl.dtype)
            resolved = self._resolve_value(node)
            if resolved is not None:
                try:
                    if isinstance(resolved, tlc.dtype):
                        return self._ttgl_attr(last)
                except Exception:
                    pass
                if resolved is tlc.dtype and last == "dtype":
                    return self._ttgl_attr("dtype")
                if resolved is tlc.tensor and last == "tensor":
                    return self._ttgl_attr("tensor")
                if resolved is tlc.constexpr and last == "constexpr":
                    return self._ttgl_attr("constexpr")
            if last == "tensor_descriptor":
                return self._ttgl_attr("nvidia.hopper.tma.tensor_descriptor")
        return node

    def visit_Name(self, node):
        node = self.generic_visit(node)
        fn_obj = self._resolve_value(node)
        if fn_obj is not None:
            if isinstance(fn_obj, triton.language.core.constexpr):
                name = getattr(node, "id", None)
                if name is not None:
                    # Use the current capture scope's file for the defining module
                    module_file = self._scope.get("__file__")
                    if isinstance(module_file, str):
                        bucket = self._constexpr_globals.setdefault(module_file, {})
                        bucket[name] = fn_obj
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        node = self.generic_visit(node)
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

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        # Rewrite parameter annotations: triton.language.constexpr -> ttgl.constexpr
        # Positional-only and regular args
        for arg in list(getattr(node.args, "posonlyargs", [])) + list(node.args.args):
            arg.annotation = self._maybe_rewrite_constexpr_annotation(arg.annotation)
        # Vararg / kwarg
        if node.args.vararg is not None:
            node.args.vararg.annotation = self._maybe_rewrite_constexpr_annotation(node.args.vararg.annotation)
        if node.args.kwarg is not None:
            node.args.kwarg.annotation = self._maybe_rewrite_constexpr_annotation(node.args.kwarg.annotation)
        # Keyword-only args
        for arg in node.args.kwonlyargs:
            arg.annotation = self._maybe_rewrite_constexpr_annotation(arg.annotation)
        if self._is_jit:
            node.decorator_list.insert(0, ast.Attribute(value=ast.Name(id="gluon", ctx=ast.Load()), attr="jit", ctx=ast.Load()))
        else:
            node.decorator_list.insert(0, ast.Attribute(value=ast.Name(id="gluon", ctx=ast.Load()), attr="constexpr_function", ctx=ast.Load()))
        # Process body
        return self.generic_visit(node)

    # Simplified: per-op helpers removed in favor of mapping in visit_Call

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        return self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        return self.generic_visit(node)
    

def _unparse_original_assignments(constexpr_globals: dict) -> list[str]:
    # Build assignment strings for captured globals by parsing each module once.
    def _collect_names(t, out):
        if isinstance(t, ast.Name):
            out.append(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                _collect_names(e, out)

    def _parse_assigns_and_imports(path: str) -> tuple[dict[str, ast.AST], dict[str, str]]:
        try:
            with open(path, "r") as f:
                mod = ast.parse(f.read())
        except Exception:
            return {}, {}
        assigns: dict[str, ast.AST] = {}
        imports: dict[str, str] = {}
        for n in getattr(mod, "body", []):
            if isinstance(n, ast.Assign):
                names: list[str] = []
                for tgt in n.targets:
                    _collect_names(tgt, names)
                for nm in names:
                    assigns[nm] = n
            elif isinstance(n, ast.AnnAssign):
                names: list[str] = []
                _collect_names(n.target, names)
                if n.value is not None:
                    for nm in names:
                        assigns[nm] = n
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and isinstance(n.module, str):
                for alias in n.names:
                    alias_name = alias.asname or alias.name.split(".")[-1]
                    imports[alias_name] = n.module
        return assigns, imports

    def _rewrite_constexpr_to_ttgl(node: ast.AST) -> ast.AST:
        class _R(ast.NodeTransformer):
            def visit_Call(self, n: ast.Call) -> ast.AST:
                n = self.generic_visit(n)
                if isinstance(n.func, ast.Attribute) and n.func.attr == "constexpr":
                    n.func = ast.copy_location(ast.Attribute(value=ast.Name(id="ttgl", ctx=ast.Load()), attr="constexpr", ctx=ast.Load()), n.func)
                return n
        return _R().visit(node)

    results: list[str] = []
    imported_cache: dict[str, dict[str, ast.AST]] = {}
    for mod_file, name_to_obj in constexpr_globals.items():
        assigns, imports = _parse_assigns_and_imports(mod_file)
        for name in sorted(name_to_obj.keys()):
            node = assigns.get(name)
            if node is None:
                target_mod = imports.get(name)
                if target_mod:
                    try:
                        spec = importlib.util.find_spec(target_mod)
                        origin = getattr(spec, "origin", None) if spec is not None else None
                    except Exception:
                        origin = None
                    if origin:
                        assign_map = imported_cache.get(origin)
                        if assign_map is None:
                            assign_map, _ = _parse_assigns_and_imports(origin)
                            imported_cache[origin] = assign_map
                        node = assign_map.get(name)
            if node is not None:
                edited = _rewrite_constexpr_to_ttgl(copy.deepcopy(node))
                ast.fix_missing_locations(edited)
                results.append(ast.unparse(edited))
            else:
                results.append(f"{name} = {repr(name_to_obj[name])}")
    return results


def convert_triton_to_gluon(src: triton.runtime.jit.JITCallable) -> str:
    tree = ast.parse(src._src)
    capture_scope = getattr(src, "__globals__", {}) or {}
    shared_jit_set: set = set()
    function_queue: list = []
    constexpr_globals: dict = {}
    transformer = TritonToGluonTransformer(globals_map=capture_scope,
                                           shared_jit_set=shared_jit_set,
                                           shared_queue=function_queue,
                                           is_jit=True,
                                           constexpr_globals=constexpr_globals)
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    out = ast.unparse(new_tree)
    # Process discovered callee JITFunctions, converting and appending them
    while function_queue:
        callee = function_queue.pop(0)
        callee_src = callee._src
        callee_tree = ast.parse(callee_src)
        callee_scope = getattr(callee, "__globals__", {}) or {}
        jit = isinstance(callee, triton.runtime.JITFunction)
        callee_transformer = TritonToGluonTransformer(globals_map=callee_scope,
                                                     shared_jit_set=shared_jit_set,
                                                     shared_queue=function_queue,
                                                     is_jit=jit,
                                                     constexpr_globals=constexpr_globals)
        callee_new = callee_transformer.visit(callee_tree)
        ast.fix_missing_locations(callee_new)
        out += "\n\n" + ast.unparse(callee_new)

    out = "\n\n" + out

    # pull constexpr globals from the original source code
    for line in _unparse_original_assignments(constexpr_globals):
        out = line + "\n" + out
    # add imports
    out = GLUON_IMPORT_LINES + "\n\n" + out
    return out