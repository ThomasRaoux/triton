# triton_to_gluon.py
import ast
import inspect
from typing import Union, Optional
import triton
import triton.language.core as tlc
from triton.experimental.gluon import language as ttgl_mod
import sys
import importlib

TL_TO_TTGL_ATTR: set[str] = {
    "float16", "bfloat16", "float32", "float64",
    "int1", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64", "float8e5", 
    "float8e5b16", "float8e4nv", "float8e4b8", "float8e4b15", "constexpr", "tensor",
    "dtype",
}

GLUON_IMPORT_LINES = ("from triton.experimental import gluon\n"
                      "from triton.experimental.gluon import language as ttgl\n"
                      "from helpers import *\n")


class TritonToGluonTransformer(ast.NodeTransformer):

    def __init__(self, globals_map: Optional[dict],
                 shared_jit_set: Optional[set],
                 shared_queue: Optional[list],
                 is_jit):
        super().__init__()
        # Temp counter for generating unique variable names
        self._temp_counter: int = 0
        # Resolution scope (globals ∪ nonlocals)
        self._scope: dict = globals_map or {}
        # Track discovered JIT functions to inline/append later
        self._jit_functions: set = shared_jit_set if shared_jit_set is not None else set()
        self.queue: list = shared_queue if shared_queue is not None else []
        self._is_jit = is_jit

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

    def _new_temp_name(self, base: str) -> str:
        self._temp_counter += 1
        return f"__{base}_{self._temp_counter}"

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
        return ast.Call(func=target_func, args=list(node.args), keywords=list(node.keywords))

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
                    # For shape/layout changing ops, wrap to reset layout
                    if simple in {"reshape", "trans", "split", "join"}:
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
            if isinstance(node.func, ast.Attribute) and node.func.attr in ["reshape", "trans", "split", "join"]:
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
    


def convert_triton_to_gluon(src: triton.runtime.jit.JITCallable) -> str:
    tree = ast.parse(src._src)
    capture_scope = getattr(src, "__globals__", {}) or {}
    shared_jit_set: set = set()
    function_queue: list = []
    transformer = TritonToGluonTransformer(globals_map=capture_scope,
                                           shared_jit_set=shared_jit_set,
                                           shared_queue=function_queue,
                                           is_jit=True)
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
                                                     is_jit=jit)
        callee_new = callee_transformer.visit(callee_tree)
        ast.fix_missing_locations(callee_new)
        out += "\n\n" + ast.unparse(callee_new)
    # add imports
    out = GLUON_IMPORT_LINES + "\n\n" + out
    return out