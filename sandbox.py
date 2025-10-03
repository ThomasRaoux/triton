# triton_to_gluon.py
import ast
import inspect
from typing import Union, Optional
import triton
import triton.language.core as tlc

GLUON_IMPORT_LINES = ("from triton.experimental import gluon\n"
                      "from triton.experimental.gluon import language as ttgl\n"
                      "from helpers import *\n")


class TritonToGluonTransformer(ast.NodeTransformer):

    def __init__(self, globals_map: Optional[dict]):
        super().__init__()
        # Temp counter for generating unique variable names
        self._temp_counter: int = 0
        # Globals from the original context (module/function) for resolving names
        self._globals: dict = globals_map

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
          return self._globals.get(expr.id)
      if isinstance(expr, ast.Attribute):
          parts = self._flatten_attr(expr)
          if not parts:
              return None
          obj = self._globals.get(parts[0])
          for attr in parts[1:]:
              if obj is None:
                  return None
              obj = getattr(obj, attr, None)
          return obj
      return None


    def _forward_call(self, node: ast.Call, target_func: ast.expr) -> ast.Call:
        return ast.Call(func=target_func, args=list(node.args), keywords=list(node.keywords))

    def visit_Call(self, node: ast.Call) -> ast.AST:
        fn_obj = self._resolve_value(node.func)
        if fn_obj is not None:
            fn_obj = triton.language.core._unwrap_if_constexpr(fn_obj)
            if triton.language.core.is_builtin(fn_obj):
                base = getattr(fn_obj, "fn", fn_obj)
                name = getattr(base, "__qualname__", getattr(base, "__name__", str(base)))
                simple = name.split(".")[-1]
                mapping: dict[str, ast.expr] = {
                    "arange": ast.Name(id="tl_arange", ctx=ast.Load()),
                    "program_id": self._ttgl_attr("program_id"),
                    "load": self._ttgl_attr("load"),
                    "store": self._ttgl_attr("store"),
                    "cdiv": self._ttgl_attr("cdiv"),
                    "zeros": ast.Name(id="tl_zeros", ctx=ast.Load()),
                    "full": ast.Name(id="tl_full", ctx=ast.Load()),
                    "dot": ast.Name(id="dot_accumulate", ctx=ast.Load()),
                }
                target = mapping.get(simple)
                if target is not None:
                    return self._forward_call(node, target)
            if isinstance(fn_obj, triton.runtime.JITFunction):
                base = getattr(fn_obj, "fn", fn_obj)
                name = getattr(base, "__qualname__", getattr(base, "__name__", str(base)))
                print("call jit function", name)
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
        return node
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
        # Unconditionally add @gluon.jit decorator
        node.decorator_list.insert(0, ast.Attribute(value=ast.Name(id="gluon", ctx=ast.Load()), attr="jit", ctx=ast.Load()))
        # Process body
        return self.generic_visit(node)

    # Simplified: per-op helpers removed in favor of mapping in visit_Call

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        return self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        return self.generic_visit(node)

def convert_triton_to_gluon(src: triton.runtime.JITFunction) -> str:
    tree = ast.parse(src._src)
    transformer = TritonToGluonTransformer(globals_map=getattr(src, "__globals__", {}))
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    out = ast.unparse(new_tree)
    # add imports
    out = GLUON_IMPORT_LINES + "\n\n" + out
    return out