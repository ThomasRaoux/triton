#import sys
##import importlib.util
##import torch
#import triton
#import triton.language as tl
#import kernel as k
#from triton_kernels.numerics_details.mxfp_details._downcast_to_mxfp import _downcast_to_mxfp
#from triton_kernels.matmul_ogs import matmul_ogs_set_idle_sms, matmul_ogs, matmul_ogs_torch
from ki.matmul import _matmul_ogs_stoch_round
import inspect, ast, textwrap

#import pytest

from sandbox import convert_triton_to_gluon

#import inspect, ast, textwrap


def test_simple_matmul(tmp_path):
    #fn = _matmul_ogs_stoch_round.fn
    #print(ast.unparse(ast.parse(_matmul_ogs_stoch_round._src)))
    print("\n\n\n----\n\n")
    txt = convert_triton_to_gluon(_matmul_ogs_stoch_round)
    print(txt)
 #   print(_p_matmul_ogs_default)
   