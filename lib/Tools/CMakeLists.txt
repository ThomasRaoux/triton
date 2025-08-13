add_triton_library(TritonTools
  GenericSwizzling.cpp
  LayoutUtils.cpp
  LinearLayout.cpp

  DEPENDS

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  f2reduce
)


add_triton_library(TritonPassTools
  PressureAnnotation.cpp

  DEPENDS

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRPass
  TritonIR
  TritonGPUIR
  TritonGPUTransforms
  f2reduce
)
