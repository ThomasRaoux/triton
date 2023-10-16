add_mlir_dialect_library(TritonGPUTransforms
  AccelerateMatmul.cpp
  Coalesce.cpp
  DecomposeConversions.cpp
  OptimizeDotOperands.cpp
  OptimizeEpilogue.cpp
  Pipeliner/SoftwarePipeliner.cpp
  Pipeliner/ScheduleCreator.cpp
  Pipeliner/Expander.cpp
  Prefetch.cpp
  RemoveLayoutConversions.cpp
  ReorderInstructions.cpp
  TritonGPUConversion.cpp
  Utility.cpp

  DEPENDS
  TritonGPUTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRTransforms
  MLIRTransformUtils
  TritonAnalysis
  TritonIR
  TritonGPUIR
  TritonNvidiaGPUIR
  MLIRTransformUtils
)
