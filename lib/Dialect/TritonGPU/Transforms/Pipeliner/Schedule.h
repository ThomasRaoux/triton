#ifndef TRITON_TRITONGPU_TRANSFORM_PIPELINE_SCHEDULE_H_
#define TRITON_TRITONGPU_TRANSFORM_PIPELINE_SCHEDULE_H_

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include <vector>
#include "PipelineExpander.h"

namespace mlir {
namespace triton {

bool preProcessLoopAndGetSchedule(scf::ForOp& forOp, int numStages,
                                  mlir::triton::PipeliningOption &options);

} // namespace triton
} // namespace mlir
#endif // TRITON_TRITONGPU_TRANSFORM_PIPELINE_SCHEDULE_H_
