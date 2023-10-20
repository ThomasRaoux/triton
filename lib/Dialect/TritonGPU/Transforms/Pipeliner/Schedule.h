#ifndef TRITON_TRITONGPU_TRANSFORM_PIPELINE_SCHEDULE_H_
#define TRITON_TRITONGPU_TRANSFORM_PIPELINE_SCHEDULE_H_

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include <vector>

namespace mlir {
namespace triton {

// Create a full loop schedule based on a coarse schedule passed by user.
// This helper will generate an opinionated schedule based on the coarse
// schedule passed by user. In the future we should allow different modes based
// on the type of loop we want to schedule.
std::vector<std::pair<Operation *, unsigned>>
createSchedule(scf::ForOp& op, int numStages,
               ArrayRef<std::pair<Operation *, unsigned>> coarseSchedule);

} // namespace triton
} // namespace mlir
#endif // TRITON_TRITONGPU_TRANSFORM_PIPELINE_SCHEDULE_H_
