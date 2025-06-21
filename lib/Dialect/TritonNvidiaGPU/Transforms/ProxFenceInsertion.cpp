#include "mlir/IR/Block.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"

//===----------------------------------------------------------------------===//
//
// This pass works after all other passes, inserting fences to ensure that
// memory operations are properly ordered across generic and async proxy.
//
//===----------------------------------------------------------------------===//

namespace ttg = mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONGPUPROXYFENCEINSERTION
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

bool isAsyncProxyWrite(Operation *op) {
  return isa<triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp,
             triton::nvidia_gpu::AsyncTMAGatherOp>(op);
}

Value getSmemDest(Operation *op) {
  if (auto asyncTMACopyGlobalToLocalOp = dyn_cast<triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp>(op)) {
    return asyncTMACopyGlobalToLocalOp.getResult();
  }
  if (auto asyncTMAGatherOp = dyn_cast<triton::nvidia_gpu::AsyncTMAGatherOp>(op)) {
    return asyncTMAGatherOp.getResult();
  }
  return Value();
}

bool isAsyncProxyRead(Operation *op) {
  return isa<triton::nvidia_gpu::WarpGroupDotOp,
             triton::nvidia_gpu::TCGen5MMAOp,
             triton::nvidia_gpu::TCGen5MMAScaledOp,
             triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp,
             triton::nvidia_gpu::AsyncTMAScatterOp,
             triton::nvidia_gpu::AsyncTMAReduceOp>(op);
}

bool ignoreOpForProxyFence(Operation *op) {
  return isAsyncProxyRead(op) || isAsyncProxyWrite(op) ||
         isa<triton::nvidia_gpu::ArriveBarrierOp,
             triton::nvidia_gpu::TMEMCopyOp,
             triton::nvidia_gpu::WaitBarrierOp,
             triton::nvidia_gpu::InitBarrierOp,
             triton::nvidia_gpu::InvalBarrierOp>(op);
}

bool filterFn(Operation *op, Operation *other) {
  return ignoreOpForProxyFence(other);
}

//===----------------------------------------------------------------------===//
// Proxy Fence Analysis
//===----------------------------------------------------------------------===//
class ProxyFenceAnalysis {
  using VirtualBlock = std::pair<Block *, Block::iterator>;

public:
  using FuncBlockInfoMapT = CallGraph<BlockInfo>::FuncDataMapT;
  /// Creates a new fence analysis that generates the shared memory barrier
  /// in the following circumstances:
  /// - RAW: If a shared memory write is followed by an op reading shared memory using the async proxy read, and
  /// their addresses are intersected, a fence is inserted.
  /// - WAR: If a shared memory read is followed by an op writing shared memory using the async proxy write, and
  /// their addresses are intersected, a fence is inserted.
  ProxyFenceAnalysis() = default;
  explicit ProxyFenceAnalysis(Allocation *allocation)
      : allocation(allocation) {}

  /// Runs the fence analysis to the given operation, inserts a fence if
  /// necessary.
  void run(FuncBlockInfoMapT &funcBlockInfoMap);

private:
  void resolve(FunctionOpInterface funcOp, FuncBlockInfoMapT *funcBlockInfoMap,
               OpBuilder *builder);

  /// Updates the BlockInfo operation based on the operation.
  void update(Operation *operation, BlockInfo *blockInfo,
              FuncBlockInfoMapT *funcBlockInfoMap, OpBuilder *builder);

  /// Collects the successors of the terminator
  void visitTerminator(Operation *operation,
                       SmallVector<VirtualBlock> &successors);

  void  insertFence(Operation *operation, OpBuilder *builder);

private:
  Allocation *allocation = nullptr;
};

/// Postorder traversal on the callgraph to insert fence instructions
/// of each function.
/// Each function maintains a BlockInfo map that includes all potential buffers
/// after returning. This way users do not have to explicitly insert proxy fences
/// before and after function calls, but might be a bit conservative.
class ModuleProxyFenceAnalysis : public CallGraph<BlockInfo> {
public:
  ModuleProxyFenceAnalysis(ModuleAllocation *moduleAllocation)
      : CallGraph<BlockInfo>(moduleAllocation->getModuleOp()),
        moduleAllocation(moduleAllocation) {}

  void run() {
    walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
        // Pre-order walk callback
        [](CallOpInterface callOp, FunctionOpInterface funcOp) {},
        // Post-order walk callback
        [&](FunctionOpInterface funcOp) {
          auto *allocation = moduleAllocation->getFuncData(funcOp);
          auto [it, inserted] = funcMap.try_emplace(funcOp, BlockInfo());
          if (inserted) {
            ProxyFenceAnalysis analysis(allocation);
            analysis.run(funcMap);
          }
        });
  }

private:
  ModuleAllocation *moduleAllocation;
};


void ProxyFenceAnalysis::run(FuncBlockInfoMapT &funcBlockInfoMap) {
  FunctionOpInterface funcOp =
      dyn_cast<FunctionOpInterface>(allocation->getOperation());
  OpBuilder builder(funcOp.getContext());
  resolve(funcOp, &funcBlockInfoMap, &builder);
}

void ProxyFenceAnalysis::resolve(FunctionOpInterface funcOp,
                             FuncBlockInfoMapT *funcBlockInfoMap,
                             OpBuilder *builder) {
  // Initialize the blockList. Operations are organized into "virtual blocks",
  // which represent segments of straight-line code analyzed by each iteration
  // of the dataflow analysis. Virtual blocks abstract over both control flow
  // represented by basic blocks and block successors (i.e. `BranchOpInterface`)
  // and control flow represented by regions (i.e. `RegionBranchOpInterface`).
  //
  // A virtual block consists of a parent block and a starting iterator, where
  // the virtual block starts on the operation *after* the starting iterator. A
  // null iterator is used to represent the beginning of the block. The virtual
  // block ends at any region branch operation or the basic block terminator.
  // Thus, basic blocks are broken up into multiple virtual blocks at each
  // region operation.
  //
  // Entry virtual blocks are represented by a null iterator. Populate the
  // blockList with the entry virtual blocks in the function. Then, each
  // iteration scans until a terminator or region branch operation is found.
  DenseMap<VirtualBlock, BlockInfo> inputBlockInfoMap;
  DenseMap<VirtualBlock, BlockInfo> outputBlockInfoMap;
  std::deque<VirtualBlock> blockList;
  funcOp.walk<WalkOrder::PreOrder>([&](Block *block) {
    // Start the analysis from the entry blocks of any nested isolated from
    // above regions.
    if (block->isEntryBlock() &&
        !isa<RegionBranchOpInterface>(block->getParentOp()))
      blockList.emplace_back(block, Block::iterator());
  });

  // A fixed point algorithm
  while (!blockList.empty()) {
    VirtualBlock block = blockList.front();
    blockList.pop_front();
    // Make a copy of the inputblockInfo but not update
    auto inputBlockInfo = inputBlockInfoMap[block];
    SmallVector<VirtualBlock> successors;
    Block::iterator startIt =
        block.second.isValid() ? std::next(block.second) : block.first->begin();
    for (Operation &op : llvm::make_range(startIt, block.first->end())) {
      if (op.hasTrait<OpTrait::IsTerminator>() ||
          isa<RegionBranchOpInterface>(op)) {
        visitTerminator(&op, successors);
        break;
      }
      update(&op, &inputBlockInfo, funcBlockInfoMap, builder);
    }
    // Get the reference because we want to update if it changed
    if (outputBlockInfoMap.count(block) &&
        inputBlockInfo == outputBlockInfoMap[block]) {
      // If we have seen the block before and the inputBlockInfo is the same as
      // the outputBlockInfo, we skip the successors
      continue;
    }
    // Update the current block. The block transfer function is not monotonic,
    // so overwrite the output state entirely.
    outputBlockInfoMap[block] = inputBlockInfo;
    // Update the successors
    for (VirtualBlock successor : successors) {
      inputBlockInfoMap[successor].join(outputBlockInfoMap[block]);
      blockList.emplace_back(successor);
    }
  }

  // Update the final dangling buffers that haven't been synced
  BlockInfo &funcBlockInfo = (*funcBlockInfoMap)[funcOp];
  funcOp.walk<WalkOrder::PreOrder>([&](triton::ReturnOp returnOp) {
    // A basic block can be broken into several virtual blocks. Find all virtual
    // blocks that belong to the basic block containing the return.
    SmallVector<std::pair<VirtualBlock, BlockInfo>> virtualBlocks;
    for (auto &[block, blockInfo] : outputBlockInfoMap) {
      if (block.first == returnOp->getBlock())
        virtualBlocks.emplace_back(block, blockInfo);
    }
    // The return is a terminator, so the virtual block that contains this
    // return starts after all other ones. Find it by comparing the start
    // iterators of the virtual blocks.
    auto maxIt = llvm::max_element(virtualBlocks, [&](auto &lhs, auto &rhs) {
      assert(lhs.first.first == rhs.first.first);
      Block::iterator lhsIt = lhs.first.second, rhsIt = rhs.first.second;
      return !lhsIt.isValid() ||
             (rhsIt.isValid() && lhsIt->isBeforeInBlock(&*rhsIt));
    });

    funcBlockInfo.join(maxIt->second);
  });
}

void ProxyFenceAnalysis::visitTerminator(Operation *op,
                                     SmallVector<VirtualBlock> &successors) {
  if (isa<BranchOpInterface>(op)) {
    // Collect the block successors of the branch.
    for (Block *successor : op->getSuccessors())
      successors.emplace_back(successor, Block::iterator());
    return;
  }

  if (auto br = dyn_cast<RegionBranchOpInterface>(op)) {
    // The successors of an operation with regions can be queried via an
    // interface. The operation branches to the entry blocks of its region
    // successors. It can also branch to after itself.
    SmallVector<RegionSuccessor> regions;
    br.getSuccessorRegions(RegionBranchPoint::parent(), regions);
    for (RegionSuccessor &region : regions) {
      if (region.isParent()) {
        successors.emplace_back(br->getBlock(), br->getIterator());
      } else {
        Block &block = region.getSuccessor()->front();
        successors.emplace_back(&block, Block::iterator());
      }
    }
    return;
  }

  // FIXME: `ReturnLike` adds `RegionBranchTerminatorOpInterface` for some
  // reason. Check that the parent is actually a `RegionBranchOpInterface`.
  auto br = dyn_cast<RegionBranchTerminatorOpInterface>(op);
  if (br && isa<RegionBranchOpInterface>(br->getParentOp())) {
    // Check the successors of a region branch terminator. It can branch to
    // another region of its parent operation or to after the parent op.
    SmallVector<Attribute> operands(br->getNumOperands());
    SmallVector<RegionSuccessor> regions;
    br.getSuccessorRegions(operands, regions);
    for (RegionSuccessor &region : regions) {
      if (region.isParent()) {
        Operation *parent = br->getParentOp();
        successors.emplace_back(parent->getBlock(), parent->getIterator());
      } else {
        Block &block = region.getSuccessor()->front();
        successors.emplace_back(&block, Block::iterator());
      }
    }
    return;
  }

  // Otherwise, it could be a return op
  if (op->hasTrait<OpTrait::ReturnLike>())
    return;
  llvm_unreachable("Unknown terminator encountered in membar analysis");
}

void ProxyFenceAnalysis::insertFence(Operation *op, OpBuilder *builder) {
  OpBuilder::InsertionGuard g(*builder);
  builder->create<triton::nvidia_gpu::FenceAsyncSharedOp>(op->getLoc(), false);
}

void ProxyFenceAnalysis::update(Operation *op, BlockInfo *blockInfo,
            
                            FuncBlockInfoMapT *funcBlockInfoMap,
                            OpBuilder *builder) {
  if (isa<triton::nvidia_gpu::FenceAsyncSharedOp>(op)) {
    // If the current op is a fence, we clear previous reads and writes
    blockInfo->sync();
    return;
  }                              
  BlockInfo curBlockInfo;
  BlockInfo proxyBlockInfo;

  auto scratchBufferId = Allocation::InvalidBufferId;
  if (isa<triton::CallOp>(op)) {
    // Inter-function dependencies
    auto callOpInterface = dyn_cast<CallOpInterface>(op);
    if (auto callee =
            dyn_cast<FunctionOpInterface>(callOpInterface.resolveCallable()))
      curBlockInfo = funcBlockInfoMap->lookup(callee);
  } else {
    // Intra-function dependencies
    if (auto memoryEffectOpInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
      // Explicit buffer
      SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>>
          effectInstances;
      memoryEffectOpInterface.getEffects(effectInstances);
      for (auto effectInstance : effectInstances) {
        if (auto value = effectInstance.getValue()) {
          for (auto bufferId : allocation->getBufferIds(value)) {
            if (bufferId != Allocation::InvalidBufferId) {
              // TODO: handle proxy read cases. Those are currently handled in
              // FenceInsertionPass where it can generate better placement for
              // the fence. But we should support a safe fallback here.
              if (isAsyncProxyWrite(op)) {
                if (value == getSmemDest(op)) {
                  proxyBlockInfo.syncWriteIntervals[allocation->getAllocatedInterval(bufferId)].insert(op);
                }
              } else if (isa<MemoryEffects::Write>(
                             effectInstance.getEffect())) {
                curBlockInfo
                    .syncWriteIntervals[allocation->getAllocatedInterval(
                        bufferId)]
                    .insert(op);
              } else if (isa<MemoryEffects::Read>(effectInstance.getEffect())) {
                curBlockInfo
                    .syncReadIntervals[allocation->getAllocatedInterval(
                        bufferId)]
                    .insert(op);
              }
            }
          }
        }
      }
    }
    scratchBufferId = allocation->getBufferId(op);
  }

  // Scratch buffer operations consist of a series of shared memory operations
  // starting from a shared memory write, followed by a series of shared memory
  // read/write operations, mark them as a read.
  if (scratchBufferId != Allocation::InvalidBufferId) {
    auto interval = allocation->getAllocatedInterval(scratchBufferId);
    curBlockInfo.syncReadIntervals[interval].insert(op);
  } 
  if (isAsyncProxyWrite(op) || isAsyncProxyRead(op)) {
    if (proxyBlockInfo.isIntersected(*blockInfo, filterFn)) {
      builder->setInsertionPoint(op);
      insertFence(op, builder);
      blockInfo->sync();
    }
  }
  
  // Update the region info, even if barrier is inserted, we have to maintain
  // the current op's read/write buffers.
  blockInfo->join(curBlockInfo);
}
} // namespace

struct ProxyFenceInsertionPass
    : public impl::TritonGPUProxyFenceInsertionBase<ProxyFenceInsertionPass> {

public:
  using impl::TritonGPUProxyFenceInsertionBase<
      ProxyFenceInsertionPass>::TritonGPUProxyFenceInsertionBase;
  void runOnOperation() override {
    // Only insert fences for compute capability 9.0
    if (computeCapability < 90)
      return;
    ModuleOp mod = getOperation();
    ModuleAllocation allocation(mod);
    ModuleProxyFenceAnalysis analysis(&allocation);
    analysis.run();
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
