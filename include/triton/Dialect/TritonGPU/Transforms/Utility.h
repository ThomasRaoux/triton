#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_UTILITY_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_UTILITY_H_

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/MapVector.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {

namespace triton {
class LoadOp;
class StoreOp;
class FuncOp;
namespace gpu {
class SharedEncodingAttr;
}
} // namespace triton

LogicalResult fixupLoops(ModuleOp mod);

SmallVector<unsigned, 3> mmaVersionToInstrShape(int version,
                                                const ArrayRef<int64_t> &shape,
                                                RankedTensorType type);

/// Returns true if the Load is for TMA
bool isLoadFromTensorPtr(triton::LoadOp op);

/// Returns true if the store is for TMA
bool isStoreToTensorPtr(triton::StoreOp op);

/// Return the first consumer of v
Operation *getFirstUser(Value v);

/// Return the proper SharedEncodingAttr according to shape/order
triton::gpu::SharedEncodingAttr getSharedEncoding(RankedTensorType tensorTy);

/* Dump Triton IR in graphviz dot format.
 *
 * You can override `onValue` and `onOperation` in a subclass to mark
 * specific Values and Operations. The below subclass
 * GraphLayoutMarker is an example.
 *
 * Default NodeInfo for Value nodes:
 *   {{"shape": "box"},
 *    {"style", "filled"},
 *    {"fillcolor", "white"},
 *    {"label", shapeStr}}
 *
 * Default NodeInfo for Operation nodes:
 *   {{"shape": "ellipse"},
 *    {"style", "filled"},
 *    {"fillcolor", "white"},
 *    {"label", operationName}}
 *
 * If the key "label" is not set by `onValue` or `onOperation`, default labels
 * will be generated. For Value node, the default label is the shape string and
 * for Operation node, it is the operation name.
 *
 * Reference:
 *   https://graphviz.org/doc/info/shapes.html
 *   https://graphviz.org/doc/info/colors.html
 *
 * Usage:
 *   C++:   GraphDumper().dumpToFile(func, "func.dot");
 *   Shell: dot -Tjpg func.dot -o func.jpg
 */
class GraphDumper {
public:
  using NodeInfo = std::map<std::string, std::string>;

  // Override this function to mark specific Values
  virtual NodeInfo onValue(Value value) const;
  // Override this function to mark specific Operations
  virtual NodeInfo onOperation(Operation *op) const;

  std::string dump(triton::FuncOp func) const;
  void dumpToFile(triton::FuncOp func, const std::string &filename) const;

protected:
  std::string getShapeStr(const Type &type) const;

  std::string getUniqueId(Value value) const;
  std::string getUniqueId(Operation *op) const;

  std::string emitNode(const std::string &id, const NodeInfo style) const;
  std::string emitEdge(const std::string &srcId,
                       const std::string &destId) const;

  std::string emitValueNode(Value value) const;
  std::string emitOperationNode(Operation *op) const;
};

/* A subclass of GraphDumper that marks different layout kinds in different
 * colors.*/
class GraphLayoutMarker : public GraphDumper {
public:
  NodeInfo onValue(Value value) const override;

protected:
  std::string getColor(const Type &type) const;
};

bool isExpensiveLoadOrStore(Operation *op);

bool canFoldIntoConversion(Operation *op, Attribute targetEncoding);

// skipInit is True when we only consider the operands of the initOp but
// not the initOp itself.
int simulateBackwardRematerialization(
    Operation *initOp, SetVector<Operation *> &processed,
    SetVector<Attribute> &layout, llvm::MapVector<Value, Attribute> &toConvert,
    Attribute targetEncoding);

Operation *cloneWithInferType(mlir::OpBuilder &rewriter, Operation *op,
                              IRMapping &mapping);

void rematerializeConversionChain(
    const llvm::MapVector<Value, Attribute> &toConvert,
    mlir::PatternRewriter &rewriter, SetVector<Operation *> &processed,
    IRMapping &mapping);

// Convert an \param index to a multi-dim coordinate given \param shape and
// \param order.
SmallVector<Value> delinearize(OpBuilder &b, Location loc, Value linear,
                               ArrayRef<unsigned> shape,
                               ArrayRef<unsigned> order);

SmallVector<Value> delinearize(OpBuilder &b, Location loc, unsigned linear,
                               ArrayRef<unsigned> shape);

SmallVector<Value> delinearize(OpBuilder &b, Location loc, Value linear,
                               ArrayRef<unsigned> shape);
Value linearize(OpBuilder &b, Location loc, ArrayRef<Value> multiDim,
                ArrayRef<unsigned> shape, ArrayRef<unsigned> order);

Value linearize(OpBuilder &b, Location loc, ArrayRef<Value> multiDim,
                ArrayRef<unsigned> shape);

// Returns null if the op is not inside a agent region (warp specialization
// mode). Note that there should be at most one agent id attached to the
// operation.
std::optional<int> getWSAgentId(Operation *op);
std::optional<int> getWSRoleId(Operation *op);
void setRoleId(Operation *op, int roleId);

} // namespace mlir

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_UTILITY_H_
