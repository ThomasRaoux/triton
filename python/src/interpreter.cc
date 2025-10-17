// Nanobind-based implementation of the CPU interpreter helpers previously
// implemented with pybind11. Semantics are kept equivalent.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <atomic>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <type_traits>

namespace nb = nanobind;

enum class MemSemantic { ACQUIRE_RELEASE, ACQUIRE, RELEASE, RELAXED };
enum class RMWOp { ADD, FADD, AND, OR, XOR, XCHG, MAX, MIN, UMIN, UMAX };

namespace {

struct npy_half {
  uint16_t value;
};

std::mutex atomic_op_guard;

template <typename T>
constexpr bool is_reinterpret_cast_to_atomic_safe =
    std::is_trivially_copyable_v<T> &&
    std::is_trivially_copyable_v<std::atomic<T>> &&
    std::is_standard_layout_v<T> && std::is_standard_layout_v<std::atomic<T>> &&
    sizeof(T) == sizeof(std::atomic<T>) &&
    alignof(T) == alignof(std::atomic<T>);

std::map<MemSemantic, std::memory_order> mem_semantic_map = {
    {MemSemantic::ACQUIRE_RELEASE, std::memory_order_acq_rel},
    {MemSemantic::ACQUIRE, std::memory_order_acquire},
    {MemSemantic::RELEASE, std::memory_order_release},
    {MemSemantic::RELAXED, std::memory_order_relaxed},
};

enum class RMWInnerOp { ADD, FADD, AND, OR, XOR, XCHG, MAX, MIN, UMIN, UMAX };

template <bool is_min, typename T>
T atomic_cmp(T *ptr, T val, std::memory_order order) {
  auto cmp = [](T old, T val) {
    if constexpr (is_min) {
      return old > val;
    } else {
      return old < val;
    }
  };

  T old_val;
  if constexpr (is_reinterpret_cast_to_atomic_safe<T>) {
    std::atomic<T> *atomic_ptr = reinterpret_cast<std::atomic<T> *>(ptr);
    old_val = atomic_ptr->load(order);
    while (cmp(old_val, val)) {
      if (atomic_ptr->compare_exchange_weak(old_val, val, order, order)) {
        break;
      }
    }
  } else {
    const std::lock_guard<std::mutex> lock(atomic_op_guard);
    old_val = *ptr;
    if (cmp(old_val, val)) {
      *ptr = val;
    }
  }
  return old_val;
}

template <typename T> T atomic_fadd(T *loc, T value, std::memory_order order) {
  static_assert(std::is_floating_point<T>::value,
                "T must be a floating-point type");
  T old_value;

  if constexpr (is_reinterpret_cast_to_atomic_safe<T>) {
    T new_value;
    std::atomic<T> *atomic_loc = reinterpret_cast<std::atomic<T> *>(loc);
    old_value = atomic_loc->load(order);
    do {
      new_value = old_value + value;
    } while (
        !atomic_loc->compare_exchange_weak(old_value, new_value, order, order));
  } else {
    const std::lock_guard<std::mutex> lock(atomic_op_guard);
    old_value = *loc;
    *loc = old_value + value;
  }

  return old_value;
}

template <typename To, typename From>
inline To BitCast(const From &from) noexcept {
  static_assert(sizeof(To) == sizeof(From),
                "both data types must have the same size");
  static_assert(std::is_trivially_copyable_v<To> &&
                    std::is_trivially_copyable_v<From>,
                "both data types must be trivially copyable");
  To to;
  memcpy(&to, &from, sizeof(from));
  return to;
}

template <bool gen_overflow = true, bool gen_underflow = true,
          bool round_even = true>
inline uint16_t FromFloatBits(uint32_t f) {
  uint32_t f_exp, f_sig;
  uint16_t h_sgn, h_exp, h_sig;

  h_sgn = (uint16_t)((f & 0x80000000u) >> 16);
  f_exp = (f & 0x7f800000u);

  if (f_exp >= 0x47800000u) {
    if (f_exp == 0x7f800000u) {
      f_sig = (f & 0x007fffffu);
      if (f_sig != 0) {
        uint16_t ret = (uint16_t)(0x7c00u + (f_sig >> 13));
        if (ret == 0x7c00u) {
          ret++;
        }
        return h_sgn + ret;
      } else {
        return (uint16_t)(h_sgn + 0x7c00u);
      }
    } else {
      if constexpr (gen_overflow) {
        throw std::overflow_error("overflow to signed inf");
      }
      return (uint16_t)(h_sgn + 0x7c00u);
    }
  }

  if (f_exp <= 0x38000000u) {
    if (f_exp < 0x33000000u) {
      if constexpr (gen_underflow) {
        if ((f & 0x7fffffff) != 0) {
          throw std::underflow_error("");
        }
      }
      return h_sgn;
    }
    f_exp >>= 23;
    f_sig = (0x00800000u + (f & 0x007fffffu));
    if constexpr (gen_underflow) {
      if ((f_sig & (((uint32_t)1 << (126 - f_exp)) - 1)) != 0) {
        throw std::underflow_error("");
      }
    }
    f_sig >>= (113 - f_exp);
    if constexpr (round_even) {
      if (((f_sig & 0x00003fffu) != 0x00001000u) || (f & 0x000007ffu)) {
        f_sig += 0x00001000u;
      }
    } else {
      f_sig += 0x00001000u;
    }
    h_sig = (uint16_t)(f_sig >> 13);
    return (uint16_t)(h_sgn + h_sig);
  }

  h_exp = (uint16_t)((f_exp - 0x38000000u) >> 13);
  f_sig = (f & 0x007fffffu);
  if constexpr (round_even) {
    if ((f_sig & 0x00003fffu) != 0x00001000u) {
      f_sig += 0x00001000u;
    }
  } else {
    f_sig += 0x00001000u;
  }
  h_sig = (uint16_t)(f_sig >> 13);
  if constexpr (gen_overflow) {
    h_sig += h_exp;
    if (h_sig == 0x7c00u) {
      throw std::overflow_error("");
    }
    return h_sgn + h_sig;
  } else {
    return h_sgn + h_exp + h_sig;
  }
}

constexpr uint32_t ToFloatBits(uint16_t h) {
  uint16_t h_exp = (h & 0x7c00u);
  uint32_t f_sgn = ((uint32_t)h & 0x8000u) << 16;
  switch (h_exp) {
  case 0x0000u: {
    uint16_t h_sig = (h & 0x03ffu);
    if (h_sig == 0) {
      return f_sgn;
    }
    h_sig <<= 1;
    while ((h_sig & 0x0400u) == 0) {
      h_sig <<= 1;
      h_exp++;
    }
    uint32_t f_exp = ((uint32_t)(127 - 15 - h_exp)) << 23;
    uint32_t f_sig = ((uint32_t)(h_sig & 0x03ffu)) << 13;
    return f_sgn + f_exp + f_sig;
  }
  case 0x7c00u:
    return f_sgn + 0x7f800000u + (((uint32_t)(h & 0x03ffu)) << 13);
  default:
    return f_sgn + (((uint32_t)(h & 0x7fffu) + 0x1c000u) << 13);
  }
}

npy_half npy_float_to_half(float f) { return {FromFloatBits(BitCast<uint32_t>(f))}; }
float npy_half_to_float(npy_half h) { return BitCast<float>(ToFloatBits(h.value)); }

template <>
npy_half atomic_fadd<npy_half>(npy_half *loc, npy_half value,
                               std::memory_order) {
  npy_half old_value;
  const std::lock_guard<std::mutex> lock(atomic_op_guard);
  old_value = *loc;
  *loc = npy_float_to_half(npy_half_to_float(old_value) + npy_half_to_float(value));
  return old_value;
}

class AtomicOp {
public:
  AtomicOp(const uint64_t *ptr, size_t numel, std::memory_order order)
      : ptr(ptr), numel(numel), order(order) {}
  void apply() {
    for (size_t i = 0; i < numel; ++i) {
      applyAt(reinterpret_cast<void *>(ptr[i]), i);
    }
  }
  virtual ~AtomicOp() = default;

protected:
  virtual void applyAt(void *, size_t i) = 0;
  const uint64_t *ptr;
  size_t numel;
  std::memory_order order;
};

template <typename DType> class AtomicRMWOpBase : public AtomicOp {
public:
  AtomicRMWOpBase(const uint64_t *ptr, const void *val, void *ret,
                  const bool *mask, size_t numel, std::memory_order order)
      : AtomicOp(ptr, numel, order), val(val), ret(ret), mask(mask) {}

protected:
  void applyAt(void *loc, size_t i) override final {
    if (mask[i]) {
      DType *ptr = static_cast<DType *>(loc);
      *(static_cast<DType *>(ret) + i) =
          applyAtMasked(ptr, *(static_cast<const DType *>(val) + i), this->order);
    }
  }
  virtual DType applyAtMasked(DType *loc, const DType value,
                              std::memory_order order) = 0;
  const void *val;
  void *ret;
  const bool *mask;
};

template <typename DType, RMWInnerOp Op, typename = void>
class AtomicRMWOp : public AtomicRMWOpBase<DType> {
public:
  using AtomicRMWOpBase<DType>::AtomicRMWOpBase;
};

template <typename DType, RMWInnerOp Op>
class AtomicRMWOp<DType, Op, std::enable_if_t<Op == RMWInnerOp::ADD>>
    : public AtomicRMWOpBase<DType> {
public:
  using AtomicRMWOpBase<DType>::AtomicRMWOpBase;

protected:
  DType applyAtMasked(DType *loc, const DType value,
                      std::memory_order order) override {
    DType old_val;
    if constexpr (is_reinterpret_cast_to_atomic_safe<DType>) {
      std::atomic<DType> *atomic_loc =
          reinterpret_cast<std::atomic<DType> *>(loc);
      old_val = std::atomic_fetch_add_explicit(atomic_loc, value, order);
    } else {
      const std::lock_guard<std::mutex> lock(atomic_op_guard);
      old_val = *loc;
      *loc = *loc + value;
    }
    return old_val;
  }
};

template <typename DType, RMWInnerOp Op>
class AtomicRMWOp<DType, Op, std::enable_if_t<Op == RMWInnerOp::FADD>>
    : public AtomicRMWOpBase<DType> {
public:
  using AtomicRMWOpBase<DType>::AtomicRMWOpBase;

protected:
  DType applyAtMasked(DType *loc, const DType value,
                      std::memory_order order) override {
    return atomic_fadd(loc, value, order);
  }
};

template <typename DType, RMWInnerOp Op>
class AtomicRMWOp<DType, Op, std::enable_if_t<Op == RMWInnerOp::AND>>
    : public AtomicRMWOpBase<DType> {
public:
  using AtomicRMWOpBase<DType>::AtomicRMWOpBase;

protected:
  DType applyAtMasked(DType *loc, const DType value,
                      std::memory_order order) override {
    DType old_val;
    if constexpr (is_reinterpret_cast_to_atomic_safe<DType>) {
      std::atomic<DType> *atomic_loc =
          reinterpret_cast<std::atomic<DType> *>(loc);
      old_val = std::atomic_fetch_and_explicit(atomic_loc, value, order);
    } else {
      const std::lock_guard<std::mutex> lock(atomic_op_guard);
      old_val = *loc;
      *loc = *loc & value;
    }
    return old_val;
  }
};

template <typename DType, RMWInnerOp Op>
class AtomicRMWOp<DType, Op, std::enable_if_t<Op == RMWInnerOp::OR>>
    : public AtomicRMWOpBase<DType> {
public:
  using AtomicRMWOpBase<DType>::AtomicRMWOpBase;

protected:
  DType applyAtMasked(DType *loc, const DType value,
                      std::memory_order order) override {
    DType old_val;
    if constexpr (is_reinterpret_cast_to_atomic_safe<DType>) {
      std::atomic<DType> *atomic_loc =
          reinterpret_cast<std::atomic<DType> *>(loc);
      old_val = std::atomic_fetch_or_explicit(atomic_loc, value, order);
    } else {
      const std::lock_guard<std::mutex> lock(atomic_op_guard);
      old_val = *loc;
      *loc = *loc | value;
    }
    return old_val;
  }
};

template <typename DType, RMWInnerOp Op>
class AtomicRMWOp<DType, Op, std::enable_if_t<Op == RMWInnerOp::XOR>>
    : public AtomicRMWOpBase<DType> {
public:
  using AtomicRMWOpBase<DType>::AtomicRMWOpBase;

protected:
  DType applyAtMasked(DType *loc, const DType value,
                      std::memory_order order) override {
    DType old_val;
    if constexpr (is_reinterpret_cast_to_atomic_safe<DType>) {
      std::atomic<DType> *atomic_loc =
          reinterpret_cast<std::atomic<DType> *>(loc);
      old_val = std::atomic_fetch_xor_explicit(atomic_loc, value, order);
    } else {
      const std::lock_guard<std::mutex> lock(atomic_op_guard);
      old_val = *loc;
      *loc = *loc ^ value;
    }
    return old_val;
  }
};

template <typename DType, RMWInnerOp Op>
class AtomicRMWOp<DType, Op,
                  std::enable_if_t<Op == RMWInnerOp::MAX || Op == RMWInnerOp::UMAX>>
    : public AtomicRMWOpBase<DType> {
public:
  using AtomicRMWOpBase<DType>::AtomicRMWOpBase;

protected:
  DType applyAtMasked(DType *loc, const DType value,
                      std::memory_order order) override {
    return atomic_cmp</*is_min=*/false>(loc, value, order);
  }
};

template <typename DType, RMWInnerOp Op>
class AtomicRMWOp<DType, Op,
                  std::enable_if_t<Op == RMWInnerOp::MIN || Op == RMWInnerOp::UMIN>>
    : public AtomicRMWOpBase<DType> {
public:
  using AtomicRMWOpBase<DType>::AtomicRMWOpBase;

protected:
  DType applyAtMasked(DType *loc, const DType value,
                      std::memory_order order) override {
    return atomic_cmp</*is_min=*/true>(loc, value, order);
  }
};

template <typename DType, RMWInnerOp Op>
class AtomicRMWOp<DType, Op, std::enable_if_t<Op == RMWInnerOp::XCHG>>
    : public AtomicRMWOpBase<DType> {
public:
  using AtomicRMWOpBase<DType>::AtomicRMWOpBase;

protected:
  DType applyAtMasked(DType *loc, const DType value,
                      std::memory_order order) override {
    DType old_val;
    if constexpr (is_reinterpret_cast_to_atomic_safe<DType>) {
      std::atomic<DType> *atomic_loc =
          reinterpret_cast<std::atomic<DType> *>(loc);
      old_val = atomic_loc->exchange(value, order);
    } else {
      const std::lock_guard<std::mutex> lock(atomic_op_guard);
      old_val = *loc;
      *loc = value;
    }
    return old_val;
  }
};

template <typename T>
void atomic_compare_exchange_strong(void *loc, void *expected,
                                    const void *desired, size_t i,
                                    std::memory_order order) {
  T desired_val = *(static_cast<const T *>(desired) + i);
  T *expected_uint = static_cast<T *>(expected) + i;

  if constexpr (is_reinterpret_cast_to_atomic_safe<T>) {
    std::atomic<T> *atomic_loc = reinterpret_cast<std::atomic<T> *>(loc);
    atomic_loc->compare_exchange_strong(*expected_uint, desired_val, order,
                                        order);
  } else {
    const std::lock_guard<std::mutex> lock(atomic_op_guard);
    T *atomic_loc = static_cast<T *>(loc);
    if (*atomic_loc == *expected_uint) {
      *atomic_loc = desired_val;
    } else {
      *expected_uint = *atomic_loc;
    }
  }
}

class AtomicCASOp : public AtomicOp {
public:
  AtomicCASOp(const uint64_t *ptr, void *expected, const void *desired,
              size_t itemsize, size_t numel, std::memory_order order)
      : AtomicOp(ptr, numel, order), expected(expected), desired(desired),
        itemsize(itemsize) {}

protected:
  void applyAt(void *loc, size_t i) override {
    if (itemsize == 1) {
      atomic_compare_exchange_strong<uint8_t>(loc, expected, desired, i, this->order);
    } else if (itemsize == 2) {
      atomic_compare_exchange_strong<uint16_t>(loc, expected, desired, i, this->order);
    } else if (itemsize == 4) {
      atomic_compare_exchange_strong<uint32_t>(loc, expected, desired, i, this->order);
    } else if (itemsize == 8) {
      atomic_compare_exchange_strong<uint64_t>(loc, expected, desired, i, this->order);
    } else {
      throw std::invalid_argument("Invalid byte size");
    }
  }

private:
  void *expected;
  const void *desired;
  size_t itemsize;
};

template <RMWInnerOp Op>
std::unique_ptr<AtomicOp>
makeAtomicRMWOp(nanobind::dlpack::dtype dtype, const uint64_t *ptr, const void *val,
                void *ret, const bool *mask, size_t numel,
                std::memory_order order) {
  std::unique_ptr<AtomicOp> atomic_op;
  auto code = (nanobind::dlpack::dtype_code) dtype.code;
  int bits = dtype.bits;

  auto make_int = [&](auto tag) {
    using T = decltype(tag);
    atomic_op = std::make_unique<AtomicRMWOp<T, Op>>(ptr, val, ret, mask, numel, order);
  };

  if constexpr (Op == RMWInnerOp::FADD) {
    if (code == nanobind::dlpack::dtype_code::Float) {
      if (bits == 16)
        atomic_op = std::make_unique<AtomicRMWOp<npy_half, Op>>(ptr, val, ret, mask, numel, order);
      else if (bits == 32)
        atomic_op = std::make_unique<AtomicRMWOp<float, Op>>(ptr, val, ret, mask, numel, order);
      else if (bits == 64)
        atomic_op = std::make_unique<AtomicRMWOp<double, Op>>(ptr, val, ret, mask, numel, order);
    }
  } else if constexpr (Op == RMWInnerOp::ADD || Op == RMWInnerOp::AND || Op == RMWInnerOp::OR ||
                       Op == RMWInnerOp::XOR || Op == RMWInnerOp::XCHG) {
    if (code == nanobind::dlpack::dtype_code::Int) {
      if (bits == 32) make_int(int32_t{});
      else if (bits == 64) make_int(int64_t{});
    } else if (code == nanobind::dlpack::dtype_code::UInt) {
      if (bits == 32) make_int(uint32_t{});
      else if (bits == 64) make_int(uint64_t{});
    }
  } else if constexpr (Op == RMWInnerOp::MAX || Op == RMWInnerOp::MIN) {
    if (code == nanobind::dlpack::dtype_code::Int) {
      if (bits == 32) make_int(int32_t{});
      else if (bits == 64) make_int(int64_t{});
    }
  } else if constexpr (Op == RMWInnerOp::UMAX || Op == RMWInnerOp::UMIN) {
    if (code == nanobind::dlpack::dtype_code::UInt) {
      if (bits == 32) make_int(uint32_t{});
      else if (bits == 64) make_int(uint64_t{});
    }
  }

  if (!atomic_op)
    throw std::invalid_argument("Unsupported data type for requested RMW op");
  return atomic_op;
}

} // namespace

void init_triton_interpreter(nb::module_ &&m) {
  using nb::ndarray;

  nb::enum_<MemSemantic>(m, "MEM_SEMANTIC")
      .value("ACQUIRE_RELEASE", MemSemantic::ACQUIRE_RELEASE)
      .value("ACQUIRE", MemSemantic::ACQUIRE)
      .value("RELEASE", MemSemantic::RELEASE)
      .value("RELAXED", MemSemantic::RELAXED)
      .export_values();

  nb::enum_<RMWOp>(m, "RMW_OP")
      .value("ADD", RMWOp::ADD)
      .value("FADD", RMWOp::FADD)
      .value("AND", RMWOp::AND)
      .value("OR", RMWOp::OR)
      .value("XOR", RMWOp::XOR)
      .value("XCHG", RMWOp::XCHG)
      .value("MAX", RMWOp::MAX)
      .value("MIN", RMWOp::MIN)
      .value("UMIN", RMWOp::UMIN)
      .value("UMAX", RMWOp::UMAX)
      .export_values();

  m.def("load",
        [](ndarray<uint64_t, nb::c_contig> ptr,
           ndarray<bool, nb::c_contig> mask,
           ndarray<> other,
           nb::handle /*ret_dtype*/) -> ndarray<> {
          size_t numel = ptr.size();
          size_t ndim = ptr.ndim();
          size_t itemsize = ((size_t) other.dtype().bits + 7) / 8;

          std::vector<size_t> shape(ndim);
          for (size_t i = 0; i < ndim; ++i)
            shape[i] = ptr.shape(i);

          size_t nbytes = itemsize * numel;
          void *data = std::malloc(nbytes);
          if (!data)
            throw std::bad_alloc();
          nb::capsule owner(data, [](void *p) noexcept { std::free(p); });

          ndarray<uint8_t> ret((uint8_t *) data, ndim, shape.data(), owner,
                               nullptr, other.dtype(), nb::device::cpu::value);

          const uint64_t *p_ptr = ptr.data();
          const bool *p_mask = mask.data();
          const uint8_t *p_other = (const uint8_t *) other.data();
          uint8_t *p_out = ret.data();

          for (size_t i = 0; i < numel; ++i) {
            if (p_mask[i]) {
              std::memcpy(p_out + i * itemsize, (void *) (uintptr_t) p_ptr[i], itemsize);
            } else {
              std::memcpy(p_out + i * itemsize, p_other + i * itemsize, itemsize);
            }
          }

          return ndarray<>(ret);
        });

  m.def("store",
        [](ndarray<uint64_t, nb::c_contig> ptr,
           ndarray<> value,
           ndarray<bool, nb::c_contig> mask) {
          size_t numel = ptr.size();
          size_t itemsize = ((size_t) value.dtype().bits + 7) / 8;

          const uint64_t *p_ptr = ptr.data();
          const bool *p_mask = mask.data();
          const uint8_t *p_val = (const uint8_t *) value.data();

          for (size_t i = 0; i < numel; ++i) {
            if (p_mask[i]) {
              std::memcpy((void *) (uintptr_t) p_ptr[i], p_val + i * itemsize, itemsize);
            }
          }
        });

  m.def("atomic_rmw",
        [](RMWOp rmw_op,
           ndarray<uint64_t, nb::c_contig> ptr,
           ndarray<> val,
           ndarray<bool, nb::c_contig> mask,
           MemSemantic sem) -> ndarray<> {
          std::memory_order order = mem_semantic_map[sem];
          size_t numel = ptr.size();
          size_t itemsize = ((size_t) val.dtype().bits + 7) / 8;
          size_t ndim = ptr.ndim();

          std::vector<size_t> shape(ndim);
          for (size_t i = 0; i < ndim; ++i)
            shape[i] = ptr.shape(i);

          void *data = std::malloc(itemsize * numel);
          if (!data)
            throw std::bad_alloc();
          nb::capsule owner(data, [](void *p) noexcept { std::free(p); });
          ndarray<uint8_t> ret((uint8_t *) data, ndim, shape.data(), owner,
                               nullptr, val.dtype(), nb::device::cpu::value);

          const uint64_t *ptr_data = ptr.data();
          const bool *mask_data = mask.data();
          const void *val_data = val.data();
          void *ret_data = ret.data();

          std::unique_ptr<AtomicOp> atomic_op;
          switch (rmw_op) {
          case RMWOp::ADD:
            atomic_op = makeAtomicRMWOp<RMWInnerOp::ADD>(val.dtype(), ptr_data, val_data, ret_data, mask_data, numel, order);
            break;
          case RMWOp::FADD:
            atomic_op = makeAtomicRMWOp<RMWInnerOp::FADD>(val.dtype(), ptr_data, val_data, ret_data, mask_data, numel, order);
            break;
          case RMWOp::AND:
            atomic_op = makeAtomicRMWOp<RMWInnerOp::AND>(val.dtype(), ptr_data, val_data, ret_data, mask_data, numel, order);
            break;
          case RMWOp::OR:
            atomic_op = makeAtomicRMWOp<RMWInnerOp::OR>(val.dtype(), ptr_data, val_data, ret_data, mask_data, numel, order);
            break;
          case RMWOp::XOR:
            atomic_op = makeAtomicRMWOp<RMWInnerOp::XOR>(val.dtype(), ptr_data, val_data, ret_data, mask_data, numel, order);
            break;
          case RMWOp::MAX:
            atomic_op = makeAtomicRMWOp<RMWInnerOp::MAX>(val.dtype(), ptr_data, val_data, ret_data, mask_data, numel, order);
            break;
          case RMWOp::UMAX:
            atomic_op = makeAtomicRMWOp<RMWInnerOp::UMAX>(val.dtype(), ptr_data, val_data, ret_data, mask_data, numel, order);
            break;
          case RMWOp::MIN:
            atomic_op = makeAtomicRMWOp<RMWInnerOp::MIN>(val.dtype(), ptr_data, val_data, ret_data, mask_data, numel, order);
            break;
          case RMWOp::UMIN:
            atomic_op = makeAtomicRMWOp<RMWInnerOp::UMIN>(val.dtype(), ptr_data, val_data, ret_data, mask_data, numel, order);
            break;
          case RMWOp::XCHG:
            atomic_op = makeAtomicRMWOp<RMWInnerOp::XCHG>(val.dtype(), ptr_data, val_data, ret_data, mask_data, numel, order);
            break;
          }

          atomic_op->apply();
          return ndarray<>(ret);
        });

  m.def("atomic_cas",
        [](ndarray<uint64_t, nb::c_contig> ptr,
           ndarray<> cmp,
           ndarray<> val,
           MemSemantic sem) -> ndarray<> {
          std::memory_order order = mem_semantic_map[sem];
          size_t numel = ptr.size();
          size_t itemsize = ((size_t) cmp.dtype().bits + 7) / 8;
          size_t ndim = ptr.ndim();

          std::vector<size_t> shape(ndim);
          for (size_t i = 0; i < ndim; ++i)
            shape[i] = ptr.shape(i);

          void *data = std::malloc(itemsize * numel);
          if (!data)
            throw std::bad_alloc();
          nb::capsule owner(data, [](void *p) noexcept { std::free(p); });
          ndarray<uint8_t> ret((uint8_t *) data, ndim, shape.data(), owner,
                               nullptr, cmp.dtype(), nb::device::cpu::value);

          std::memcpy(ret.data(), cmp.data(), itemsize * numel);
          AtomicCASOp(ptr.data(), ret.data(), val.data(), itemsize, numel, order).apply();
          return ndarray<>(ret);
        });
}
