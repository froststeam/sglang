#pragma once
#include <sgl_kernel/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>
#if defined(USE_MUSA) || defined(__MUSACC__)
#include <musa_bf16.h>
#endif

#if defined(__CUDACC__) || defined(__MUSACC__)
#include <sgl_kernel/utils.cuh>
#endif

namespace host {

namespace details {

static constexpr int kAnyDeviceID = -1;
static constexpr int64_t kAnySize = static_cast<int64_t>(-1);
static constexpr int64_t kNullSize = static_cast<int64_t>(-1);
static constexpr DLDataTypeCode kNullDType = static_cast<DLDataTypeCode>(18u);
static constexpr DLDeviceType kNullDevice = static_cast<DLDeviceType>(-1);

struct SizeRef;
struct DTypeRef;
struct DeviceRef;

template <typename T, typename Enable = void>
struct _dtype_trait {};

template <typename T>
struct _dtype_trait<T, typename std::enable_if<std::is_integral<T>::value>::type> {
  static constexpr DLDataType value = DLDataType{
      std::is_signed<T>::value ? DLDataTypeCode::kDLInt : DLDataTypeCode::kDLUInt,
      static_cast<std::uint8_t>(sizeof(T) * 8),
      1};
};

template <typename T>
struct _dtype_trait<T, typename std::enable_if<std::is_floating_point<T>::value>::type> {
  static constexpr DLDataType value = DLDataType{
      DLDataTypeCode::kDLFloat,
      static_cast<std::uint8_t>(sizeof(T) * 8),
      1};
};
#if defined(USE_MUSA) || defined(__MUSACC__)
template <>
struct _dtype_trait<__mt_bfloat16, void> {
  static constexpr DLDataType value = DLDataType{DLDataTypeCode::kDLBfloat, 16, 1};
};
#endif

#if defined(__CUDACC__) || defined(__MUSACC__)
template <>
struct _dtype_trait<fp16_t, void> {
  static constexpr DLDataType value = DLDataType{DLDataTypeCode::kDLFloat, 16, 1};
};

#if !defined(__MUSACC__) && !defined(USE_MUSA)
template <>
struct _dtype_trait<bf16_t, void> {
  static constexpr DLDataType value = DLDataType{DLDataTypeCode::kDLBfloat, 16, 1};
};
#endif

template <>
struct _dtype_trait<fp8_e4m3_t, void> {
  static constexpr DLDataType value = DLDataType{DLDataTypeCode::kDLFloat8_e4m3fn, 8, 1};
};
#endif

template <DLDeviceType Code>
struct _device_trait {
  static constexpr DLDevice value = DLDevice{Code, kAnyDeviceID};
};

template <typename... Ts>
struct DTypeListHolder {
  static constexpr std::array<DLDataType, sizeof...(Ts)> value = {{
      _dtype_trait<Ts>::value...
  }};
};

template <typename... Ts>
constexpr std::array<DLDataType, sizeof...(Ts)> DTypeListHolder<Ts...>::value;

template <DLDeviceType... Codes>
struct DeviceListHolder {
  static constexpr std::array<DLDevice, sizeof...(Codes)> value = {{
      _device_trait<Codes>::value...
  }};
};

template <DLDeviceType... Codes>
constexpr std::array<DLDevice, sizeof...(Codes)> DeviceListHolder<Codes...>::value;

template <typename T>
struct PrintAbleSpan {
  explicit PrintAbleSpan(const std::vector<T>& data) : ptr(&data) {}
  const std::vector<T>* ptr;
};

inline const char* device_type_to_string(DLDeviceType type) {
  switch (type) {
    case DLDeviceType::kDLCPU:
      return "cpu";
    case DLDeviceType::kDLCUDA:
      return "cuda";
    case DLDeviceType::kDLCUDAHost:
      return "cuda_host";
    case DLDeviceType::kDLOpenCL:
      return "opencl";
    case DLDeviceType::kDLVulkan:
      return "vulkan";
    case DLDeviceType::kDLMetal:
      return "metal";
    case DLDeviceType::kDLVPI:
      return "vpi";
    case DLDeviceType::kDLROCM:
      return "rocm";
    case DLDeviceType::kDLROCMHost:
      return "rocm_host";
    case DLDeviceType::kDLExtDev:
      return "ext_dev";
    case DLDeviceType::kDLCUDAManaged:
      return "cuda_managed";
    case DLDeviceType::kDLOneAPI:
      return "oneapi";
    case DLDeviceType::kDLWebGPU:
      return "webgpu";
    case DLDeviceType::kDLHexagon:
      return "hexagon";
    case DLDeviceType::kDLMAIA:
      return "maia";
    case DLDeviceType::kDLTrn:
      return "trn";
    default:
      return nullptr;
  }
}

struct PrintableDevice {
  DLDevice device;
};

inline std::ostream& operator<<(std::ostream& os, DLDevice device) {
  const char* name = device_type_to_string(device.device_type);
  RuntimeCheck(name != nullptr, "Unknown device: ", int(device.device_type));
  os << name;
  if (device.device_id != kAnyDeviceID && device.device_type != DLDeviceType::kDLCPU) {
    os << ":" << device.device_id;
  }
  return os;
}

inline std::ostream& operator<<(std::ostream& os, PrintableDevice pd) {
  return os << pd.device;
}

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const PrintAbleSpan<T>& span) {
  os << "[";
  for (std::size_t i = 0; i < span.ptr->size(); ++i) {
    if (i > 0) {
      os << ", ";
    }
    os << (*span.ptr)[i];
  }
  os << "]";
  return os;
}

}  // namespace details

template <typename T>
inline bool is_type(DLDataType dtype) {
  return dtype == details::_dtype_trait<T>::value;
}

struct SymbolicSize {
 public:
  explicit SymbolicSize(std::string_view annotation = std::string_view())
      : m_value(details::kNullSize), m_annotation(annotation) {}

  SymbolicSize(const SymbolicSize&) = delete;
  SymbolicSize& operator=(const SymbolicSize&) = delete;

  std::string_view get_name() const {
    return m_annotation;
  }

  void set_value(int64_t value) {
    RuntimeCheck(!this->has_value(), "Size value already set");
    m_value = value;
  }

  bool has_value() const {
    return m_value != details::kNullSize;
  }

  std::optional<int64_t> get_value() const {
    return this->has_value() ? std::optional<int64_t>(m_value) : std::optional<int64_t>();
  }

  int64_t unwrap(DebugInfo info = {}) const {
    RuntimeCheck(info, this->has_value(), "Size value is not set");
    return m_value;
  }

  void verify(int64_t value, const char* prefix, int64_t dim) {
    if (this->has_value()) {
      if (m_value != value) {
        Panic("Size mismatch for ", m_name_str(prefix, dim), ": expected ", m_value, " but got ", value);
      }
    } else {
      this->set_value(value);
    }
  }

  std::string value_or_name(const char* prefix, int64_t dim) const {
    if (this->has_value()) {
      return std::to_string(m_value);
    } else {
      return m_name_str(prefix, dim);
    }
  }

 private:
  std::string m_name_str(const char* prefix, int64_t dim) const {
    std::ostringstream os;
    os << prefix << '#' << dim;
    if (!m_annotation.empty()) {
      os << "('" << m_annotation << "')";
    }
    return os.str();
  }

  std::int64_t m_value;
  std::string_view m_annotation;
};

inline bool operator==(DLDevice lhs, DLDevice rhs) {
  return lhs.device_type == rhs.device_type && lhs.device_id == rhs.device_id;
}

struct SymbolicDType {
 public:
  SymbolicDType() : m_value(DLDataType{details::kNullDType, 0, 0}) {}

  SymbolicDType(const SymbolicDType&) = delete;
  SymbolicDType& operator=(const SymbolicDType&) = delete;

  void set_value(DLDataType value) {
    RuntimeCheck(!this->has_value(), "Dtype value already set");
    RuntimeCheck(
        m_check(value),
        "Dtype value [",
        value,
        "] not in the allowed options: ",
        details::PrintAbleSpan<DLDataType>(m_options));
    m_value = value;
  }

  bool has_value() const {
    return m_value.code != details::kNullDType;
  }

  std::optional<DLDataType> get_value() const {
    return this->has_value() ? std::optional<DLDataType>(m_value) : std::optional<DLDataType>();
  }

  DLDataType unwrap(DebugInfo info = {}) const {
    RuntimeCheck(info, this->has_value(), "Dtype value is not set");
    return m_value;
  }

  void set_options(const std::vector<DLDataType>& options) {
    m_options = options;
  }

  void set_options(std::initializer_list<DLDataType> options) {
    m_options.assign(options.begin(), options.end());
  }

  template <typename... Ts>
  void set_options() {
    m_options.assign(
        details::DTypeListHolder<Ts...>::value.begin(),
        details::DTypeListHolder<Ts...>::value.end());
  }

  void verify(DLDataType dtype) {
    if (this->has_value()) {
      RuntimeCheck(m_value == dtype, "DType mismatch: expected ", m_value, " but got ", dtype);
    } else {
      this->set_value(dtype);
    }
  }

  template <typename T>
  bool is_type() const {
    return ::host::is_type<T>(m_value);
  }

 private:
  bool m_check(DLDataType value) const {
    if (m_options.empty()) {
      return true;
    }
    return std::find(m_options.begin(), m_options.end(), value) != m_options.end();
  }

  std::vector<DLDataType> m_options;
  DLDataType m_value;
};

struct SymbolicDevice {
 public:
  SymbolicDevice() : m_value(DLDevice{details::kNullDevice, details::kAnyDeviceID}) {}

  SymbolicDevice(const SymbolicDevice&) = delete;
  SymbolicDevice& operator=(const SymbolicDevice&) = delete;

  void set_value(DLDevice value) {
    RuntimeCheck(!this->has_value(), "Device value already set");
    RuntimeCheck(
        m_check(value),
        "Device value [",
        details::PrintableDevice{value},
        "] not in the allowed options: ",
        details::PrintAbleSpan<DLDevice>(m_options));
    m_value = value;
  }

  bool has_value() const {
    return m_value.device_type != details::kNullDevice;
  }

  std::optional<DLDevice> get_value() const {
    return this->has_value() ? std::optional<DLDevice>(m_value) : std::optional<DLDevice>();
  }

  DLDevice unwrap(DebugInfo info = {}) const {
    RuntimeCheck(info, this->has_value(), "Device value is not set");
    return m_value;
  }

  void set_options(const std::vector<DLDevice>& options) {
    m_options = options;
  }

  void set_options(std::initializer_list<DLDevice> options) {
    m_options.assign(options.begin(), options.end());
  }

  template <DLDeviceType... Codes>
  void set_options() {
    m_options.assign(
        details::DeviceListHolder<Codes...>::value.begin(),
        details::DeviceListHolder<Codes...>::value.end());
  }

  void verify(DLDevice device) {
    if (this->has_value()) {
      RuntimeCheck(
          m_value == device,
          "Device mismatch: expected ",
          details::PrintableDevice{m_value},
          " but got ",
          details::PrintableDevice{device});
    } else {
      this->set_value(device);
    }
  }

 private:
  bool m_check(DLDevice value) const {
    if (m_options.empty()) {
      return true;
    }
    for (std::size_t i = 0; i < m_options.size(); ++i) {
      const DLDevice& opt = m_options[i];
      if (opt.device_type != value.device_type) {
        continue;
      }
      if (opt.device_id == details::kAnyDeviceID || opt.device_id == value.device_id) {
        return true;
      }
    }
    return false;
  }

  std::vector<DLDevice> m_options;
  DLDevice m_value;
};

namespace details {

template <typename T>
struct BaseRef {
 public:
  BaseRef() : m_ref(&m_cache), m_cache() {}
  explicit BaseRef(T& value) : m_ref(&value), m_cache() {}

  BaseRef(const BaseRef&) = delete;
  BaseRef& operator=(const BaseRef&) = delete;

  BaseRef(BaseRef&& other) noexcept : m_ref(&m_cache), m_cache() {
    if (other.m_ref == &other.m_cache) {
      m_ref = &m_cache;
      if (const auto value = other.m_cache.get_value(); value.has_value()) {
        m_cache.set_value(*value);
      }
    } else {
      m_ref = other.m_ref;
    }
  }

  BaseRef& operator=(BaseRef&& other) noexcept {
    if (this != &other) {
      if (other.m_ref == &other.m_cache) {
        m_ref = &m_cache;
        m_cache = T{};
        if (const auto value = other.m_cache.get_value(); value.has_value()) {
          m_cache.set_value(*value);
        }
      } else {
        m_ref = other.m_ref;
      }
    }
    return *this;
  }

  T* operator->() const {
    return m_ref;
  }

  T& operator*() const {
    return *m_ref;
  }

  void rebind(T& other) {
    m_ref = &other;
  }

 private:
  T* m_ref;
  T m_cache;
};

struct SizeRef : BaseRef<SymbolicSize> {
  SizeRef() : BaseRef<SymbolicSize>() {}

  SizeRef(SymbolicSize& size) : BaseRef<SymbolicSize>(size) {}

  SizeRef(int64_t value) : BaseRef<SymbolicSize>() {
    if (value != kAnySize) {
      (**this).set_value(value);
    }
  }
};

struct DTypeRef : BaseRef<SymbolicDType> {
  DTypeRef() : BaseRef<SymbolicDType>() {}

  DTypeRef(SymbolicDType& dtype) : BaseRef<SymbolicDType>(dtype) {}

  explicit DTypeRef(DLDataType option) : BaseRef<SymbolicDType>() {
    (**this).set_value(option);
  }

  DTypeRef(std::initializer_list<DLDataType> options) : BaseRef<SymbolicDType>() {
    (**this).set_options(options);
  }

  explicit DTypeRef(const std::vector<DLDataType>& options) : BaseRef<SymbolicDType>() {
    (**this).set_options(options);
  }
};

struct DeviceRef : BaseRef<SymbolicDevice> {
  DeviceRef() : BaseRef<SymbolicDevice>() {}

  DeviceRef(SymbolicDevice& device) : BaseRef<SymbolicDevice>(device) {}

  explicit DeviceRef(DLDevice option) : BaseRef<SymbolicDevice>() {
    (**this).set_value(option);
  }

  DeviceRef(std::initializer_list<DLDevice> options) : BaseRef<SymbolicDevice>() {
    (**this).set_options(options);
  }

  explicit DeviceRef(const std::vector<DLDevice>& options) : BaseRef<SymbolicDevice>() {
    (**this).set_options(options);
  }
};

}  // namespace details

struct TensorMatcher {
 private:
  typedef details::SizeRef SizeRef;
  typedef details::DTypeRef DTypeRef;
  typedef details::DeviceRef DeviceRef;

 public:
  TensorMatcher(const TensorMatcher&) = delete;
  TensorMatcher& operator=(const TensorMatcher&) = delete;

  explicit TensorMatcher(std::initializer_list<SizeRef> shape)
      : m_shape(),
        m_strides(),
        m_dtype(),
        m_device(),
        m_has_dtype(false),
        m_has_device(false) {
    m_shape.reserve(shape.size());
    for (const SizeRef& ref : shape) {
      m_shape.push_back(m_clone_size_ref(ref));
    }
  }

  TensorMatcher(TensorMatcher&& other) noexcept
      : m_shape(std::move(other.m_shape)),
        m_strides(std::move(other.m_strides)),
        m_dtype(),
        m_device(),
        m_has_dtype(other.m_has_dtype),
        m_has_device(other.m_has_device) {
    m_dtype.rebind(*other.m_dtype);
    m_device.rebind(*other.m_device);
  }

  TensorMatcher& operator=(TensorMatcher&& other) noexcept {
    if (this != &other) {
      m_shape = std::move(other.m_shape);
      m_strides = std::move(other.m_strides);
      m_has_dtype = other.m_has_dtype;
      m_has_device = other.m_has_device;
      m_dtype.rebind(*other.m_dtype);
      m_device.rebind(*other.m_device);
    }
    return *this;
  }

  TensorMatcher&& with_strides(std::initializer_list<SizeRef> strides) && {
    RuntimeCheck(m_strides.empty(), "Strides already specified");
    RuntimeCheck(m_shape.size() == strides.size(), "Strides size must match shape size");
    m_strides.reserve(strides.size());
    for (const SizeRef& ref : strides) {
      m_strides.push_back(m_clone_size_ref(ref));
    }
    return std::move(*this);
  }

  template <typename... Ts>
  TensorMatcher&& with_dtype(DTypeRef&& dtype) && {
    m_init_dtype();
    m_dtype.rebind(*dtype);
    m_dtype->template set_options<Ts...>();
    return std::move(*this);
  }

  template <typename... Ts>
  TensorMatcher&& with_dtype() && {
    static_assert(sizeof...(Ts) > 0, "At least one dtype option must be specified");
    m_init_dtype();
    m_dtype->template set_options<Ts...>();
    return std::move(*this);
  }

  template <DLDeviceType... Codes>
  TensorMatcher&& with_device(DeviceRef&& device) && {
    m_init_device();
    m_device.rebind(*device);
    m_device->template set_options<Codes...>();
    return std::move(*this);
  }

  template <DLDeviceType... Codes>
  TensorMatcher&& with_device() && {
    static_assert(sizeof...(Codes) > 0, "At least one device option must be specified");
    m_init_device();
    m_device->template set_options<Codes...>();
    return std::move(*this);
  }

  const TensorMatcher&& verify(tvm::ffi::TensorView view, DebugInfo info = {}) const && {
    try {
      m_verify_impl(view);
    } catch (PanicError& e) {
      std::ostringstream oss;
      oss << "Tensor match failed for ";
      s_print_tensor(oss, view);
      oss << " at " << info.file_name() << ":" << info.line() << "\n- Root cause: " << e.root_cause();
      throw PanicError(oss.str());
    }
    return std::move(*this);
  }

 private:
  static SizeRef m_clone_size_ref(const SizeRef& ref) {
    if (const auto value = (*ref).get_value(); value.has_value()) {
      return SizeRef(*value);
    }
    return SizeRef(*ref);
  }

  static void s_print_tensor(std::ostringstream& oss, tvm::ffi::TensorView view) {
    oss << "Tensor<";
    int64_t dim = 0;
    for (const auto& size : view.shape()) {
      if (dim++ > 0) {
        oss << ", ";
      }
      oss << size;
    }
    oss << ">[strides=<";
    dim = 0;
    for (const auto& stride : view.strides()) {
      if (dim++ > 0) {
        oss << ", ";
      }
      oss << stride;
    }
    oss << ">, dtype=" << view.dtype();
    oss << ", device=" << details::PrintableDevice{view.device()} << "]";
  }

  void m_verify_impl(tvm::ffi::TensorView view) const {
    const std::size_t dim = static_cast<std::size_t>(view.dim());
    RuntimeCheck(dim == m_shape.size(), "Tensor dimension mismatch: expected ", m_shape.size(), " but got ", dim);

    for (std::size_t i = 0; i < dim; ++i) {
      m_shape[i]->verify(view.size(static_cast<int64_t>(i)), "shape", static_cast<int64_t>(i));
    }

    if (m_has_strides()) {
      for (std::size_t i = 0; i < dim; ++i) {
        if (view.size(static_cast<int64_t>(i)) != 1 || !m_strides[i]->has_value()) {
          m_strides[i]->verify(view.stride(static_cast<int64_t>(i)), "stride", static_cast<int64_t>(i));
        }
      }
    } else {
      RuntimeCheck(view.is_contiguous(), "Tensor is not contiguous as expected");
    }

    m_dtype->verify(view.dtype());
    m_device->verify(view.device());
  }

  void m_init_dtype() {
    RuntimeCheck(!m_has_dtype, "DType already specified");
    m_has_dtype = true;
  }

  void m_init_device() {
    RuntimeCheck(!m_has_device, "Device already specified");
    m_has_device = true;
  }

  bool m_has_strides() const {
    return !m_strides.empty();
  }

  std::vector<SizeRef> m_shape;
  std::vector<SizeRef> m_strides;
  DTypeRef m_dtype;
  DeviceRef m_device;
  bool m_has_dtype;
  bool m_has_device;
};

}  // namespace host
