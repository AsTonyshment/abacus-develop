#ifndef ATEN_CORE_TENSOR_RESERVE_H_
#define ATEN_CORE_TENSOR_RESERVE_H_

#include "source_base/module_container/ATen/core/tensor.h"

#include <algorithm>
#include <cstdint>

namespace container
{
//! Reserve typed device scratch storage, retaining a valid pointer for an empty partition.
template <typename T, typename Device>
void reserve_buffer(Tensor* buffer, const int64_t size)
{
    using CtDevice = typename PsiToContainer<Device>::type;
    const DeviceType device = DeviceTypeToEnum<CtDevice>::value;
    const DataType dtype = DataTypeToEnum<T>::value;
    const int64_t elements = std::max<int64_t>(1, size);
    if (buffer->NumElements() < elements || buffer->data_type() != dtype || buffer->device_type() != device)
    {
        *buffer = Tensor(dtype, device, {elements});
    }
}
} // namespace container

#endif
