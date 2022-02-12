#include "ConcatBnRelu.h"

#include "csrc/utils/ipex_op_profile.h"

#if defined(CPU_CAPABILITY_AVX512)
#include "csrc/cpu/vec512/concat_bn_relu.h"
#endif
#include <torch/csrc/autograd/function.h>

namespace torch_ipex {
namespace cpu {
using Tensor = at::Tensor;

/**
 * This kernel fuses Concat+BN+ReLU as a signel operator.
 * The conditions are set as all the input tensors
 * should have the same dimension (4D or 5D), sizes,
 * ChannelsLast(3d) memory format and data type (float).
 * All the inputs should be 4D or 5D tensors. The condition
 * check should be done on graph rewrite operation before
 * calling this kernel.
 **/
Tensor ConcatBnRelu(
    const c10::List<Tensor>& a,
    const Tensor& bn_beta,
    const c10::optional<Tensor>& bn_scale,
    const c10::optional<Tensor>& bn_bias,
    const c10::optional<Tensor>& bn_mean,
    const c10::optional<Tensor>& bn_var,
    bool bn_training,
    double bn_momentum,
    double bn_eps,
    bool bn_cudnn_enabled,
    int dim) {
  IPEX_RECORD_FUNCTION("ConcatBnRelu", std::vector<c10::IValue>({}));

  int64_t list_length = a.size();

  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(bn_scale);
  const Tensor& bn_weight = *weight_maybe_owned;
  std::vector<long int> output_dim(a[0].ndimension());
  for (int64_t i = 0; i < list_length; ++i) {
    output_dim[1] += a[i].size(1);
  }
  for (int64_t i = 0; i < a[0].ndimension(); ++i) {
    if (i != 1) {
      output_dim[i] = a[0].size(i);
    }
  }
  Tensor output = at::empty(
      output_dim,
      a[0].options()
          .dtype(at::kFloat)
          .memory_format(a[0].suggest_memory_format()));

#if defined(CPU_CAPABILITY_AVX512)
  torch_ipex::cpu::kernel::vec::vec512::ConcatBnReluKernelImpl_ChannelsLast<
      float>(a, bn_weight, bn_beta, output);
  return output;
#else
  std::vector<Tensor> concat_input(list_length);
  for (int64_t i = 0; i < list_length; ++i)
    concat_input[i] = a[i];
  auto bn_res = at::batch_norm(
      at::cat(concat_input, (int64_t)dim),
      bn_scale,
      bn_bias,
      bn_mean,
      bn_var,
      bn_training,
      bn_momentum,
      bn_eps,
      bn_cudnn_enabled);
  return at::relu(bn_res);
#endif
}

} // namespace cpu
} // namespace torch_ipex