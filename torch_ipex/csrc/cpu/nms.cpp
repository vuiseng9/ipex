// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "ExtendOPs.h"
#include <ATen/Parallel.h>
#include <algorithm>
#include <c10/util/Exception.h>
#include <torch/csrc/autograd/function.h>
#include "torch_ipex/csrc/autocast_mode.h"
#include "torch_ipex/csrc/autocast_verbose.h"
namespace torch_ipex {

/*
 When calculating the Intersection over Union:
  MaskRCNN: bias = 1
  SSD-Resnet34: bias = 0
*/
template <typename scalar_t>
at::Tensor nms_cpu_kernel(const at::Tensor& dets,
                          const at::Tensor& scores,
                          const float threshold, float bias=1) {
  AT_ASSERTM(!dets.type().is_cuda(), "dets must be a CPU tensor");
  AT_ASSERTM(!scores.type().is_cuda(), "scores must be a CPU tensor");
  AT_ASSERTM(dets.type() == scores.type(), "dets should have the same type as scores");

  if (dets.numel() == 0) {
    return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
  }

  auto x1_t = dets.select(1, 0).contiguous();
  auto y1_t = dets.select(1, 1).contiguous();
  auto x2_t = dets.select(1, 2).contiguous();
  auto y2_t = dets.select(1, 3).contiguous();

  at::Tensor areas_t = (x2_t - x1_t + bias) * (y2_t - y1_t + bias);

  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));

  auto ndets = dets.size(0);
  at::Tensor suppressed_t = at::zeros({ndets}, dets.options().dtype(at::kByte).device(at::kCPU));

  auto suppressed = suppressed_t.data<uint8_t>();
  auto order = order_t.data<int64_t>();
  auto x1 = x1_t.data<scalar_t>();
  auto y1 = y1_t.data<scalar_t>();
  auto x2 = x2_t.data<scalar_t>();
  auto y2 = y2_t.data<scalar_t>();
  auto areas = areas_t.data<scalar_t>();
  for (int64_t _i = 0; _i < ndets; _i++) {
    auto i = order[_i];
    if (suppressed[i] == 1)
      continue;
    auto ix1 = x1[i];
    auto iy1 = y1[i];
    auto ix2 = x2[i];
    auto iy2 = y2[i];
    auto iarea = areas[i];

#ifdef _OPENMP
#if (_OPENMP >= 201307)
# pragma omp parallel for simd schedule(static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#else
# pragma omp parallel for schedule(static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#endif
#endif
    for (int64_t _j = _i + 1; _j < ndets; _j++) {
      auto j = order[_j];
      if (suppressed[j] == 1)
        continue;
      auto xx1 = std::max(ix1, x1[j]);
      auto yy1 = std::max(iy1, y1[j]);
      auto xx2 = std::min(ix2, x2[j]);
      auto yy2 = std::min(iy2, y2[j]);

      auto w = std::max(static_cast<scalar_t>(0), xx2 - xx1 + bias);
      auto h = std::max(static_cast<scalar_t>(0), yy2 - yy1 + bias);
      auto inter = w * h;
      auto ovr = inter / (iarea + areas[j] - inter);
      if (ovr >= threshold)
        suppressed[j] = 1;
   }
  }
  return at::nonzero(suppressed_t == 0).squeeze(1);
}

std::vector<at::Tensor> remove_empty(std::vector<at::Tensor>& candidate, int64_t start, int64_t end) {
  std::vector<at::Tensor> valid_candidate;
  for (int64_t i = start; i < end; i++) {
    if (candidate[i].defined()) {
      valid_candidate.push_back(candidate[i]);
    }
  }
  return valid_candidate;
}

template <typename scalar_t>
std::vector<std::tuple<at::Tensor, at::Tensor, at::Tensor>> batch_score_nms_kernel(const at::Tensor& batch_dets,
                          const at::Tensor& batch_scores,
                          const float threshold, const int max_output=200) {
  // Reference to: https://github.com/mlcommons/inference/blob/0f096a18083c3fd529c1fbf97ebda7bc3f1fda70/others/cloud/single_stage_detector/pytorch/utils.py#L163
  // batch_dets: (batchsize, num_bbox, 4) For example: batch_dets: (1, 15130, 4)
  // batch_scores: (batchsize, num_bbox, label_num) For example: batch_scores: (1, 15130, 81)
  auto nbatch = batch_scores.size(0); // number of batches
  auto ndets = batch_scores.size(1); // number of boxes
  auto nscore = batch_scores.size(2); // number of labels

  auto nbatch_x_nscore = nbatch * nscore; // (number of batches) * (number of labels)
  std::vector<at::Tensor> bboxes_out(nbatch_x_nscore);
  std::vector<at::Tensor> scores_out(nbatch_x_nscore);
  std::vector<at::Tensor> labels_out(nbatch_x_nscore);

#ifdef _OPENMP
#if (_OPENMP >= 201307)
# pragma omp parallel for simd schedule(static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#else
# pragma omp parallel for schedule(static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#endif
#endif
  // skip background (i = 0)
  for(int index = 0; index < nbatch_x_nscore; index++){
    // Parallel in the dimentaion of: batch * nscore
    auto bs = index / nscore;
    auto i = index % nscore;

    // skip background (i = 0)
    if(i == 0){
      continue;
    }

    at::Tensor dets = batch_dets[bs].squeeze(0); // dets for boxes per image: (num_bbox, 4); For example: (15130, 4)
    at::Tensor scores = batch_scores[bs].squeeze(0); // scores for boxes per image: (num_bbox, 81); For example: (15130, 81)

    at::Tensor score = scores.slice(1, i, i+1).squeeze(1); // score for boxes per image per class: (num_bbox); For example: (15130)

    at::Tensor mask_index = at::nonzero(score > 0.05).squeeze(1);
    at::Tensor bboxes = at::index_select(dets, /*dim*/0, mask_index);
    score = at::index_select(score, /*dim*/0, mask_index);

    if (score.size(0) == 0) {
      continue;
    }

    at::Tensor score_sorted, score_idx_sorted;
    std::tie(score_sorted, score_idx_sorted) = score.sort(0);

    // # select max_output indices
    score_idx_sorted = score_idx_sorted.slice(/*dim*/0, /*start*/std::max(score.size(0) - max_output, static_cast<int64_t>(0)), /*end*/score.size(0));

    at::Tensor keep = nms_cpu_kernel<scalar_t>(at::index_select(bboxes, /*dim*/0, score_idx_sorted), at::index_select(score, /*dim*/0, score_idx_sorted), threshold, /*bias*/0);
    at::Tensor candidates = at::index_select(score_idx_sorted, /*dim*/0, keep);

    bboxes_out[index] = at::index_select(bboxes, /*dim*/0, candidates);
    scores_out[index] = at::index_select(score, /*dim*/0, candidates);
    // TODO optimize the fill_
    labels_out[index] = at::empty({candidates.sizes()}).fill_(i);
  }

  std::vector<std::tuple<at::Tensor, at::Tensor, at::Tensor>> output(nbatch);
#ifdef _OPENMP
#if (_OPENMP >= 201307)
# pragma omp parallel for simd schedule(static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#else
# pragma omp parallel for schedule(static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#endif
#endif
  for(int bs = 0; bs < nbatch; bs++){
    // Post process the tensors to get the top max_output(number) for each Batchsize
    std::vector<at::Tensor> valid_bboxes_out = remove_empty(bboxes_out, bs*nscore, (bs+1)*nscore);
    std::vector<at::Tensor> valid_scores_out = remove_empty(scores_out, bs*nscore, (bs+1)*nscore);
    std::vector<at::Tensor> valid_labels_out = remove_empty(labels_out, bs*nscore, (bs+1)*nscore);

    at::Tensor bboxes_out_ = at::cat(valid_bboxes_out, 0);
    at::Tensor labels_out_ = at::cat(valid_labels_out, 0);
    at::Tensor scores_out_ = at::cat(valid_scores_out, 0);

    std::tuple<at::Tensor, at::Tensor> sort_result = scores_out_.sort(0);
    at::Tensor max_ids = std::get<1>(sort_result);
    max_ids = max_ids.slice(/*dim*/0, /*start*/std::max(max_ids.size(0) - max_output, static_cast<int64_t>(0)), /*end*/max_ids.size(0));
    output[bs] = std::tuple<at::Tensor, at::Tensor, at::Tensor>(bboxes_out_.index_select(/*dim*/0, /*index*/max_ids),
                                                                labels_out_.index_select(/*dim*/0, /*index*/max_ids),
                                                                scores_out_.index_select(/*dim*/0, /*index*/max_ids));
  }
  return output;
}

at::Tensor nms_cpu(const at::Tensor& dets,
               const at::Tensor& scores,
               const float threshold) {
  at::Tensor result;
  AT_DISPATCH_FLOATING_TYPES(dets.type(), "nms", [&] {
    result = nms_cpu_kernel<scalar_t>(dets, scores, threshold);
  });
  return result;
}

std::vector<std::tuple<at::Tensor, at::Tensor, at::Tensor>> batch_score_nms_cpu(const at::Tensor& dets,
               const at::Tensor& scores,
               const float threshold,
               const int max_output) {
  std::vector<std::tuple<at::Tensor, at::Tensor, at::Tensor>> result;
  AT_DISPATCH_FLOATING_TYPES(dets.type(), "batch_score_nms", [&] {
    result = batch_score_nms_kernel<scalar_t>(dets, scores, threshold, max_output);
  });
  return result;
}

at::Tensor AtenIpexTypeExt::nms(const at::Tensor& dets,
               const at::Tensor& scores,
               const double threshold) {
#if defined(IPEX_DISP_OP)
  printf("IpexExternal::nms\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("IpexExternal::nms", std::vector<c10::IValue>({}));
#endif
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dets.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(scores.layout() == c10::kStrided);
  auto&& result = nms_cpu(dets, scores, threshold);
  static_cast<void>(result); // Avoid warnings in case not used
  return result;
}

std::vector<std::tuple<at::Tensor, at::Tensor, at::Tensor>> AtenIpexTypeExt::batch_score_nms(const at::Tensor& dets,
               const at::Tensor& scores,
               const double threshold,
               const int64_t max_output) {
#if defined(IPEX_DISP_OP)
  printf("IpexExternal::batch_score_nms\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("IpexExternal::batch_score_nms", std::vector<c10::IValue>({}));
#endif
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dets.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(scores.layout() == c10::kStrided);
  auto&& result = batch_score_nms_cpu(dets, scores, threshold, max_output);
  static_cast<void>(result); // Avoid warnings in case not used
  return result;
}

template <typename scalar_t>
at::Tensor scale_back_batch_kernel(const at::Tensor& _ipex_bboxes_in,
                                         const at::Tensor& _ipex_dboxes_xywh,
                                         const float scale_xy,
                                         const float scale_wh) {
  //_ipex_bboxes_in: [BS, number_boxes, 4], for example: [1, 15130, 4]
  auto _ipex_bboxes_in_conti = _ipex_bboxes_in.contiguous();
  auto _ipex_dboxes_xywh_conti = _ipex_dboxes_xywh.contiguous();
  int64_t batch_size = _ipex_bboxes_in.size(0);
  int64_t boxes_per_image = _ipex_bboxes_in.size(1);
  int64_t ndets = batch_size * boxes_per_image; // batchsize * boxes per image
  at::Tensor output = at::empty({_ipex_bboxes_in.size(0), _ipex_bboxes_in.size(1), _ipex_bboxes_in.size(2)}, _ipex_bboxes_in.options());
  auto output_conti = output.contiguous();

  auto* input_data = _ipex_bboxes_in_conti.data_ptr<scalar_t>();
  auto* output_data = output_conti.data_ptr<scalar_t>();
  auto* input_dboxes_xywh_data = _ipex_dboxes_xywh_conti.data_ptr<double>();

#ifdef _OPENMP
#if (_OPENMP >= 201307)
# pragma omp parallel for simd schedule(static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#else
# pragma omp parallel for schedule(static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#endif
#endif
  for(int64_t k = 0; k < ndets; k++){
    int64_t i = k / boxes_per_image;
    int64_t j = k % boxes_per_image;

    int64_t index = i * boxes_per_image * 4 + j * 4;

    // bboxes_in[:, :, :2] = self.scale_xy*bboxes_in[:, :, :2]
    output_data[index] = input_data[index] * scale_xy;
    output_data[index+1] = input_data[index+1] * scale_xy;
    // bboxes_in[:, :, 2:] = self.scale_wh*bboxes_in[:, :, 2:]
    output_data[index+2] = input_data[index+2] * scale_wh;
    output_data[index+3] = input_data[index+3] * scale_wh;

    int64_t index_dboxes_xywh = j * 4;
    // bboxes_in[:, :, :2] = bboxes_in[:, :, :2]*self.dboxes_xywh[:, :, 2:] + self.dboxes_xywh[:, :, :2]
    output_data[index] = output_data[index] * input_dboxes_xywh_data[index_dboxes_xywh+2] + input_dboxes_xywh_data[index_dboxes_xywh];
    output_data[index+1] = output_data[index+1] * input_dboxes_xywh_data[index_dboxes_xywh+3] + input_dboxes_xywh_data[index_dboxes_xywh+1];
    // bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp()*self.dboxes_xywh[:, :, 2:]
    output_data[index+2] = exp(output_data[index+2])*input_dboxes_xywh_data[index_dboxes_xywh+2];
    output_data[index+3] = exp(output_data[index+3])*input_dboxes_xywh_data[index_dboxes_xywh+3];

    /*
    # Transform format to ltrb
    l, t, r, b = bboxes_in[:, :, 0] - 0.5*bboxes_in[:, :, 2],\
                  bboxes_in[:, :, 1] - 0.5*bboxes_in[:, :, 3],\
                  bboxes_in[:, :, 0] + 0.5*bboxes_in[:, :, 2],\
                  bboxes_in[:, :, 1] + 0.5*bboxes_in[:, :, 3]

    bboxes_in[:, :, 0] = l
    bboxes_in[:, :, 1] = t
    bboxes_in[:, :, 2] = r
    bboxes_in[:, :, 3] = b
    */

    auto l = output_data[index] - 0.5 * output_data[index+2];
    auto t = output_data[index+1] - 0.5 * output_data[index+3];
    auto r = output_data[index] + 0.5 * output_data[index+2];
    auto b = output_data[index+1] + 0.5 * output_data[index+3];
    output_data[index] = l;
    output_data[index+1] = t;
    output_data[index+2] = r;
    output_data[index+3] = b;
  }
  return output;
}

std::tuple<at::Tensor, at::Tensor> AtenIpexTypeExt::parallel_scale_back_batch(const at::Tensor& bboxes_in,
                                                                                 const at::Tensor& scores_in,
                                                                                 const at::Tensor& dboxes_xywh,
                                                                                 const double scale_xy,
                                                                                 const double scale_wh){
#if defined(IPEX_DISP_OP)
  printf("IpexExternal::parallel_scale_back_batch\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("IpexExternal::parallel_scale_back_batch", std::vector<c10::IValue>({}));
#endif
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(bboxes_in.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dboxes_xywh.layout() == c10::kStrided);

  at::Tensor bbox_result;
  AT_DISPATCH_FLOATING_TYPES(bboxes_in.type(), "scale_back_batch", [&] {
    bbox_result = scale_back_batch_kernel<scalar_t>(bboxes_in, dboxes_xywh, scale_xy, scale_wh);
  });

  auto&& scores_result = at::softmax(scores_in, -1);

  return std::tuple<at::Tensor, at::Tensor>(bbox_result, scores_result);
}
} // namespace torch_ipex


namespace {
static auto dispatch =
    torch::RegisterOperators()
        .op("torch_ipex::nms", &torch_ipex::AtenIpexTypeExt::nms)
        .op("torch_ipex::batch_score_nms", &torch_ipex::AtenIpexTypeExt::batch_score_nms)
        .op("torch_ipex::parallel_scale_back_batch", &torch_ipex::AtenIpexTypeExt::parallel_scale_back_batch);
}

namespace torch_ipex {
namespace autocast {

at::Tensor nms(const at::Tensor& dets,
               const at::Tensor& scores,
               const double threshold) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op = torch::Dispatcher::singleton()
    .findSchemaOrThrow("torch_ipex::nms", "")
    .typed<decltype(nms)>();
#if defined(ENABLE_AUTOCAST_VERBOSE)
  verbose::OpNameGuard op_name("nms");
#endif
  return op.call(dets, cpu_cached_cast(at::kFloat, scores), threshold);
}

std::vector<std::tuple<at::Tensor, at::Tensor, at::Tensor>> batch_score_nms(const at::Tensor& dets,
                           const at::Tensor& scores,
                           const double threshold,
                           const int64_t max_output) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op = torch::Dispatcher::singleton()
    .findSchemaOrThrow("torch_ipex::batch_score_nms", "")
    .typed<decltype(batch_score_nms)>();
#if defined(ENABLE_AUTOCAST_VERBOSE)
  verbose::OpNameGuard op_name("batch_score_nms");
#endif
  return op.call(dets, cpu_cached_cast(at::kFloat, scores), threshold, max_output);
}

std::tuple<at::Tensor, at::Tensor> parallel_scale_back_batch(const at::Tensor& bboxes_in,
                                                                const at::Tensor& scores_in,
                                                                const at::Tensor& dboxes_xywh,
                                                                const double scale_xy,
                                                                const double scale_wh){
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op = torch::Dispatcher::singleton()
    .findSchemaOrThrow("torch_ipex::parallel_scale_back_batch", "")
    .typed<decltype(parallel_scale_back_batch)>();
#if defined(ENABLE_AUTOCAST_VERBOSE)
  verbose::OpNameGuard op_name("parallel_scale_back_batch");
#endif
  return op.call(cpu_cached_cast(at::kFloat, bboxes_in), cpu_cached_cast(at::kFloat, scores_in),
                 cpu_cached_cast(at::kFloat, dboxes_xywh), scale_xy, scale_wh);
}

TORCH_LIBRARY_IMPL(torch_ipex, AutocastCPU, m){
  m.impl("nms", torch_ipex::autocast::nms);
  m.impl("batch_score_nms", torch_ipex::autocast::batch_score_nms);
  m.impl("parallel_scale_back_batch", torch_ipex::autocast::parallel_scale_back_batch);
}

} // namespace autocast
} // namespace torch_ipex
