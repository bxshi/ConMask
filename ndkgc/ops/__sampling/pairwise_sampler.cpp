//
// Created by Baoxu Shi on 3/15/17.
//

#include <unordered_set>
#include <random>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/util/guarded_philox_random.h"

using namespace tensorflow;

using ::shape_inference::DimensionHandle;
using ::shape_inference::InferenceContext;
using ::shape_inference::ShapeHandle;

using ::gtl::ArraySlice;
using ::gtl::MutableArraySlice;
using ::random;

namespace {
  Status PairwiseSamplingShapeFn(InferenceContext *c) {
    int32 num_true;
    int32 num_false;
    int32 max_range;
    TF_RETURN_IF_ERROR(c->GetAttr("num_true", &num_true));
    TF_RETURN_IF_ERROR(c->GetAttr("num_false", &num_false));
    TF_RETURN_IF_ERROR(c->GetAttr("max_range", &max_range));
    // extract batch size using input entities
    auto batch_size = c->Dim(c->input(0), 0);
    ShapeHandle true_target_shape = c->Matrix(batch_size, num_true);
    ShapeHandle false_target_shape = c->Matrix(batch_size, num_false);
    c->set_output(0, true_target_shape);
    c->set_output(1, false_target_shape);
    return Status::OK();
  }

  Status SingleNegativeSamplingShapeFn(InferenceContext *c) {
    c->set_output(0, c->Scalar());
    return Status::OK();
  }

  Status MultipleNegativeSamplingShapeFn(InferenceContext *c) {
    int32 num_sampled;
    TF_RETURN_IF_ERROR(c->GetAttr("num_sampled", &num_sampled));
    c->set_output(0, c->Vector(num_sampled));
    return Status::OK();
  }

  class MultipleNegativeSamplingOp : public OpKernel {
  private:
    int _num_sampled;
    int _max_range;
  public:
    explicit MultipleNegativeSamplingOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("num_sampled", &_num_sampled));
        OP_REQUIRES_OK(context, context->GetAttr("max_range", &_max_range));
    }
    void Compute(OpKernelContext *context) override {
        const Tensor &input_targets = context->input(0);
        OP_REQUIRES(context, TensorShapeUtils::IsVector(input_targets.shape()),
                    errors::InvalidArgument("targets must be a vector"));
        gtl::ArraySlice<int32> targets(input_targets.vec<int32>().data(),
                                       static_cast<size_t>(input_targets.dim_size(0)));
        Tensor *output_false_targets = nullptr;
        TensorShape output_shape;
        output_shape.AddDim(_num_sampled);
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_false_targets));

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int32> distribution(0, _max_range);

        std::unordered_set<int32> avoids;
        avoids.insert(targets.begin(), targets.end());
        int num_sampled = 0;
        while (num_sampled < _num_sampled) {
          int sampled = distribution(gen);
          if (gtl::InsertIfNotPresent(&avoids, sampled)) {
            *(output_false_targets->vec<int32>().data() + num_sampled) = sampled;
            ++num_sampled;
          }
        }
    }
  };

  class SingleNegativeSamplingOp : public OpKernel {
  private:
      int _max_range;
  public:
      explicit SingleNegativeSamplingOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("max_range", &_max_range));
      }

      void Compute(OpKernelContext *context) override {
        const Tensor &input_targets = context->input(0);
        OP_REQUIRES(context, TensorShapeUtils::IsVector(input_targets.shape()),
                    errors::InvalidArgument("targets must be a vector"));
        gtl::ArraySlice<int32> targets(input_targets.vec<int32>().data(),
                                       static_cast<size_t>(input_targets.dim_size(0)));
        Tensor *output_false_target = nullptr;
        TensorShape output_shape;
        output_shape.AddDim(1);
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_false_target));

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int32> distribution(0, _max_range);

        std::unordered_set<int32> avoids;
        avoids.insert(targets.begin(), targets.end());
        int num_sampled = 0;
        while (num_sampled == 0) {
          int sampled = distribution(gen);
          if (gtl::InsertIfNotPresent(&avoids, sampled)) {
            *(output_false_target->vec<int32>().data()) = sampled;
            ++num_sampled;
          }
        }
      }
  };

  class PairwiseSamplingOp : public OpKernel {
  private:
      int _max_range;
      int _num_true;
      int _num_false;
  public:
      explicit PairwiseSamplingOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("max_range", &_max_range));
        OP_REQUIRES_OK(context, context->GetAttr("num_true", &_num_true));
        OP_REQUIRES_OK(context, context->GetAttr("num_false", &_num_false));
      }

      void Compute(OpKernelContext *context) override {
        // Sanity check on inputs
        const Tensor &input_known_entities = context->input(0);
        OP_REQUIRES(context, TensorShapeUtils::IsVector(input_known_entities.shape()),
                    errors::InvalidArgument("known_entities must be a vector"));
        const Tensor &input_targets = context->input(1);
        OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_targets.shape()),
                    errors::InvalidArgument("targets must be a matrix"));

        const Tensor &input_num_targets = context->input(2);
        OP_REQUIRES(context, TensorShapeUtils::IsVector(input_num_targets.shape()),
                    errors::InvalidArgument("num_targets must be a vector"));

        const int64 batch_size = input_known_entities.dim_size(0);
        Tensor *output_true_targets = nullptr;
        TensorShape output_true_targets_shape;
        // create shape of [batch_size, _num_true]
        output_true_targets_shape.AddDim(batch_size);
        output_true_targets_shape.AddDim(_num_true);
        OP_REQUIRES_OK(context, context->allocate_output(0, output_true_targets_shape, &output_true_targets));

        Tensor *output_false_targets = nullptr;
        TensorShape output_false_targets_shape;
        output_false_targets_shape.AddDim(batch_size);
        output_false_targets_shape.AddDim(_num_false);
        OP_REQUIRES_OK(context, context->allocate_output(1, output_false_targets_shape, &output_false_targets));
        /*
          sampling happens here
          for training instance i(^th row)
          1. Sample num_true elements from the [i, 0:num_targets[i]] with replacement
          2. Sample num_false elements from [0, max_range) avoiding elements in
              [i, 0:num_targets[i]] and known_entities[i].
        */

        // sample true elements
        gtl::ArraySlice<int32> known_entities(input_known_entities.vec<int32>().data(),
                                              static_cast<size_t>(batch_size));
        gtl::ArraySlice<int32> num_targets(input_num_targets.vec<int32>().data(),
                                           static_cast<size_t>(input_num_targets.dim_size(0)));
        const int32 *targets = input_targets.matrix<int32>().data();
        std::random_device rd;
        std::mt19937 gen(rd());

        // pointer to a [batch_size, num_true] matrix
        int32 *true_targets = output_true_targets->matrix<int32>().data();
        const int64 row_length = input_targets.dim_size(1);
        for (int i = 0; i < batch_size; ++i) {
          // dist between [0, num_targets[i])
          std::uniform_int_distribution<int32> distribution(0, num_targets[i] - 1);
          // must have less true targets than maximum number of target columns
          OP_REQUIRES(context, row_length > num_targets[i],
                      errors::InvalidArgument("num_targets value is larger than the max number of columns in targets"));
          OP_REQUIRES(context, num_targets[i] >= 0,
                      errors::InvalidArgument("num_targets must be non-negative values"));
          for (int j = 0; j < _num_true; ++j) {
            *(true_targets + (i * 2) + j) = *(targets + row_length * i + distribution(gen));
          }
        }

        // sample false elements
        int32 *false_targets = output_false_targets->matrix<int32>().data();
        std::uniform_int_distribution<int32> neg_dist(0, _max_range);
        for (int i = 0; i < batch_size; ++i) {
          // avoid known entity and all positive targets
          std::unordered_set<int32> avoid_values;
          avoid_values.insert(known_entities[i]);
          for (int true_idx = 0; true_idx < num_targets[i]; ++true_idx) {
            avoid_values.insert(*(targets + row_length * i + true_idx));
          }

          // sample at most num_false elements
          OP_REQUIRES(context, avoid_values.size() + _num_false < _max_range,
                      errors::InvalidArgument(
                          "There is not enough elements to sample. Please make sure the number of true targets is smaller than max range"));
          int sampled = 0;
          while (sampled < _num_false) {
            int32 sampled_value = neg_dist(gen);
            if (gtl::InsertIfNotPresent(&avoid_values, sampled_value)) {
              *(false_targets + (i * 2) + sampled) = sampled_value;
              ++sampled;
            }
          }
        }

      }
  };

}

REGISTER_KERNEL_BUILDER(Name("SingleNegativeSampling").Device(DEVICE_CPU), SingleNegativeSamplingOp);
REGISTER_OP("SingleNegativeSampling")
    .Input("targets: int32")
    .Attr("max_range: int")
    .Output("false_target: int32")
    .SetShapeFn(SingleNegativeSamplingShapeFn)
    .Doc(R"doc(
Sample one negative example in range[0, max_range] that does not overlaps with targets.
)doc");

REGISTER_KERNEL_BUILDER(Name("MultipleNegativeSampling").Device(DEVICE_CPU), MultipleNegativeSamplingOp);
REGISTER_OP("MultipleNegativeSampling")
    .Input("targets: int32")
    .Attr("max_range: int")
    .Attr("num_sampled: int")
    .Output("false_target: int32")
    .SetShapeFn(MultipleNegativeSamplingShapeFn)
    .Doc(R"doc(
Sample `num_sampled` negative example in range[0, max_range] that does not overlaps with targets.
)doc");

REGISTER_KERNEL_BUILDER(Name("PairwiseSampling").Device(DEVICE_CPU), PairwiseSamplingOp);
REGISTER_OP("PairwiseSampling")
    .Input("known_entities: int32")
    .Input("targets: int32")
    .Input("num_targets: int32")
    .Attr("max_range: int")
    .Attr("num_true: int")
    .Attr("num_false: int")
    .Output("true_targets: int32")
    .Output("false_targets: int32")
    .SetShapeFn(PairwiseSamplingShapeFn)
    .Doc(R"doc(
Generate sampled positive and negative labels for pairwise training.

  Given an padded target matrix targets [batch_size, ?] and num_targets [batch_size,] indicating
the true number of targets per row. We randomly sample `num_true` positive examples and
`num_false` negative examples.

`true_targets` is a [batch_size, num_true] matrix of positive targets
`false_targets` is a [batch_size, num_false] matrix of negative targets

)doc");

