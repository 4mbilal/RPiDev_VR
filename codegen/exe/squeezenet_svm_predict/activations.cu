//
// File: activations.cu
//
// GPU Coder version                    : 1.5
// CUDA/C/C++ source code generated on  : 06-Aug-2020 05:06:39
//

// Include Files
#include "activations.h"
#include "DeepLearningNetwork.h"
#include "MWCudaDimUtility.hpp"
#include "rt_nonfinite.h"
#include "squeezenet_svm_predict.h"

// Function Declarations
static __global__ void b_DeepLearningNetwork_activatio(const unsigned char in
  [154587], unsigned char b_in[154587]);
static __global__ void c_DeepLearningNetwork_activatio(const unsigned char in
  [154587], float miniBatchT[154587]);
static __global__ void d_DeepLearningNetwork_activatio(const float outMiniBatch
  [1000], float outT[1000]);

// Function Definitions

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const unsigned char in[154587]
//                unsigned char b_in[154587]
// Return Type  : void
//
static __global__ __launch_bounds__(512, 1) void b_DeepLearningNetwork_activatio
  (const unsigned char in[154587], unsigned char b_in[154587])
{
  unsigned int threadId;
  int i;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  i = static_cast<int>(threadId);
  if (i < 154587) {
    b_in[i] = in[i];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const unsigned char in[154587]
//                float miniBatchT[154587]
// Return Type  : void
//
static __global__ __launch_bounds__(512, 1) void c_DeepLearningNetwork_activatio
  (const unsigned char in[154587], float miniBatchT[154587])
{
  unsigned int threadId;
  int p;
  int i1;
  int i;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  i = static_cast<int>(threadId % 227U);
  threadId = (threadId - static_cast<unsigned int>(i)) / 227U;
  i1 = static_cast<int>(threadId % 227U);
  threadId = (threadId - static_cast<unsigned int>(i1)) / 227U;
  p = static_cast<int>(threadId);
  if (p < 3) {
    miniBatchT[(i + 227 * i1) + 51529 * p] = static_cast<float>(in[(i1 + 227 * i)
      + 51529 * p]);
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const float outMiniBatch[1000]
//                float outT[1000]
// Return Type  : void
//
static __global__ __launch_bounds__(512, 1) void d_DeepLearningNetwork_activatio
  (const float outMiniBatch[1000], float outT[1000])
{
  unsigned int threadId;
  int p;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  p = static_cast<int>(threadId);
  if (p < 1000) {
    outT[p] = outMiniBatch[p];
  }
}

//
// Arguments    : b_squeezenet_0 *obj
//                const unsigned char in[154587]
//                float outT[1000]
// Return Type  : void
//
void DeepLearningNetwork_activations(b_squeezenet_0 *obj, const unsigned char
  in[154587], float outT[1000])
{
  float (*gpu_miniBatchT)[154587];
  float (*gpu_outMiniBatch)[1000];
  unsigned char (*gpu_in)[154587];
  unsigned char (*b_gpu_in)[154587];
  float (*gpu_outT)[1000];
  cudaMalloc(&gpu_outT, 4000UL);
  cudaMalloc(&gpu_outMiniBatch, 4000UL);
  cudaMalloc(&gpu_miniBatchT, 618348UL);
  cudaMalloc(&b_gpu_in, 154587UL);
  cudaMalloc(&gpu_in, 154587UL);
  cudaMemcpy(gpu_in, (void *)&in[0], 154587UL, cudaMemcpyHostToDevice);
  b_DeepLearningNetwork_activatio<<<dim3(302U, 1U, 1U), dim3(512U, 1U, 1U)>>>
    (*gpu_in, *b_gpu_in);
  c_DeepLearningNetwork_activatio<<<dim3(302U, 1U, 1U), dim3(512U, 1U, 1U)>>>
    (*b_gpu_in, *gpu_miniBatchT);
  cudaMemcpy(obj->getInputDataPointer(), *gpu_miniBatchT, 154587U * sizeof(float),
             cudaMemcpyDeviceToDevice);
  obj->activations(66);
  cudaMemcpy(*gpu_outMiniBatch, obj->getLayerOutput(66, 0), 1000U * sizeof(float),
             cudaMemcpyDeviceToDevice);
  d_DeepLearningNetwork_activatio<<<dim3(2U, 1U, 1U), dim3(512U, 1U, 1U)>>>
    (*gpu_outMiniBatch, *gpu_outT);
  cudaMemcpy(&outT[0], gpu_outT, 4000UL, cudaMemcpyDeviceToHost);
  cudaFree(*gpu_in);
  cudaFree(*b_gpu_in);
  cudaFree(*gpu_miniBatchT);
  cudaFree(*gpu_outMiniBatch);
  cudaFree(*gpu_outT);
}

//
// File trailer for activations.cu
//
// [EOF]
//
