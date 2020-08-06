//
// File: squeezenet_svm_predict.cu
//
// GPU Coder version                    : 1.5
// CUDA/C/C++ source code generated on  : 06-Aug-2020 05:06:39
//

// Include Files
#include "squeezenet_svm_predict.h"
#include "CompactClassificationECOC.h"
#include "DeepLearningNetwork.h"
#include "MWCudaDimUtility.hpp"
#include "activations.h"
#include "math_constants.h"
#include "rt_nonfinite.h"
#include "squeezenet_svm_predict_data.h"
#include "squeezenet_svm_predict_initialize.h"
#include <cstring>

// Type Definitions
struct c_classreg_learning_coder_class
{
  double ClassNames[7];
  int ClassNamesLength[7];
  double Prior[7];
  double Cost[49];
  double CodingMatrix[147];
  char ScoreType[3];
};

struct struct_T
{
  c_classreg_learning_coder_class classifier;
};

// Variable Definitions
static b_squeezenet_0 net;
static bool net_not_empty;
static struct_T v;
__device__ struct_T gpu_v;

// Function Declarations
static __device__ double rt_powd_snf(double u0, double u1);
static __device__ double rt_roundd_snf(double u);
static __global__ void squeezenet_svm_predict_kernel1(const double dv[7]);
static __global__ void squeezenet_svm_predict_kernel10(const double rowWeights
  [2724], const int xoffset, double rowWeightsTotal[227]);
static __global__ void squeezenet_svm_predict_kernel11(const double colWeights
  [2043], double colWeightsTotal[227]);
static __global__ void squeezenet_svm_predict_kernel12(const double colWeights
  [2043], const int xoffset, double colWeightsTotal[227]);
static __global__ void squeezenet_svm_predict_kernel13(const double
  rowWeightsTotal[227], const double rowWeights[2724], const unsigned char in
  [921600], const short ipRowIndices[2724], unsigned char partialResize[326880]);
static __global__ void squeezenet_svm_predict_kernel14(const double
  colWeightsTotal[227], const double colWeights[2043], const unsigned char
  partialResize[326880], const short ipColIndices[2043], unsigned char out
  [154587]);
static __global__ void squeezenet_svm_predict_kernel2(const signed char iv[49]);
static __global__ void squeezenet_svm_predict_kernel3(const signed char iv1[147]);
static __global__ void squeezenet_svm_predict_kernel4(const char cv[3]);
static __global__ void squeezenet_svm_predict_kernel5(short aux1[1280]);
static __global__ void squeezenet_svm_predict_kernel6(short aux2[960]);
static __global__ void squeezenet_svm_predict_kernel7(const short aux1[1280],
  double rowWeights[2724], short ipRowIndices[2724]);
static __global__ void squeezenet_svm_predict_kernel8(const short aux2[960],
  double colWeights[2043], short ipColIndices[2043]);
static __global__ void squeezenet_svm_predict_kernel9(const double rowWeights
  [2724], double rowWeightsTotal[227]);

// Function Definitions

//
// Arguments    : double u0
//                double u1
// Return Type  : double
//
static __device__ double rt_powd_snf(double u0, double u1)
{
  double b_y;
  if ((static_cast<int>(isnan(u0))) || (static_cast<int>(isnan(u1)))) {
    b_y = CUDART_NAN;
  } else {
    double d;
    double d1;
    d = fabs(u0);
    d1 = fabs(u1);
    if (isinf(u1)) {
      if (d == 1.0) {
        b_y = 1.0;
      } else if (d > 1.0) {
        if (u1 > 0.0) {
          b_y = CUDART_INF;
        } else {
          b_y = 0.0;
        }
      } else if (u1 > 0.0) {
        b_y = 0.0;
      } else {
        b_y = CUDART_INF;
      }
    } else if (d1 == 0.0) {
      b_y = 1.0;
    } else if (d1 == 1.0) {
      if (u1 > 0.0) {
        b_y = u0;
      } else {
        b_y = 1.0 / u0;
      }
    } else if (u1 == 2.0) {
      b_y = u0 * u0;
    } else if ((static_cast<int>(u1 == 0.5)) && (static_cast<int>(u0 >= 0.0))) {
      b_y = sqrt(u0);
    } else if ((static_cast<int>(u0 < 0.0)) && (static_cast<int>(u1 > floor(u1))))
    {
      b_y = CUDART_NAN;
    } else {
      b_y = pow(u0, u1);
    }
  }

  return b_y;
}

//
// Arguments    : double u
// Return Type  : double
//
static __device__ double rt_roundd_snf(double u)
{
  double b_y;
  if (fabs(u) < 4.503599627370496E+15) {
    if (u >= 0.5) {
      b_y = floor(u + 0.5);
    } else if (u > -0.5) {
      b_y = u * 0.0;
    } else {
      b_y = ceil(u - 0.5);
    }
  } else {
    b_y = u;
  }

  return b_y;
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double dv[7]
// Return Type  : void
//
static __global__ __launch_bounds__(32, 1) void squeezenet_svm_predict_kernel1(
  const double dv[7])
{
  unsigned int threadId;
  int oldIdx;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  oldIdx = static_cast<int>(threadId);
  if (oldIdx < 7) {
    gpu_v.classifier.ClassNames[oldIdx] = static_cast<double>(oldIdx) + 1.0;
    gpu_v.classifier.ClassNamesLength[oldIdx] = 1;
    gpu_v.classifier.Prior[oldIdx] = dv[oldIdx];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double rowWeights[2724]
//                const int xoffset
//                double rowWeightsTotal[227]
// Return Type  : void
//
static __global__ __launch_bounds__(256, 1) void squeezenet_svm_predict_kernel10
  (const double rowWeights[2724], const int xoffset, double rowWeightsTotal[227])
{
  unsigned int threadId;
  int oldIdx;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  oldIdx = static_cast<int>(threadId);
  if (oldIdx < 227) {
    rowWeightsTotal[oldIdx] += rowWeights[xoffset + oldIdx];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double colWeights[2043]
//                double colWeightsTotal[227]
// Return Type  : void
//
static __global__ __launch_bounds__(256, 1) void squeezenet_svm_predict_kernel11
  (const double colWeights[2043], double colWeightsTotal[227])
{
  unsigned int threadId;
  int oldIdx;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  oldIdx = static_cast<int>(threadId);
  if (oldIdx < 227) {
    colWeightsTotal[oldIdx] = colWeights[oldIdx];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double colWeights[2043]
//                const int xoffset
//                double colWeightsTotal[227]
// Return Type  : void
//
static __global__ __launch_bounds__(256, 1) void squeezenet_svm_predict_kernel12
  (const double colWeights[2043], const int xoffset, double colWeightsTotal[227])
{
  unsigned int threadId;
  int oldIdx;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  oldIdx = static_cast<int>(threadId);
  if (oldIdx < 227) {
    colWeightsTotal[oldIdx] += colWeights[xoffset + oldIdx];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double rowWeightsTotal[227]
//                const double rowWeights[2724]
//                const unsigned char in[921600]
//                const short ipRowIndices[2724]
//                unsigned char partialResize[326880]
// Return Type  : void
//
static __global__ __launch_bounds__(512, 1) void squeezenet_svm_predict_kernel13
  (const double rowWeightsTotal[227], const double rowWeights[2724], const
   unsigned char in[921600], const short ipRowIndices[2724], unsigned char
   partialResize[326880])
{
  unsigned int threadId;
  int colIdx;
  int rowIdx;
  double sumVal;
  int oldIdx;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  oldIdx = static_cast<int>(threadId % 3U);
  threadId = (threadId - static_cast<unsigned int>(oldIdx)) / 3U;
  rowIdx = static_cast<int>(threadId % 227U);
  threadId = (threadId - static_cast<unsigned int>(rowIdx)) / 227U;
  colIdx = static_cast<int>(threadId);
  if (colIdx < 480) {
    unsigned char u;
    sumVal = 0.0;
    for (int l = 0; l < 12; l++) {
      sumVal += static_cast<double>(in[((static_cast<int>(ipRowIndices[rowIdx +
        227 * l]) + 640 * colIdx) + 307200 * oldIdx) - 1]) * (rowWeights[rowIdx
        + 227 * l] / rowWeightsTotal[rowIdx]);
    }

    sumVal = rt_roundd_snf(sumVal);
    if (sumVal < 256.0) {
      if (sumVal >= 0.0) {
        u = static_cast<unsigned char>(sumVal);
      } else {
        u = static_cast<unsigned char>(0U);
      }
    } else if (sumVal >= 256.0) {
      u = MAX_uint8_T;
    } else {
      u = static_cast<unsigned char>(0U);
    }

    partialResize[(rowIdx + 227 * colIdx) + 108960 * oldIdx] = u;
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double colWeightsTotal[227]
//                const double colWeights[2043]
//                const unsigned char partialResize[326880]
//                const short ipColIndices[2043]
//                unsigned char out[154587]
// Return Type  : void
//
static __global__ __launch_bounds__(512, 1) void squeezenet_svm_predict_kernel14
  (const double colWeightsTotal[227], const double colWeights[2043], const
   unsigned char partialResize[326880], const short ipColIndices[2043], unsigned
   char out[154587])
{
  unsigned int threadId;
  int colIdx;
  int rowIdx;
  double sumVal;
  int oldIdx;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  oldIdx = static_cast<int>(threadId % 3U);
  threadId = (threadId - static_cast<unsigned int>(oldIdx)) / 3U;
  rowIdx = static_cast<int>(threadId % 227U);
  threadId = (threadId - static_cast<unsigned int>(rowIdx)) / 227U;
  colIdx = static_cast<int>(threadId);
  if (colIdx < 227) {
    unsigned char u;
    sumVal = 0.0;
    for (int l = 0; l < 9; l++) {
      sumVal += static_cast<double>(partialResize[(rowIdx + 227 * (static_cast<
        int>(ipColIndices[colIdx + 227 * l]) - 1)) + 108960 * oldIdx]) *
        (colWeights[colIdx + 227 * l] / colWeightsTotal[colIdx]);
    }

    sumVal = rt_roundd_snf(sumVal);
    if (sumVal < 256.0) {
      if (sumVal >= 0.0) {
        u = static_cast<unsigned char>(sumVal);
      } else {
        u = static_cast<unsigned char>(0U);
      }
    } else if (sumVal >= 256.0) {
      u = MAX_uint8_T;
    } else {
      u = static_cast<unsigned char>(0U);
    }

    out[(rowIdx + 227 * colIdx) + 51529 * oldIdx] = u;
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const signed char iv[49]
// Return Type  : void
//
static __global__ __launch_bounds__(64, 1) void squeezenet_svm_predict_kernel2(
  const signed char iv[49])
{
  unsigned int threadId;
  int oldIdx;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  oldIdx = static_cast<int>(threadId);
  if (oldIdx < 49) {
    gpu_v.classifier.Cost[oldIdx] = static_cast<double>(iv[oldIdx]);
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const signed char iv1[147]
// Return Type  : void
//
static __global__ __launch_bounds__(160, 1) void squeezenet_svm_predict_kernel3(
  const signed char iv1[147])
{
  unsigned int threadId;
  int oldIdx;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  oldIdx = static_cast<int>(threadId);
  if (oldIdx < 147) {
    gpu_v.classifier.CodingMatrix[oldIdx] = static_cast<double>(iv1[oldIdx]);
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const char cv[3]
// Return Type  : void
//
static __global__ __launch_bounds__(32, 1) void squeezenet_svm_predict_kernel4(
  const char cv[3])
{
  unsigned int threadId;
  int oldIdx;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  oldIdx = static_cast<int>(threadId);
  if (oldIdx < 3) {
    gpu_v.classifier.ScoreType[oldIdx] = cv[oldIdx];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                short aux1[1280]
// Return Type  : void
//
static __global__ __launch_bounds__(512, 1) void squeezenet_svm_predict_kernel5
  (short aux1[1280])
{
  unsigned int threadId;
  int oldIdx;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  oldIdx = static_cast<int>(threadId);
  if (oldIdx < 1280) {
    if (oldIdx + 1 <= 640) {
      aux1[oldIdx] = static_cast<short>(oldIdx + 1);
    } else {
      aux1[oldIdx] = static_cast<short>(1280 - oldIdx);
    }
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                short aux2[960]
// Return Type  : void
//
static __global__ __launch_bounds__(512, 1) void squeezenet_svm_predict_kernel6
  (short aux2[960])
{
  unsigned int threadId;
  int oldIdx;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  oldIdx = static_cast<int>(threadId);
  if (oldIdx < 960) {
    if (oldIdx + 1 <= 480) {
      aux2[oldIdx] = static_cast<short>(oldIdx + 1);
    } else {
      aux2[oldIdx] = static_cast<short>(960 - oldIdx);
    }
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const short aux1[1280]
//                double rowWeights[2724]
//                short ipRowIndices[2724]
// Return Type  : void
//
static __global__ __launch_bounds__(512, 1) void squeezenet_svm_predict_kernel7(
  const short aux1[1280], double rowWeights[2724], short ipRowIndices[2724])
{
  unsigned int threadId;
  int l;
  int rowIdx;
  int k;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  k = static_cast<int>(threadId % 12U);
  rowIdx = static_cast<int>((threadId - static_cast<unsigned int>(k)) / 12U);
  if (rowIdx < 227) {
    double absx2;
    double absx;
    int oldIdx;
    double sumVal;
    sumVal = (static_cast<double>(rowIdx) + 1.0) / 0.3546875 +
      -0.90969162995594721;
    oldIdx = static_cast<int>(floor(sumVal - 5.6387665198237888));
    absx = fabs(0.3546875 * (sumVal - (static_cast<double>(oldIdx + k) + 1.0)));
    absx2 = absx * absx;
    sumVal = rt_powd_snf(absx, 3.0);
    oldIdx = (oldIdx + k) + 1;
    if (oldIdx - 1 == 0) {
      l = 0;
    } else {
      l = static_cast<int>(fmod(static_cast<double>(oldIdx) - 1.0, 1280.0));
      if ((static_cast<int>(l != 0)) && (static_cast<int>(oldIdx - 1 < 0))) {
        l += 1280;
      }
    }

    ipRowIndices[rowIdx + 227 * k] = aux1[l];
    rowWeights[rowIdx + 227 * k] = 0.3546875 * (((1.5 * sumVal - 2.5 * absx2) +
      1.0) * static_cast<double>(absx <= 1.0) + (((-0.5 * sumVal + 2.5 * absx2)
      - 4.0 * absx) + 2.0) * static_cast<double>((static_cast<int>(1.0 < absx)) &&
      (static_cast<int>(absx <= 2.0))));
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const short aux2[960]
//                double colWeights[2043]
//                short ipColIndices[2043]
// Return Type  : void
//
static __global__ __launch_bounds__(512, 1) void squeezenet_svm_predict_kernel8(
  const short aux2[960], double colWeights[2043], short ipColIndices[2043])
{
  unsigned int threadId;
  int l;
  int colIdx;
  int k;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  k = static_cast<int>(threadId % 9U);
  colIdx = static_cast<int>((threadId - static_cast<unsigned int>(k)) / 9U);
  if (colIdx < 227) {
    double absx2;
    double absx;
    int oldIdx;
    double sumVal;
    sumVal = (static_cast<double>(colIdx) + 1.0) / 0.47291666666666665 +
      -0.55726872246696035;
    oldIdx = static_cast<int>(floor(sumVal - 4.2290748898678414));
    absx = fabs(0.47291666666666665 * (sumVal - (static_cast<double>(oldIdx + k)
      + 1.0)));
    absx2 = absx * absx;
    sumVal = rt_powd_snf(absx, 3.0);
    oldIdx = (oldIdx + k) + 1;
    if (oldIdx - 1 == 0) {
      l = 0;
    } else {
      l = static_cast<int>(fmod(static_cast<double>(oldIdx) - 1.0, 960.0));
      if ((static_cast<int>(l != 0)) && (static_cast<int>(oldIdx - 1 < 0))) {
        l += 960;
      }
    }

    ipColIndices[colIdx + 227 * k] = aux2[l];
    colWeights[colIdx + 227 * k] = 0.47291666666666665 * (((1.5 * sumVal - 2.5 *
      absx2) + 1.0) * static_cast<double>(absx <= 1.0) + (((-0.5 * sumVal + 2.5 *
      absx2) - 4.0 * absx) + 2.0) * static_cast<double>((static_cast<int>(1.0 <
      absx)) && (static_cast<int>(absx <= 2.0))));
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double rowWeights[2724]
//                double rowWeightsTotal[227]
// Return Type  : void
//
static __global__ __launch_bounds__(256, 1) void squeezenet_svm_predict_kernel9(
  const double rowWeights[2724], double rowWeightsTotal[227])
{
  unsigned int threadId;
  int oldIdx;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  oldIdx = static_cast<int>(threadId);
  if (oldIdx < 227) {
    rowWeightsTotal[oldIdx] = rowWeights[oldIdx];
  }
}

//
// Arguments    : const unsigned char in[921600]
// Return Type  : double
//
double squeezenet_svm_predict(const unsigned char in[921600])
{
  double out;
  static const signed char iv[49] = { 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
    1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
    0, 1, 1, 1, 1, 1, 1, 1, 0 };

  static const double dv[7] = { 0.019064124783362217, 0.090121317157712308,
    0.20537261698440207, 0.15251299826689774, 0.20103986135181975,
    0.23743500866551126, 0.094454072790294621 };

  static const signed char iv1[147] = { 1, -1, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0,
    0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, -1, 0, 1, 0, 0,
    0, 0, 0, -1, 0, 1, -1, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 1, 0, 0, -1, 0,
    0, 0, 1, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 1, -1, 0, 0, 0, 0, 0, 1,
    0, -1, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 1, -1, 0,
    0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0,
    0, 1, 0, -1, 0, 0, 0, 0, 0, 1, -1 };

  static const char cv[3] = { 'i', 'n', 'f' };

  int k;
  static unsigned char b_out[154587];
  float featuresTest[1000];
  float score[7];
  double (*gpu_dv)[7];
  signed char (*gpu_iv)[49];
  signed char (*gpu_iv1)[147];
  char (*gpu_cv)[3];
  short (*gpu_aux1)[1280];
  short (*gpu_aux2)[960];
  double (*gpu_rowWeights)[2724];
  short (*gpu_ipRowIndices)[2724];
  double (*gpu_colWeights)[2043];
  short (*gpu_ipColIndices)[2043];
  double (*gpu_rowWeightsTotal)[227];
  double (*gpu_colWeightsTotal)[227];
  unsigned char (*gpu_in)[921600];
  unsigned char (*gpu_partialResize)[326880];
  unsigned char (*gpu_out)[154587];
  double b[7];
  double c[7];
  double d[147];
  if (!isInitialized_squeezenet_svm_predict) {
    squeezenet_svm_predict_initialize();
  }

  cudaMalloc(&gpu_out, 154587UL);
  cudaMalloc(&gpu_partialResize, 326880UL);
  cudaMalloc(&gpu_ipColIndices, 4086UL);
  cudaMalloc(&gpu_in, 921600UL);
  cudaMalloc(&gpu_ipRowIndices, 5448UL);
  cudaMalloc(&gpu_colWeightsTotal, 1816UL);
  cudaMalloc(&gpu_colWeights, 16344UL);
  cudaMalloc(&gpu_rowWeightsTotal, 1816UL);
  cudaMalloc(&gpu_rowWeights, 21792UL);
  cudaMalloc(&gpu_aux2, 1920UL);
  cudaMalloc(&gpu_aux1, 2560UL);
  cudaMalloc(&gpu_cv, 3UL);
  cudaMalloc(&gpu_iv1, 147UL);
  cudaMalloc(&gpu_iv, 49UL);
  cudaMalloc(&gpu_dv, 56UL);

  // keep this function definition separate and not merge with the calling function  
  if (!net_not_empty) {
    DeepLearningNetwork_setup(&net);
    net_not_empty = true;
    cudaMemcpy(gpu_dv, (void *)&dv[0], 56UL, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(gpu_v, &v, 1720UL, 0UL, cudaMemcpyHostToDevice);
    squeezenet_svm_predict_kernel1<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>
      (*gpu_dv);
    cudaMemcpyFromSymbol(&v, gpu_v, 1720UL, 0UL, cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_iv, (void *)&iv[0], 49UL, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(gpu_v, &v, 1720UL, 0UL, cudaMemcpyHostToDevice);
    squeezenet_svm_predict_kernel2<<<dim3(1U, 1U, 1U), dim3(64U, 1U, 1U)>>>
      (*gpu_iv);
    cudaMemcpyFromSymbol(&v, gpu_v, 1720UL, 0UL, cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_iv1, (void *)&iv1[0], 147UL, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(gpu_v, &v, 1720UL, 0UL, cudaMemcpyHostToDevice);
    squeezenet_svm_predict_kernel3<<<dim3(1U, 1U, 1U), dim3(160U, 1U, 1U)>>>
      (*gpu_iv1);
    cudaMemcpyFromSymbol(&v, gpu_v, 1720UL, 0UL, cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_cv, (void *)&cv[0], 3UL, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(gpu_v, &v, 1720UL, 0UL, cudaMemcpyHostToDevice);
    squeezenet_svm_predict_kernel4<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>
      (*gpu_cv);
    cudaMemcpyFromSymbol(&v, gpu_v, 1720UL, 0UL, cudaMemcpyDeviceToHost);
  }

  squeezenet_svm_predict_kernel5<<<dim3(3U, 1U, 1U), dim3(512U, 1U, 1U)>>>
    (*gpu_aux1);
  squeezenet_svm_predict_kernel6<<<dim3(2U, 1U, 1U), dim3(512U, 1U, 1U)>>>
    (*gpu_aux2);
  squeezenet_svm_predict_kernel7<<<dim3(6U, 1U, 1U), dim3(512U, 1U, 1U)>>>
    (*gpu_aux1, *gpu_rowWeights, *gpu_ipRowIndices);
  squeezenet_svm_predict_kernel8<<<dim3(4U, 1U, 1U), dim3(512U, 1U, 1U)>>>
    (*gpu_aux2, *gpu_colWeights, *gpu_ipColIndices);
  squeezenet_svm_predict_kernel9<<<dim3(1U, 1U, 1U), dim3(256U, 1U, 1U)>>>
    (*gpu_rowWeights, *gpu_rowWeightsTotal);
  for (k = 0; k < 11; k++) {
    squeezenet_svm_predict_kernel10<<<dim3(1U, 1U, 1U), dim3(256U, 1U, 1U)>>>
      (*gpu_rowWeights, (k + 1) * 227, *gpu_rowWeightsTotal);
  }

  squeezenet_svm_predict_kernel11<<<dim3(1U, 1U, 1U), dim3(256U, 1U, 1U)>>>
    (*gpu_colWeights, *gpu_colWeightsTotal);
  for (k = 0; k < 8; k++) {
    squeezenet_svm_predict_kernel12<<<dim3(1U, 1U, 1U), dim3(256U, 1U, 1U)>>>
      (*gpu_colWeights, (k + 1) * 227, *gpu_colWeightsTotal);
  }

  cudaMemcpy(gpu_in, (void *)&in[0], 921600UL, cudaMemcpyHostToDevice);
  squeezenet_svm_predict_kernel13<<<dim3(639U, 1U, 1U), dim3(512U, 1U, 1U)>>>
    (*gpu_rowWeightsTotal, *gpu_rowWeights, *gpu_in, *gpu_ipRowIndices,
     *gpu_partialResize);
  squeezenet_svm_predict_kernel14<<<dim3(302U, 1U, 1U), dim3(512U, 1U, 1U)>>>
    (*gpu_colWeightsTotal, *gpu_colWeights, *gpu_partialResize,
     *gpu_ipColIndices, *gpu_out);
  cudaMemcpy(&b_out[0], gpu_out, 154587UL, cudaMemcpyDeviceToHost);
  DeepLearningNetwork_activations(&net, b_out, featuresTest);
  for (int e = 0; e < 7; e++) {
    b[e] = v.classifier.ClassNames[e];
  }

  for (int f = 0; f < 7; f++) {
    c[f] = v.classifier.Prior[f];
  }

  std::memcpy(&d[0], &v.classifier.CodingMatrix[0], 147U * sizeof(double));
  c_CompactClassificationECOC_pre(b, c, d, featuresTest, &out, score);

  //  score(out)
  cudaFree(*gpu_dv);
  cudaFree(*gpu_iv);
  cudaFree(*gpu_iv1);
  cudaFree(*gpu_cv);
  cudaFree(*gpu_aux1);
  cudaFree(*gpu_aux2);
  cudaFree(*gpu_rowWeights);
  cudaFree(*gpu_rowWeightsTotal);
  cudaFree(*gpu_colWeights);
  cudaFree(*gpu_colWeightsTotal);
  cudaFree(*gpu_ipRowIndices);
  cudaFree(*gpu_in);
  cudaFree(*gpu_ipColIndices);
  cudaFree(*gpu_partialResize);
  cudaFree(*gpu_out);
  return out;
}

//
// Arguments    : void
// Return Type  : void
//
void squeezenet_svm_predict_init()
{
  net_not_empty = false;
}

//
// File trailer for squeezenet_svm_predict.cu
//
// [EOF]
//
