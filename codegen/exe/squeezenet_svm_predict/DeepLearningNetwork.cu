//
// File: DeepLearningNetwork.cu
//
// GPU Coder version                    : 1.5
// CUDA/C/C++ source code generated on  : 06-Aug-2020 05:06:39
//

// Include Files
#include "DeepLearningNetwork.h"
#include "rt_nonfinite.h"
#include "squeezenet_svm_predict.h"
#include <cstdio>

// Type Definitions
#include "cnn_api.hpp"
#include "MWConvLayer.hpp"
#include "MWDepthConcatenationLayer.hpp"
#include "MWElementwiseAffineLayer.hpp"
#include "MWTargetNetworkImpl.hpp"

// Named Constants
const char * errorString =
  "Abnormal termination due to: %s.\nError in %s (line %d).";

// Function Declarations
static void checkCleanupCudaError(cudaError_t errCode, const char * file,
  unsigned int line);

// Function Definitions

//
// Arguments    : void
// Return Type  : void
//

//
// Arguments    : int MaxBufSize
//                int numBufstoAllocate
// Return Type  : void
//

//
// Arguments    : void
// Return Type  : void
//

//
// Arguments    : void
// Return Type  : void
//

//
// Arguments    : MWTargetNetworkImpl *b_targetImpl
//                MWTensor *b
//                int PoolSizeH
//                int PoolSizeW
//                int StrideH
//                int StrideW
//                int PaddingH_T
//                int PaddingH_B
//                int PaddingW_L
//                int PaddingW_R
//                int c
// Return Type  : void
//

//
// Arguments    : MWTargetNetworkImpl *b_targetImpl
//                MWTensor *b
//                int FilterSizeH
//                int FilterSizeW
//                int NumChannels
//                int NumFilters
//                int StrideH
//                int StrideW
//                int PaddingH_Top
//                int PaddingH_Bottom
//                int PaddingW_Left
//                int PaddingW_Right
//                int DilationFactorH
//                int DilationFactorW
//                int NumGroups
//                const char * c_a___codegen_exe_squeezenet_sv
//                const char * d_a___codegen_exe_squeezenet_sv
//                int c
// Return Type  : void
//

//
// Arguments    : MWTargetNetworkImpl *b_targetImpl
//                MWTensor *b
//                int ScaleHeight
//                int ScaleWidth
//                int ScaleChannels
//                int OffsetHeight
//                int OffsetWidth
//                int OffsetChannels
//                bool IsClippedAffine
//                int LowerBound
//                int UpperBound
//                const char * c_a___codegen_exe_squeezenet_sv
//                const char * d_a___codegen_exe_squeezenet_sv
//                int c
// Return Type  : void
//

//
// Arguments    : MWTargetNetworkImpl *b_targetImpl
//                MWTensor *m_in
//                int height
//                int width
//                int channels
//                int withAvg
//                const char * b
//                int c
// Return Type  : void
//

//
// Arguments    : MWTargetNetworkImpl *b_targetImpl
//                MWTensor *b
//                int c
// Return Type  : void
//

//
// Arguments    : MWTargetNetworkImpl *b_targetImpl
//                MWTensor *b
//                int c
// Return Type  : void
//

//
// Arguments    : MWTargetNetworkImpl *b_targetImpl
//                MWTensor *b
//                int inPlaceOp
//                int c
// Return Type  : void
//

//
// Arguments    : MWTargetNetworkImpl *b_targetImpl
//                MWTensor *b
//                int c
// Return Type  : void
//

//
// Arguments    : void
// Return Type  : void
//

//
// Arguments    : void
// Return Type  : void
//

//
// Arguments    : int b
// Return Type  : void
//

//
// Arguments    : void
// Return Type  : float *
//

//
// Arguments    : int handle
// Return Type  : void
//

//
// Arguments    : int b_index
// Return Type  : MWTensor *
//

//
// Arguments    : MWCNNLayer *b_layers[69]
//                int b_numLayers
//                int outputLayerIndices[1]
//                int outputPortIndices[1]
//                int numOutputs
// Return Type  : void
//

//
// Arguments    : void
// Return Type  : void
//

//
// Arguments    : int b_batchSize
// Return Type  : void
//

//
// Arguments    : int b_batchSize
// Return Type  : void
//

//
// Arguments    : int channels
// Return Type  : void
//

//
// Arguments    : float *data
// Return Type  : void
//

//
// Arguments    : int height
// Return Type  : void
//

//
// Arguments    : bool isSequenceNetwork
// Return Type  : void
//

//
// Arguments    : const char * name
// Return Type  : void
//

//
// Arguments    : int sequenceLength
// Return Type  : void
//

//
// Arguments    : int width
// Return Type  : void
//

//
// Arguments    : cudaError_t errCode
//                const char * file
//                unsigned int line
// Return Type  : void
//
static void checkCleanupCudaError(cudaError_t errCode, const char * file,
  unsigned int line)
{
  if ((errCode != cudaSuccess) && (errCode != cudaErrorCudartUnloading)) {
    printf(errorString, cudaGetErrorString(errCode), file, line);
  }
}

//
// Arguments    : int layerIdx
// Return Type  : void
//
void b_squeezenet_0::activations(int)
{
  this->targetImpl->doInference(this->batchSize);
}

//
// Arguments    : void
// Return Type  : void
//
void b_squeezenet_0::allocate()
{
  this->targetImpl->allocate(0, 0);
}

//
// Arguments    : void
// Return Type  : void
//
void b_squeezenet_0::cleanup()
{
  this->deallocate();
  for (int idx = 0; idx < 69; idx++) {
    this->layers[idx]->cleanup();
  }

  if (this->targetImpl) {
    this->targetImpl->cleanup();
  }
}

//
// Arguments    : void
// Return Type  : void
//
void b_squeezenet_0::deallocate()
{
  this->targetImpl->deallocate();
  for (int idx = 0; idx < 69; idx++) {
    this->layers[idx]->deallocate();
  }
}

//
// Arguments    : void
// Return Type  : float *
//
float *b_squeezenet_0::getInputDataPointer()
{
  return this->inputTensor->getFloatData();
}

//
// Arguments    : int layerIndex
//                int portIndex
// Return Type  : float *
//
float *b_squeezenet_0::getLayerOutput(int layerIndex, int portIndex)
{
  return this->layers[layerIndex]->getLayerOutput(portIndex);
}

//
// Arguments    : void
// Return Type  : float *
//
float *b_squeezenet_0::getOutputDataPointer()
{
  return this->outputTensor->getFloatData();
}

//
// Arguments    : void
// Return Type  : void
//
void b_squeezenet_0::postsetup()
{
  int activationsIdx[1];
  int activationPortIdx[1];
  activationsIdx[0] = 66;
  activationPortIdx[0] = 0;
  this->targetImpl->postSetup(this->layers, this->numLayers, activationsIdx,
    activationPortIdx, 1);
  for (int idx = 0; idx < 69; idx++) {
    this->layers[idx]->allocate();
  }
}

//
// Arguments    : void
// Return Type  : void
//
void b_squeezenet_0::predict()
{
  this->targetImpl->doInference(this->batchSize);
}

//
// Arguments    : void
// Return Type  : void
//
void b_squeezenet_0::setup()
{
  this->targetImpl->preSetup();
  this->allocate();
  (dynamic_cast<MWInputLayer *>(this->layers[0]))->createInputLayer
    (this->targetImpl, this->inputTensor, 227, 227, 3, 0, "", -1);
  (dynamic_cast<MWElementwiseAffineLayer *>(this->layers[1]))
    ->createElementwiseAffineLayer(this->targetImpl, this->layers[0]
    ->getOutputTensor(0), 1, 1, 3, 1, 1, 3, false, 1, 1,
    "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_data_scale.bin",
    "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_data_offset.bin", -1);
  (dynamic_cast<MWConvLayer *>(this->layers[2]))->createConvLayer
    (this->targetImpl, this->layers[1]->getOutputTensor(0), 3, 3, 3, 64, 2, 2, 0,
     0, 0, 0, 1, 1, 1,
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_conv1_w.bin",
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_conv1_b.bin", -1);
  (dynamic_cast<MWReLULayer *>(this->layers[3]))->createReLULayer
    (this->targetImpl, this->layers[2]->getOutputTensor(0), 1, -1);
  (dynamic_cast<MWMaxPoolingLayer *>(this->layers[4]))->createMaxPoolingLayer
    (this->targetImpl, this->layers[3]->getOutputTensor(0), 3, 3, 2, 2, 0, 0, 0,
     0, 0, 1, -1);
  (dynamic_cast<MWConvLayer *>(this->layers[5]))->createConvLayer
    (this->targetImpl, this->layers[4]->getOutputTensor(0), 1, 1, 64, 16, 1, 1,
     0, 0, 0, 0, 1, 1, 1,
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire2-squeeze1x1_w.bin",
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire2-squeeze1x1_b.bin",
     -1);
  (dynamic_cast<MWReLULayer *>(this->layers[6]))->createReLULayer
    (this->targetImpl, this->layers[5]->getOutputTensor(0), 1, -1);
  (dynamic_cast<MWConvLayer *>(this->layers[7]))->createConvLayer
    (this->targetImpl, this->layers[6]->getOutputTensor(0), 1, 1, 16, 64, 1, 1,
     0, 0, 0, 0, 1, 1, 1,
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire2-expand1x1_w.bin",
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire2-expand1x1_b.bin",
     -1);
  (dynamic_cast<MWReLULayer *>(this->layers[8]))->createReLULayer
    (this->targetImpl, this->layers[7]->getOutputTensor(0), 1, -1);
  (dynamic_cast<MWConvLayer *>(this->layers[9]))->createConvLayer
    (this->targetImpl, this->layers[6]->getOutputTensor(0), 3, 3, 16, 64, 1, 1,
     1, 1, 1, 1, 1, 1, 1,
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire2-expand3x3_w.bin",
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire2-expand3x3_b.bin",
     -1);
  (dynamic_cast<MWReLULayer *>(this->layers[10]))->createReLULayer
    (this->targetImpl, this->layers[9]->getOutputTensor(0), 1, -1);
  (dynamic_cast<MWDepthConcatenationLayer *>(this->layers[11]))
    ->createDepthConcatenationLayer(this->targetImpl, 2, this->layers[8]
    ->getOutputTensor(0), this->layers[10]->getOutputTensor(0), -1);
  (dynamic_cast<MWConvLayer *>(this->layers[12]))->createConvLayer
    (this->targetImpl, this->layers[11]->getOutputTensor(0), 1, 1, 128, 16, 1, 1,
     0, 0, 0, 0, 1, 1, 1,
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire3-squeeze1x1_w.bin",
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire3-squeeze1x1_b.bin",
     -1);
  (dynamic_cast<MWReLULayer *>(this->layers[13]))->createReLULayer
    (this->targetImpl, this->layers[12]->getOutputTensor(0), 1, -1);
  (dynamic_cast<MWConvLayer *>(this->layers[14]))->createConvLayer
    (this->targetImpl, this->layers[13]->getOutputTensor(0), 1, 1, 16, 64, 1, 1,
     0, 0, 0, 0, 1, 1, 1,
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire3-expand1x1_w.bin",
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire3-expand1x1_b.bin",
     -1);
  (dynamic_cast<MWReLULayer *>(this->layers[15]))->createReLULayer
    (this->targetImpl, this->layers[14]->getOutputTensor(0), 1, -1);
  (dynamic_cast<MWConvLayer *>(this->layers[16]))->createConvLayer
    (this->targetImpl, this->layers[13]->getOutputTensor(0), 3, 3, 16, 64, 1, 1,
     1, 1, 1, 1, 1, 1, 1,
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire3-expand3x3_w.bin",
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire3-expand3x3_b.bin",
     -1);
  (dynamic_cast<MWReLULayer *>(this->layers[17]))->createReLULayer
    (this->targetImpl, this->layers[16]->getOutputTensor(0), 1, -1);
  (dynamic_cast<MWDepthConcatenationLayer *>(this->layers[18]))
    ->createDepthConcatenationLayer(this->targetImpl, 2, this->layers[15]
    ->getOutputTensor(0), this->layers[17]->getOutputTensor(0), -1);
  (dynamic_cast<MWMaxPoolingLayer *>(this->layers[19]))->createMaxPoolingLayer
    (this->targetImpl, this->layers[18]->getOutputTensor(0), 3, 3, 2, 2, 0, 1, 0,
     1, 0, 1, -1);
  (dynamic_cast<MWConvLayer *>(this->layers[20]))->createConvLayer
    (this->targetImpl, this->layers[19]->getOutputTensor(0), 1, 1, 128, 32, 1, 1,
     0, 0, 0, 0, 1, 1, 1,
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire4-squeeze1x1_w.bin",
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire4-squeeze1x1_b.bin",
     -1);
  (dynamic_cast<MWReLULayer *>(this->layers[21]))->createReLULayer
    (this->targetImpl, this->layers[20]->getOutputTensor(0), 1, -1);
  (dynamic_cast<MWConvLayer *>(this->layers[22]))->createConvLayer
    (this->targetImpl, this->layers[21]->getOutputTensor(0), 1, 1, 32, 128, 1, 1,
     0, 0, 0, 0, 1, 1, 1,
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire4-expand1x1_w.bin",
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire4-expand1x1_b.bin",
     -1);
  (dynamic_cast<MWReLULayer *>(this->layers[23]))->createReLULayer
    (this->targetImpl, this->layers[22]->getOutputTensor(0), 1, -1);
  (dynamic_cast<MWConvLayer *>(this->layers[24]))->createConvLayer
    (this->targetImpl, this->layers[21]->getOutputTensor(0), 3, 3, 32, 128, 1, 1,
     1, 1, 1, 1, 1, 1, 1,
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire4-expand3x3_w.bin",
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire4-expand3x3_b.bin",
     -1);
  (dynamic_cast<MWReLULayer *>(this->layers[25]))->createReLULayer
    (this->targetImpl, this->layers[24]->getOutputTensor(0), 1, -1);
  (dynamic_cast<MWDepthConcatenationLayer *>(this->layers[26]))
    ->createDepthConcatenationLayer(this->targetImpl, 2, this->layers[23]
    ->getOutputTensor(0), this->layers[25]->getOutputTensor(0), -1);
  (dynamic_cast<MWConvLayer *>(this->layers[27]))->createConvLayer
    (this->targetImpl, this->layers[26]->getOutputTensor(0), 1, 1, 256, 32, 1, 1,
     0, 0, 0, 0, 1, 1, 1,
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire5-squeeze1x1_w.bin",
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire5-squeeze1x1_b.bin",
     -1);
  (dynamic_cast<MWReLULayer *>(this->layers[28]))->createReLULayer
    (this->targetImpl, this->layers[27]->getOutputTensor(0), 1, -1);
  (dynamic_cast<MWConvLayer *>(this->layers[29]))->createConvLayer
    (this->targetImpl, this->layers[28]->getOutputTensor(0), 1, 1, 32, 128, 1, 1,
     0, 0, 0, 0, 1, 1, 1,
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire5-expand1x1_w.bin",
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire5-expand1x1_b.bin",
     -1);
  (dynamic_cast<MWReLULayer *>(this->layers[30]))->createReLULayer
    (this->targetImpl, this->layers[29]->getOutputTensor(0), 1, -1);
  (dynamic_cast<MWConvLayer *>(this->layers[31]))->createConvLayer
    (this->targetImpl, this->layers[28]->getOutputTensor(0), 3, 3, 32, 128, 1, 1,
     1, 1, 1, 1, 1, 1, 1,
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire5-expand3x3_w.bin",
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire5-expand3x3_b.bin",
     -1);
  (dynamic_cast<MWReLULayer *>(this->layers[32]))->createReLULayer
    (this->targetImpl, this->layers[31]->getOutputTensor(0), 1, -1);
  (dynamic_cast<MWDepthConcatenationLayer *>(this->layers[33]))
    ->createDepthConcatenationLayer(this->targetImpl, 2, this->layers[30]
    ->getOutputTensor(0), this->layers[32]->getOutputTensor(0), -1);
  (dynamic_cast<MWMaxPoolingLayer *>(this->layers[34]))->createMaxPoolingLayer
    (this->targetImpl, this->layers[33]->getOutputTensor(0), 3, 3, 2, 2, 0, 1, 0,
     1, 0, 1, -1);
  (dynamic_cast<MWConvLayer *>(this->layers[35]))->createConvLayer
    (this->targetImpl, this->layers[34]->getOutputTensor(0), 1, 1, 256, 48, 1, 1,
     0, 0, 0, 0, 1, 1, 1,
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire6-squeeze1x1_w.bin",
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire6-squeeze1x1_b.bin",
     -1);
  (dynamic_cast<MWReLULayer *>(this->layers[36]))->createReLULayer
    (this->targetImpl, this->layers[35]->getOutputTensor(0), 1, -1);
  (dynamic_cast<MWConvLayer *>(this->layers[37]))->createConvLayer
    (this->targetImpl, this->layers[36]->getOutputTensor(0), 1, 1, 48, 192, 1, 1,
     0, 0, 0, 0, 1, 1, 1,
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire6-expand1x1_w.bin",
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire6-expand1x1_b.bin",
     -1);
  (dynamic_cast<MWReLULayer *>(this->layers[38]))->createReLULayer
    (this->targetImpl, this->layers[37]->getOutputTensor(0), 1, -1);
  (dynamic_cast<MWConvLayer *>(this->layers[39]))->createConvLayer
    (this->targetImpl, this->layers[36]->getOutputTensor(0), 3, 3, 48, 192, 1, 1,
     1, 1, 1, 1, 1, 1, 1,
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire6-expand3x3_w.bin",
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire6-expand3x3_b.bin",
     -1);
  (dynamic_cast<MWReLULayer *>(this->layers[40]))->createReLULayer
    (this->targetImpl, this->layers[39]->getOutputTensor(0), 1, -1);
  (dynamic_cast<MWDepthConcatenationLayer *>(this->layers[41]))
    ->createDepthConcatenationLayer(this->targetImpl, 2, this->layers[38]
    ->getOutputTensor(0), this->layers[40]->getOutputTensor(0), -1);
  (dynamic_cast<MWConvLayer *>(this->layers[42]))->createConvLayer
    (this->targetImpl, this->layers[41]->getOutputTensor(0), 1, 1, 384, 48, 1, 1,
     0, 0, 0, 0, 1, 1, 1,
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire7-squeeze1x1_w.bin",
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire7-squeeze1x1_b.bin",
     -1);
  (dynamic_cast<MWReLULayer *>(this->layers[43]))->createReLULayer
    (this->targetImpl, this->layers[42]->getOutputTensor(0), 1, -1);
  (dynamic_cast<MWConvLayer *>(this->layers[44]))->createConvLayer
    (this->targetImpl, this->layers[43]->getOutputTensor(0), 1, 1, 48, 192, 1, 1,
     0, 0, 0, 0, 1, 1, 1,
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire7-expand1x1_w.bin",
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire7-expand1x1_b.bin",
     -1);
  (dynamic_cast<MWReLULayer *>(this->layers[45]))->createReLULayer
    (this->targetImpl, this->layers[44]->getOutputTensor(0), 1, -1);
  (dynamic_cast<MWConvLayer *>(this->layers[46]))->createConvLayer
    (this->targetImpl, this->layers[43]->getOutputTensor(0), 3, 3, 48, 192, 1, 1,
     1, 1, 1, 1, 1, 1, 1,
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire7-expand3x3_w.bin",
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire7-expand3x3_b.bin",
     -1);
  (dynamic_cast<MWReLULayer *>(this->layers[47]))->createReLULayer
    (this->targetImpl, this->layers[46]->getOutputTensor(0), 1, -1);
  (dynamic_cast<MWDepthConcatenationLayer *>(this->layers[48]))
    ->createDepthConcatenationLayer(this->targetImpl, 2, this->layers[45]
    ->getOutputTensor(0), this->layers[47]->getOutputTensor(0), -1);
  (dynamic_cast<MWConvLayer *>(this->layers[49]))->createConvLayer
    (this->targetImpl, this->layers[48]->getOutputTensor(0), 1, 1, 384, 64, 1, 1,
     0, 0, 0, 0, 1, 1, 1,
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire8-squeeze1x1_w.bin",
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire8-squeeze1x1_b.bin",
     -1);
  (dynamic_cast<MWReLULayer *>(this->layers[50]))->createReLULayer
    (this->targetImpl, this->layers[49]->getOutputTensor(0), 1, -1);
  (dynamic_cast<MWConvLayer *>(this->layers[51]))->createConvLayer
    (this->targetImpl, this->layers[50]->getOutputTensor(0), 1, 1, 64, 256, 1, 1,
     0, 0, 0, 0, 1, 1, 1,
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire8-expand1x1_w.bin",
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire8-expand1x1_b.bin",
     -1);
  (dynamic_cast<MWReLULayer *>(this->layers[52]))->createReLULayer
    (this->targetImpl, this->layers[51]->getOutputTensor(0), 1, -1);
  (dynamic_cast<MWConvLayer *>(this->layers[53]))->createConvLayer
    (this->targetImpl, this->layers[50]->getOutputTensor(0), 3, 3, 64, 256, 1, 1,
     1, 1, 1, 1, 1, 1, 1,
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire8-expand3x3_w.bin",
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire8-expand3x3_b.bin",
     -1);
  (dynamic_cast<MWReLULayer *>(this->layers[54]))->createReLULayer
    (this->targetImpl, this->layers[53]->getOutputTensor(0), 1, -1);
  (dynamic_cast<MWDepthConcatenationLayer *>(this->layers[55]))
    ->createDepthConcatenationLayer(this->targetImpl, 2, this->layers[52]
    ->getOutputTensor(0), this->layers[54]->getOutputTensor(0), -1);
  (dynamic_cast<MWConvLayer *>(this->layers[56]))->createConvLayer
    (this->targetImpl, this->layers[55]->getOutputTensor(0), 1, 1, 512, 64, 1, 1,
     0, 0, 0, 0, 1, 1, 1,
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire9-squeeze1x1_w.bin",
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire9-squeeze1x1_b.bin",
     -1);
  (dynamic_cast<MWReLULayer *>(this->layers[57]))->createReLULayer
    (this->targetImpl, this->layers[56]->getOutputTensor(0), 1, -1);
  (dynamic_cast<MWConvLayer *>(this->layers[58]))->createConvLayer
    (this->targetImpl, this->layers[57]->getOutputTensor(0), 1, 1, 64, 256, 1, 1,
     0, 0, 0, 0, 1, 1, 1,
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire9-expand1x1_w.bin",
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire9-expand1x1_b.bin",
     -1);
  (dynamic_cast<MWReLULayer *>(this->layers[59]))->createReLULayer
    (this->targetImpl, this->layers[58]->getOutputTensor(0), 1, -1);
  (dynamic_cast<MWConvLayer *>(this->layers[60]))->createConvLayer
    (this->targetImpl, this->layers[57]->getOutputTensor(0), 3, 3, 64, 256, 1, 1,
     1, 1, 1, 1, 1, 1, 1,
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire9-expand3x3_w.bin",
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_fire9-expand3x3_b.bin",
     -1);
  (dynamic_cast<MWReLULayer *>(this->layers[61]))->createReLULayer
    (this->targetImpl, this->layers[60]->getOutputTensor(0), 1, -1);
  (dynamic_cast<MWDepthConcatenationLayer *>(this->layers[62]))
    ->createDepthConcatenationLayer(this->targetImpl, 2, this->layers[59]
    ->getOutputTensor(0), this->layers[61]->getOutputTensor(0), -1);
  (dynamic_cast<MWPassthroughLayer *>(this->layers[63]))->createPassthroughLayer
    (this->targetImpl, this->layers[62]->getOutputTensor(0), -1);
  (dynamic_cast<MWConvLayer *>(this->layers[64]))->createConvLayer
    (this->targetImpl, this->layers[63]->getOutputTensor(0), 1, 1, 512, 1000, 1,
     1, 0, 0, 0, 0, 1, 1, 1,
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_conv10_w.bin",
     "./codegen/exe/squeezenet_svm_predict/cnn_squeezenet_conv10_b.bin", -1);
  (dynamic_cast<MWReLULayer *>(this->layers[65]))->createReLULayer
    (this->targetImpl, this->layers[64]->getOutputTensor(0), 1, -1);
  (dynamic_cast<MWAvgPoolingLayer *>(this->layers[66]))->createAvgPoolingLayer
    (this->targetImpl, this->layers[65]->getOutputTensor(0), -1, -1, 1, 1, 0, 0,
     0, 0, -1);
  (dynamic_cast<MWSoftmaxLayer *>(this->layers[67]))->createSoftmaxLayer
    (this->targetImpl, this->layers[66]->getOutputTensor(0), -1);
  (dynamic_cast<MWOutputLayer *>(this->layers[68]))->createOutputLayer
    (this->targetImpl, this->layers[67]->getOutputTensor(0), -1);
  this->postsetup();
  this->inputTensor->setData(this->layers[0]->getLayerOutput(0));
  this->outputTensor->setData(this->layers[68]->getLayerOutput(0));
  this->inputData = this->inputTensor->getFloatData();
  this->outputData = this->outputTensor->getFloatData();
}

//
// Arguments    : void
// Return Type  : void
//
b_squeezenet_0::~b_squeezenet_0()
{
  this->cleanup();
  checkCleanupCudaError(cudaGetLastError(), __FILE__, __LINE__);
  for (int idx = 0; idx < 69; idx++) {
    delete this->layers[idx];
  }

  if (this->targetImpl) {
    delete this->targetImpl;
  }

  delete this->inputTensor;
  delete this->outputTensor;
}

//
// Arguments    : void
// Return Type  : void
//
b_squeezenet_0::b_squeezenet_0()
{
  this->numLayers = 69;
  this->targetImpl = 0;
  this->layers[0] = new MWInputLayer;
  this->layers[0]->setName("data");
  this->layers[1] = new MWElementwiseAffineLayer;
  this->layers[1]->setName("data_normalization");
  this->layers[2] = new MWConvLayer;
  this->layers[2]->setName("conv1");
  this->layers[3] = new MWReLULayer;
  this->layers[3]->setName("relu_conv1");
  this->layers[4] = new MWMaxPoolingLayer;
  this->layers[4]->setName("pool1");
  this->layers[5] = new MWConvLayer;
  this->layers[5]->setName("fire2-squeeze1x1");
  this->layers[6] = new MWReLULayer;
  this->layers[6]->setName("fire2-relu_squeeze1x1");
  this->layers[7] = new MWConvLayer;
  this->layers[7]->setName("fire2-expand1x1");
  this->layers[8] = new MWReLULayer;
  this->layers[8]->setName("fire2-relu_expand1x1");
  this->layers[9] = new MWConvLayer;
  this->layers[9]->setName("fire2-expand3x3");
  this->layers[10] = new MWReLULayer;
  this->layers[10]->setName("fire2-relu_expand3x3");
  this->layers[11] = new MWDepthConcatenationLayer;
  this->layers[11]->setName("fire2-concat");
  this->layers[12] = new MWConvLayer;
  this->layers[12]->setName("fire3-squeeze1x1");
  this->layers[13] = new MWReLULayer;
  this->layers[13]->setName("fire3-relu_squeeze1x1");
  this->layers[14] = new MWConvLayer;
  this->layers[14]->setName("fire3-expand1x1");
  this->layers[15] = new MWReLULayer;
  this->layers[15]->setName("fire3-relu_expand1x1");
  this->layers[16] = new MWConvLayer;
  this->layers[16]->setName("fire3-expand3x3");
  this->layers[17] = new MWReLULayer;
  this->layers[17]->setName("fire3-relu_expand3x3");
  this->layers[18] = new MWDepthConcatenationLayer;
  this->layers[18]->setName("fire3-concat");
  this->layers[19] = new MWMaxPoolingLayer;
  this->layers[19]->setName("pool3");
  this->layers[20] = new MWConvLayer;
  this->layers[20]->setName("fire4-squeeze1x1");
  this->layers[21] = new MWReLULayer;
  this->layers[21]->setName("fire4-relu_squeeze1x1");
  this->layers[22] = new MWConvLayer;
  this->layers[22]->setName("fire4-expand1x1");
  this->layers[23] = new MWReLULayer;
  this->layers[23]->setName("fire4-relu_expand1x1");
  this->layers[24] = new MWConvLayer;
  this->layers[24]->setName("fire4-expand3x3");
  this->layers[25] = new MWReLULayer;
  this->layers[25]->setName("fire4-relu_expand3x3");
  this->layers[26] = new MWDepthConcatenationLayer;
  this->layers[26]->setName("fire4-concat");
  this->layers[27] = new MWConvLayer;
  this->layers[27]->setName("fire5-squeeze1x1");
  this->layers[28] = new MWReLULayer;
  this->layers[28]->setName("fire5-relu_squeeze1x1");
  this->layers[29] = new MWConvLayer;
  this->layers[29]->setName("fire5-expand1x1");
  this->layers[30] = new MWReLULayer;
  this->layers[30]->setName("fire5-relu_expand1x1");
  this->layers[31] = new MWConvLayer;
  this->layers[31]->setName("fire5-expand3x3");
  this->layers[32] = new MWReLULayer;
  this->layers[32]->setName("fire5-relu_expand3x3");
  this->layers[33] = new MWDepthConcatenationLayer;
  this->layers[33]->setName("fire5-concat");
  this->layers[34] = new MWMaxPoolingLayer;
  this->layers[34]->setName("pool5");
  this->layers[35] = new MWConvLayer;
  this->layers[35]->setName("fire6-squeeze1x1");
  this->layers[36] = new MWReLULayer;
  this->layers[36]->setName("fire6-relu_squeeze1x1");
  this->layers[37] = new MWConvLayer;
  this->layers[37]->setName("fire6-expand1x1");
  this->layers[38] = new MWReLULayer;
  this->layers[38]->setName("fire6-relu_expand1x1");
  this->layers[39] = new MWConvLayer;
  this->layers[39]->setName("fire6-expand3x3");
  this->layers[40] = new MWReLULayer;
  this->layers[40]->setName("fire6-relu_expand3x3");
  this->layers[41] = new MWDepthConcatenationLayer;
  this->layers[41]->setName("fire6-concat");
  this->layers[42] = new MWConvLayer;
  this->layers[42]->setName("fire7-squeeze1x1");
  this->layers[43] = new MWReLULayer;
  this->layers[43]->setName("fire7-relu_squeeze1x1");
  this->layers[44] = new MWConvLayer;
  this->layers[44]->setName("fire7-expand1x1");
  this->layers[45] = new MWReLULayer;
  this->layers[45]->setName("fire7-relu_expand1x1");
  this->layers[46] = new MWConvLayer;
  this->layers[46]->setName("fire7-expand3x3");
  this->layers[47] = new MWReLULayer;
  this->layers[47]->setName("fire7-relu_expand3x3");
  this->layers[48] = new MWDepthConcatenationLayer;
  this->layers[48]->setName("fire7-concat");
  this->layers[49] = new MWConvLayer;
  this->layers[49]->setName("fire8-squeeze1x1");
  this->layers[50] = new MWReLULayer;
  this->layers[50]->setName("fire8-relu_squeeze1x1");
  this->layers[51] = new MWConvLayer;
  this->layers[51]->setName("fire8-expand1x1");
  this->layers[52] = new MWReLULayer;
  this->layers[52]->setName("fire8-relu_expand1x1");
  this->layers[53] = new MWConvLayer;
  this->layers[53]->setName("fire8-expand3x3");
  this->layers[54] = new MWReLULayer;
  this->layers[54]->setName("fire8-relu_expand3x3");
  this->layers[55] = new MWDepthConcatenationLayer;
  this->layers[55]->setName("fire8-concat");
  this->layers[56] = new MWConvLayer;
  this->layers[56]->setName("fire9-squeeze1x1");
  this->layers[57] = new MWReLULayer;
  this->layers[57]->setName("fire9-relu_squeeze1x1");
  this->layers[58] = new MWConvLayer;
  this->layers[58]->setName("fire9-expand1x1");
  this->layers[59] = new MWReLULayer;
  this->layers[59]->setName("fire9-relu_expand1x1");
  this->layers[60] = new MWConvLayer;
  this->layers[60]->setName("fire9-expand3x3");
  this->layers[61] = new MWReLULayer;
  this->layers[61]->setName("fire9-relu_expand3x3");
  this->layers[62] = new MWDepthConcatenationLayer;
  this->layers[62]->setName("fire9-concat");
  this->layers[63] = new MWPassthroughLayer;
  this->layers[63]->setName("drop9");
  this->layers[64] = new MWConvLayer;
  this->layers[64]->setName("conv10");
  this->layers[65] = new MWReLULayer;
  this->layers[65]->setName("relu_conv10");
  this->layers[66] = new MWAvgPoolingLayer;
  this->layers[66]->setName("pool10");
  this->layers[67] = new MWSoftmaxLayer;
  this->layers[67]->setName("prob");
  this->layers[68] = new MWOutputLayer;
  this->layers[68]->setName("ClassificationLayer_predictions");
  this->targetImpl = new MWTargetNetworkImpl;
  this->targetImpl->setBatchSize(1);
  this->targetImpl->setIsSequenceNetwork(false);
  this->inputTensor = new MWTensor;
  this->inputTensor->setHeight(227);
  this->inputTensor->setWidth(227);
  this->inputTensor->setChannels(3);
  this->inputTensor->setBatchSize(1);
  this->inputTensor->setSequenceLength(1);
  this->outputTensor = new MWTensor;
}

//
// Arguments    : b_squeezenet_0 *obj
// Return Type  : void
//
void DeepLearningNetwork_setup(b_squeezenet_0 *obj)
{
  obj->setup();
  obj->batchSize = 1;
}

//
// File trailer for DeepLearningNetwork.cu
//
// [EOF]
//
