//
// File: squeezenet_svm_predict_types.h
//
// GPU Coder version                    : 1.5
// CUDA/C/C++ source code generated on  : 06-Aug-2020 05:06:39
//
#ifndef SQUEEZENET_SVM_PREDICT_TYPES_H
#define SQUEEZENET_SVM_PREDICT_TYPES_H

// Include Files
#include "rtwtypes.h"

// Type Definitions
#include "cnn_api.hpp"
#include "MWTargetNetworkImpl.hpp"
#include "cublas_v2.h"

// Type Definitions
class b_squeezenet_0
{
 public:
  void allocate();
  void postsetup();
  b_squeezenet_0();
  void deallocate();
  void setup();
  void predict();
  void activations(int layerIdx);
  void cleanup();
  float *getLayerOutput(int layerIndex, int portIndex);
  float *getInputDataPointer();
  float *getOutputDataPointer();
  ~b_squeezenet_0();
  int batchSize;
  int numLayers;
  MWTensor *inputTensor;
  MWTensor *outputTensor;
  MWCNNLayer *layers[69];
  float *inputData;
  float *outputData;
 private:
  MWTargetNetworkImpl *targetImpl;
};

#endif

//
// File trailer for squeezenet_svm_predict_types.h
//
// [EOF]
//
