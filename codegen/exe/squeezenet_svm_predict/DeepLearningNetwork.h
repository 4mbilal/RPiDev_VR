//
// File: DeepLearningNetwork.h
//
// GPU Coder version                    : 1.5
// CUDA/C/C++ source code generated on  : 06-Aug-2020 05:06:39
//
#ifndef DEEPLEARNINGNETWORK_H
#define DEEPLEARNINGNETWORK_H

// Include Files
#include <cstddef>
#include <cstdlib>
#include "rtwtypes.h"
#include "squeezenet_svm_predict_types.h"

// Type Definitions
#include "cnn_api.hpp"
#include "MWConvLayer.hpp"
#include "MWDepthConcatenationLayer.hpp"
#include "MWElementwiseAffineLayer.hpp"
#include "MWTargetNetworkImpl.hpp"

// Function Declarations
extern void DeepLearningNetwork_setup(b_squeezenet_0 *obj);

#endif

//
// File trailer for DeepLearningNetwork.h
//
// [EOF]
//
