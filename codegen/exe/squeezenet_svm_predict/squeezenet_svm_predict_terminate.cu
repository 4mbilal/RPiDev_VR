//
// File: squeezenet_svm_predict_terminate.cu
//
// GPU Coder version                    : 1.5
// CUDA/C/C++ source code generated on  : 06-Aug-2020 05:06:39
//

// Include Files
#include "squeezenet_svm_predict_terminate.h"
#include "DeepLearningNetwork.h"
#include "rt_nonfinite.h"
#include "squeezenet_svm_predict.h"
#include "squeezenet_svm_predict_data.h"

// Function Declarations
static void cublasEnsureDestruction();

// Function Definitions

//
// Arguments    : void
// Return Type  : void
//
static void cublasEnsureDestruction()
{
  if (cublasGlobalHandle != NULL) {
    cublasDestroy(cublasGlobalHandle);
    cublasGlobalHandle = NULL;
  }
}

//
// Arguments    : void
// Return Type  : void
//
void squeezenet_svm_predict_terminate()
{
  cublasEnsureDestruction();
  isInitialized_squeezenet_svm_predict = false;
}

//
// File trailer for squeezenet_svm_predict_terminate.cu
//
// [EOF]
//
