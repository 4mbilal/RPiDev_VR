//
// File: squeezenet_svm_predict_initialize.cu
//
// GPU Coder version                    : 1.5
// CUDA/C/C++ source code generated on  : 06-Aug-2020 05:06:39
//

// Include Files
#include "squeezenet_svm_predict_initialize.h"
#include "DeepLearningNetwork.h"
#include "rt_nonfinite.h"
#include "squeezenet_svm_predict.h"
#include "squeezenet_svm_predict_data.h"

// Function Declarations
static void cublasEnsureInitialization();

// Function Definitions

//
// Arguments    : void
// Return Type  : void
//
static void cublasEnsureInitialization()
{
  if (cublasGlobalHandle == NULL) {
    cublasCreate(&cublasGlobalHandle);
    cublasSetPointerMode(cublasGlobalHandle, CUBLAS_POINTER_MODE_DEVICE);
  }
}

//
// Arguments    : void
// Return Type  : void
//
void squeezenet_svm_predict_initialize()
{
  rt_InitInfAndNaN();
  squeezenet_svm_predict_init();
  cublasEnsureInitialization();
  isInitialized_squeezenet_svm_predict = true;
}

//
// File trailer for squeezenet_svm_predict_initialize.cu
//
// [EOF]
//
