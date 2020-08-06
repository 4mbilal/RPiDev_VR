/*
 * File: _coder_squeezenet_svm_predict_mex.h
 *
 * GPU Coder version                    : 1.5
 * CUDA/C/C++ source code generated on  : 06-Aug-2020 05:06:39
 */

#ifndef _CODER_SQUEEZENET_SVM_PREDICT_MEX_H
#define _CODER_SQUEEZENET_SVM_PREDICT_MEX_H

/* Include Files */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "tmwtypes.h"
#include "mex.h"
#include "emlrt.h"
#include "_coder_squeezenet_svm_predict_api.h"

/* Function Declarations */
MEXFUNCTION_LINKAGE void mexFunction(int32_T nlhs, mxArray *plhs[], int32_T nrhs,
  const mxArray *prhs[]);
extern emlrtCTX mexFunctionCreateRootTLS(void);

#endif

/*
 * File trailer for _coder_squeezenet_svm_predict_mex.h
 *
 * [EOF]
 */
