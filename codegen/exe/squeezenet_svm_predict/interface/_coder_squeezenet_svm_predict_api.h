/*
 * File: _coder_squeezenet_svm_predict_api.h
 *
 * GPU Coder version                    : 1.5
 * CUDA/C/C++ source code generated on  : 06-Aug-2020 05:06:39
 */

#ifndef _CODER_SQUEEZENET_SVM_PREDICT_API_H
#define _CODER_SQUEEZENET_SVM_PREDICT_API_H

/* Include Files */
#include <stddef.h>
#include <stdlib.h>
#include "tmwtypes.h"
#include "mex.h"
#include "emlrt.h"

/* Variable Declarations */
extern emlrtCTX emlrtRootTLSGlobal;
extern emlrtContext emlrtContextGlobal;

/* Function Declarations */
extern real_T squeezenet_svm_predict(uint8_T in[921600]);
extern void squeezenet_svm_predict_api(const mxArray * const prhs[1], int32_T
  nlhs, const mxArray *plhs[1]);
extern void squeezenet_svm_predict_atexit(void);
extern void squeezenet_svm_predict_initialize(void);
extern void squeezenet_svm_predict_terminate(void);
extern void squeezenet_svm_predict_xil_shutdown(void);
extern void squeezenet_svm_predict_xil_terminate(void);

#endif

/*
 * File trailer for _coder_squeezenet_svm_predict_api.h
 *
 * [EOF]
 */
