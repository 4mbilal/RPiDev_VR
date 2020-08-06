/*
 * File: _coder_squeezenet_svm_predict_mex.c
 *
 * GPU Coder version                    : 1.5
 * CUDA/C/C++ source code generated on  : 06-Aug-2020 05:06:39
 */

/* Include Files */
#include "_coder_squeezenet_svm_predict_mex.h"
#include "_coder_squeezenet_svm_predict_api.h"

/* Function Declarations */
MEXFUNCTION_LINKAGE void c_squeezenet_svm_predict_mexFun(int32_T nlhs, mxArray
  *plhs[1], int32_T nrhs, const mxArray *prhs[1]);

/* Function Definitions */

/*
 * Arguments    : int32_T nlhs
 *                mxArray *plhs[1]
 *                int32_T nrhs
 *                const mxArray *prhs[1]
 * Return Type  : void
 */
void c_squeezenet_svm_predict_mexFun(int32_T nlhs, mxArray *plhs[1], int32_T
  nrhs, const mxArray *prhs[1])
{
  const mxArray *outputs[1];
  emlrtStack st = { NULL,              /* site */
    NULL,                              /* tls */
    NULL                               /* prev */
  };

  st.tls = emlrtRootTLSGlobal;

  /* Check for proper number of arguments. */
  if (nrhs != 1) {
    emlrtErrMsgIdAndTxt(&st, "EMLRT:runTime:WrongNumberOfInputs", 5, 12, 1, 4,
                        22, "squeezenet_svm_predict");
  }

  if (nlhs > 1) {
    emlrtErrMsgIdAndTxt(&st, "EMLRT:runTime:TooManyOutputArguments", 3, 4, 22,
                        "squeezenet_svm_predict");
  }

  /* Call the function. */
  squeezenet_svm_predict_api(prhs, nlhs, outputs);

  /* Copy over outputs to the caller. */
  emlrtReturnArrays(1, plhs, outputs);
}

/*
 * Arguments    : int32_T nlhs
 *                mxArray *plhs[]
 *                int32_T nrhs
 *                const mxArray *prhs[]
 * Return Type  : void
 */
void mexFunction(int32_T nlhs, mxArray *plhs[], int32_T nrhs, const mxArray
                 *prhs[])
{
  mexAtExit(&squeezenet_svm_predict_atexit);

  /* Module initialization. */
  squeezenet_svm_predict_initialize();

  /* Dispatch the entry-point. */
  c_squeezenet_svm_predict_mexFun(nlhs, plhs, nrhs, prhs);

  /* Module termination. */
  squeezenet_svm_predict_terminate();
}

/*
 * Arguments    : void
 * Return Type  : emlrtCTX
 */
emlrtCTX mexFunctionCreateRootTLS(void)
{
  emlrtCreateRootTLS(&emlrtRootTLSGlobal, &emlrtContextGlobal, NULL, 1);
  return emlrtRootTLSGlobal;
}

/*
 * File trailer for _coder_squeezenet_svm_predict_mex.c
 *
 * [EOF]
 */
