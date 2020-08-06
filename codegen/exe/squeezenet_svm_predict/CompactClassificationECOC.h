//
// File: CompactClassificationECOC.h
//
// GPU Coder version                    : 1.5
// CUDA/C/C++ source code generated on  : 06-Aug-2020 05:06:39
//
#ifndef COMPACTCLASSIFICATIONECOC_H
#define COMPACTCLASSIFICATIONECOC_H

// Include Files
#include <cstddef>
#include <cstdlib>
#include "rtwtypes.h"
#include "squeezenet_svm_predict_types.h"

// Type Definitions
#include "cublas_v2.h"

// Function Declarations
extern void c_CompactClassificationECOC_pre(const double obj_ClassNames[7],
  const double obj_Prior[7], const double obj_CodingMatrix[147], const float
  Xin[1000], double *labels, float negloss[7]);

#endif

//
// File trailer for CompactClassificationECOC.h
//
// [EOF]
//
