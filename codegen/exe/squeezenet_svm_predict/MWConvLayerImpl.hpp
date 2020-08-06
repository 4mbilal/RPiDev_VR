
/* Copyright 2018 The MathWorks, Inc. */

#ifndef _MW_CONV_LAYER_IMPL
#define _MW_CONV_LAYER_IMPL

#include "MWCNNLayerImpl.hpp"

// Convolution2DWCNNLayer

class MWConvLayerImpl : public MWCNNLayerImpl {

  public:

    MWConvLayerImpl(MWCNNLayer*,
                    MWTargetNetworkImpl*,
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,
                    const char*,
                    const char*,
                    int outbufIdx);
    ~MWConvLayerImpl() {}
     void cleanup();

  private:

    int EFRxTzGDLCOxVeZLDhRL; 
    int EGsHUnogBQpOwCZJYeUd;  
    int EiBytenrthqoQrTnOFaK;
   
    float* wDZXrivCbTUzkLKxiPUh;
    float* NSzdekOvRhMhRCXdWsdY;
    void loadWeights(const char*);
    void loadBias(const char*);

    /**
     * TensorRT related
     */
    IConvolutionLayer* ConvLayerT;
    Weights filt_weights;
    Weights filt_bias;
};

#endif
