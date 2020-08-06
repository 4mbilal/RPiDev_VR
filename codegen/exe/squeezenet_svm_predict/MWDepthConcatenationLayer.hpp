/* Copyright 2017 The MathWorks, Inc. */

#ifndef __DEPTH_CONCATENTION_LAYER_HPP
#define __DEPTH_CONCATENTION_LAYER_HPP

#include "cnn_api.hpp"

/**
 *  Codegen class for Depth Concatenation Layer
 *  Depth Concatenation layer
 **/
class MWTargetNetworkImpl;
class MWDepthConcatenationLayer : public MWCNNLayer
{
  public:
    MWDepthConcatenationLayer();
    ~MWDepthConcatenationLayer();

    void createDepthConcatenationLayer(MWTargetNetworkImpl* , int numInputs, ...);
    void propagateSize();
};

#endif
