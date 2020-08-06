/* Copyright 2017-2018 The MathWorks, Inc. */

#ifndef CNN_TARGET_NTWK_IMPL
#define CNN_TARGET_NTWK_IMPL

#include <cudnn.h>
#include <string>

/*TensorRT related header files */
#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "cuda_runtime_api.h"
#include <map>

#define MW_TARGET_TYPE_TENSORRT 1

using namespace nvinfer1;
using namespace nvcaffeparser1;

class MWCNNLayer;

class MWTargetNetworkImpl {

  public:
    
    MWTargetNetworkImpl();
    ~MWTargetNetworkImpl();
    
    void preSetup();
    void allocate(int, int);
    void deallocate();
    void postSetup(MWCNNLayer* layers[],int numLayers, int layerIdxs[], int portIdxs[], int numOuts);
    void doInference(int batchSize);
    void cleanup();
    cudnnHandle_t* getCudnnHandle(); 
    
    INetworkDefinition* network;
    int batchSize;
    bool isSequenceNetwork {false};

    void setBatchSize(int);
    void setIsSequenceNetwork(bool);
    
  private:
    
    IBuilder* builder;
    ICudaEngine* engine;
    IExecutionContext* context;
    size_t ppmKOQPnPGmtxACrpKWE;
    cudnnHandle_t* PuqtLkzghXGwKBHOGAzf;

    void** m_buffers;
    
  private:
    
    void setupBuffers(MWCNNLayer* layers[], int layerIdxs[], int portIdxs[], int numOuts,
        std::map<int, std::pair<float*, std::string> > & buffers);
    float* getBuffer(MWCNNLayer* layer, int layerIdx, int portIdx);
    void markOutputs(MWCNNLayer* layers[], int layerIdxs[], int numOuts);
    
};
#endif
