/* Copyright 2017-2019 The MathWorks, Inc. */

#ifndef CNN_API_IMPL
#define CNN_API_IMPL

#include <cudnn.h>
#include <map>
#include <vector>
#include <cassert>

/* TensorRT related header files */
#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "cuda_runtime_api.h"
#include "cnn_api.hpp"


using namespace nvinfer1;
using namespace nvcaffeparser1;


class MWTargetNetworkImpl;

#define CUDA_CALL(status) cuda_call_line_file(status, __LINE__, __FILE__)
#define CUDNN_CALL(status) cudnn_call_line_file(status, __LINE__, __FILE__)
#define MALLOC_CALL(msize) malloc_call_line_file(msize,__LINE__,__FILE__)
#define CUDA_FREE_CALL(buf) call_cuda_free(buf,__LINE__,__FILE__)

//#define RANDOM
#ifdef RANDOM
#include <curand.h>
#define CURAND_CALL(status) curand_call_line_file(status, __LINE__, __FILE__)
#endif

void cuda_call_line_file(cudaError_t, const int, const char*);
float* malloc_call_line_file(size_t, const int, const char *);
void cudnn_call_line_file(cudnnStatus_t, const int, const char*);
void call_cuda_free(float* mem, int, const char*);
#ifdef RANDOM
void curand_call_line_file(curandStatus_t, const int, const char*);
#endif


class MWCNNLayerImpl {

  public:
    MWCNNLayerImpl(MWCNNLayer* layer, MWTargetNetworkImpl*);
    virtual ~MWCNNLayerImpl();
    MWTargetNetworkImpl* dwFpvfypaTkJiYAULzFs;

    void predict() {}

    // cleanup is called for all layers but may do nothing.
    virtual void cleanup() {
    }
    virtual void postSetup() {
    }

    virtual void propagateSize() {
    }
    
    // allocate is called for all layers but may do nothing.
    void allocate() {
    }

    void deallocate() {
    }

    virtual void allocateOutputData() {
    }
    virtual void deallocateOutputData();

    template <typename T>
            T* getData(int index = 0);
            
    MWCNNLayer* getLayer() {
        return aFIIJSbJDJUndqPzwinJ;
    }

    // Get the previous layer output pointer
    ITensor* getOpTensorPtr(int idx = 0) {
        assert(idx < FCVkOsYRLlakBfSVryaq.size());
        return FCVkOsYRLlakBfSVryaq[idx];
    }

    // get iTensor corresponding to MWTensor
    static ITensor* getITensor(MWTensor* tensor);

    // Plugin layer Enqueue interface
    virtual int pluginEnqueueImpl(const void* const *, void**);


  protected:
    MWCNNLayer* aFIIJSbJDJUndqPzwinJ;
    std::map<int, cudnnTensorDescriptor_t*> lqYqXvaqiZxCnTfQwVBT;
    
    float RgALmBtPIZWDevjZBUHy;
    float RFQXHGHdWUKqrdBFLaiy;
    float QKDZxzoIvpFPYOFxhkKX;

    // Get the cuDNN tensor descriptor for the output
    cudnnTensorDescriptor_t* getOutputDescriptor(int index = 0);
    // get Descriptor from a tensor
    cudnnTensorDescriptor_t* getCuDNNDescriptor(MWTensor* tensor);

    float* getZeroPtr() {
        return &RgALmBtPIZWDevjZBUHy;
    }
    float* getOnePtr() {
        return &RFQXHGHdWUKqrdBFLaiy;
    }
    float* getNegOnePtr() {
        return &QKDZxzoIvpFPYOFxhkKX;
    }

    /**
     * TensorRT API related
     */

    std::vector<ITensor*> FCVkOsYRLlakBfSVryaq;

    // set the Output tensor pointer
    void setOpTensorPtr(ITensor* outputTensor, int idx = 0) {
        assert(idx == FCVkOsYRLlakBfSVryaq.size());
        FCVkOsYRLlakBfSVryaq.push_back(outputTensor);
    }

    // get iTensor corresponding to input
    ITensor* getInputITensor(int inputIdx);

};

template <typename T>
T* MWCNNLayerImpl::getData(int index)
{
    T* data = getLayer()->getOutputTensor(index)->getData<T>();
    assert(data);
    return data;
}

// Common base class for all Plugin Layers of TensorRT
class MWPluginInterfaceImpl : public IPlugin {

  public:
    MWPluginInterfaceImpl(MWCNNLayerImpl*);

    virtual ~MWPluginInterfaceImpl() {
    }

    virtual int initialize() override {
        return 0;
    }

    virtual size_t getSerializationSize() override {
        return 0;
    }
    virtual void serialize(void* buffer) override {
    }
    virtual size_t getWorkspaceSize(int maxBatchSize) const override {
        assert(maxBatchSize != 0);
        return 0;
    }

  
    virtual int getNbOutputs() const override;
    virtual Dims getOutputDimensions(int, const Dims*, int) override;
    virtual void configure(const Dims*, int, const Dims*, int, int) override;
    virtual int enqueue(int, const void* const*, void**, void*, cudaStream_t) override;
    virtual void terminate() override {}

  protected:

    MWCNNLayerImpl* m_cnnLayerImpl;
};

class MWInputLayerImpl : public MWCNNLayerImpl {

  public:
    MWInputLayerImpl(MWCNNLayer* layer,
                     MWTargetNetworkImpl* ntwk_impl,
                     int,
                     int,
                     int,
                     int,
                     int,
                     const char* avg_file_name,
                     int outbufIdx);
    ~MWInputLayerImpl() {
    }

    void cleanup();

  private:
    /**
     * TensorRT related
     */
    ITensor* InputLayerITensor;
};

// ReLULayer

class MWReLULayerImpl : public MWCNNLayerImpl {

  public:
    MWReLULayerImpl(MWCNNLayer*, MWTargetNetworkImpl*, int, int);
    ~MWReLULayerImpl() {
    }

  private:
    /**
     * TensorRT API related
     */
    IActivationLayer* iReLULayer;
};

class MWNormLayerImpl : public MWCNNLayerImpl {

  public:
    MWNormLayerImpl(MWCNNLayer*,
                    MWTargetNetworkImpl*,
                    unsigned,
                    double,
                    double,
                    double,
                    int outbufIdx);
    ~MWNormLayerImpl() {
    }

  private:
    /**
     * TensorRT API related
     **/
    ILRNLayer* iNormLayer;
};

// MaxPooling2DLayer

class MWMaxPoolingLayerImpl : public MWCNNLayerImpl {

  public:
    MWMaxPoolingLayerImpl(MWCNNLayer*,
                          MWTargetNetworkImpl*,
                          int,  /* PoolH */
                          int,  /* PoolW */
                          int,  /* StrideH */
                          int,  /* StrideW */
                          int,  /* PaddingH_Top */
                          int,  /* PaddingH_Bottom */
                          int,  /* PaddingH_Left */
                          int,  /* PaddingH_Right */
                          bool, /* hasIndices */
                          int,  /* numOutputs */
                          const std::vector<int>& bufIndices);

    ~MWMaxPoolingLayerImpl() {
    };

    void cleanup();

    // getter for max pool indices , used by unpooling
    float* getIndexData();

    int pluginEnqueueImpl(const void* const *, void**);

  private:
    IPoolingLayer* iMaxPoolingLayer;

    // Asymmetric Padding
    IPluginLayer* HKStLBswJlYYprZPPGQx;
    IPlugin* mJHKRLbEVeCWkBgMqjYL;

    cudnnPoolingDescriptor_t mMiQeYHoTbNUGLyNakei;
    cudnnTensorDescriptor_t*  XOJRvKzQwSaZobhyUoOi;
    cudnnTensorDescriptor_t*  jNxFsuLXTFYGOUlfRwLW;
    cudnnTensorDescriptor_t*  jodrcfdVqEtXMUFJulgi;

    // 2-output MaxPool plugin setup
    bool TwiaHttwApyaipMEKPSg;
    int ekFKUFSJOBakwDuHENjA;
    int gJJWRjXklapoEujuiRhJ;

    float* UdXKaaSyRlPmZWCVHODj;
    float* RKrEonnJBdcnwoJXOHNM;

    void pluginSetup(int, int, int, int, int, int);

};

// FullyConnectedLayer

class MWFCLayerImpl : public MWCNNLayerImpl {

  public:
    MWFCLayerImpl(MWCNNLayer*, MWTargetNetworkImpl*, int, const char*, const char*, int);
    ~MWFCLayerImpl() {
    }
    void cleanup();

  private:
    float* wDZXrivCbTUzkLKxiPUh;
    float* NSzdekOvRhMhRCXdWsdY;

    void loadWeights(int, const char*);
    void loadBias(const char*);
    
    /**
     * Tensorrt API related
     */
    IFullyConnectedLayer* iFCLayer;
    Weights filt_weights;
    Weights filt_bias;
};

// SoftmaxLayer
class MWSoftmaxLayerImpl : public MWCNNLayerImpl {

  public:
    MWSoftmaxLayerImpl(MWCNNLayer*, MWTargetNetworkImpl*, int);
    ~MWSoftmaxLayerImpl() {
    }

  private:
    ISoftMaxLayer* iSoftmaxLayer;
};

// SoftmaxLayer
class MWOutputLayerImpl : public MWCNNLayerImpl {

  public:
    MWOutputLayerImpl(MWCNNLayer*, MWTargetNetworkImpl*, int);
    ~MWOutputLayerImpl() {
    }
    void cleanup();
};

// AvgPooling2DLayer
class MWAvgPoolingLayerImpl : public MWCNNLayerImpl {

  public:
    MWAvgPoolingLayerImpl(MWCNNLayer*,
                          MWTargetNetworkImpl*,
                          int, /* PoolH */
                          int, /* PoolW */
                          int, /* StrideH */
                          int, /* StrideW */
                          int, /* PaddingT */
                          int, /* PaddingB */
                          int, /* PaddingL */
                          int, /* PaddingR */
                          int  /* OutBufferIdx */
    );
    ~MWAvgPoolingLayerImpl() {
    }

  private:
    IPoolingLayer* iAvgPoolingLayer;
};

#endif
