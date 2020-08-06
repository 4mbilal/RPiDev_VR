#include <cstdlib>
#include <cassert>
#include <stdio.h>
#include <cassert>
#include <iostream>
#include "MWCNNLayerImpl.hpp"
#include "MWTargetNetworkImpl.hpp"
#include "cnn_api.hpp"
#ifdef RANDOM
#include <curand.h>
 curandGenerator_t TYgANfbwgYWWZKKtdxCC; void 
curand_call_line_file(curandStatus_t tYWUxNVtgBrSjkBemGfF, const int 
aJTwGElOoWpBrmCfheqQ, const char* RtogJCavwOREhELwknZy) { if (tYWUxNVtgBrSjkBemGfF != 
CURAND_STATUS_SUCCESS) { char buffer[100]; int numElem = sprintf(buffer, 
"%d at line: %d, file: %s\n", tYWUxNVtgBrSjkBemGfF, aJTwGElOoWpBrmCfheqQ, 
RtogJCavwOREhELwknZy); throw std::runtime_error(buffer); } }
#endif
 void call_cuda_free(float* mem, const int aJTwGElOoWpBrmCfheqQ, const char* 
RtogJCavwOREhELwknZy) { if (!mem) { return; } cudaError_t tYWUxNVtgBrSjkBemGfF = 
cudaFree(mem); } float* malloc_call_line_file(size_t msize, const int 
aJTwGElOoWpBrmCfheqQ, const char *RtogJCavwOREhELwknZy) { float * mem = 
(float*)malloc(msize); if (!mem) { char buffer[100]; int numElem = 
sprintf(buffer, "%s at line: %d, file: %s\n", "Memory allocation failed. ", 
aJTwGElOoWpBrmCfheqQ, RtogJCavwOREhELwknZy); throw std::runtime_error(buffer); } return 
mem; } void cuda_call_line_file(cudaError_t tYWUxNVtgBrSjkBemGfF, const int 
aJTwGElOoWpBrmCfheqQ, const char* RtogJCavwOREhELwknZy) { if (tYWUxNVtgBrSjkBemGfF != 
cudaSuccess) { char buffer[100]; int numElem = sprintf(buffer, 
"Cuda Error %d(%s) at line: %d, file: %s\n", tYWUxNVtgBrSjkBemGfF, 
cudaGetErrorString(tYWUxNVtgBrSjkBemGfF), aJTwGElOoWpBrmCfheqQ, RtogJCavwOREhELwknZy); 
tYWUxNVtgBrSjkBemGfF = cudaGetLastError();  throw std::runtime_error(buffer); } } 
void cudnn_call_line_file(cudnnStatus_t tYWUxNVtgBrSjkBemGfF, const int 
aJTwGElOoWpBrmCfheqQ, const char* RtogJCavwOREhELwknZy) { if (tYWUxNVtgBrSjkBemGfF != 
CUDNN_STATUS_SUCCESS) {  char buffer[100]; int numElem = sprintf(buffer, 
"CuDNN Error %d(%s) at line: %d, file: %s\n", tYWUxNVtgBrSjkBemGfF, 
cudnnGetErrorString(tYWUxNVtgBrSjkBemGfF), aJTwGElOoWpBrmCfheqQ, RtogJCavwOREhELwknZy); 
throw std::runtime_error(buffer); } } 
MWCNNLayerImpl::MWCNNLayerImpl(MWCNNLayer* layer, MWTargetNetworkImpl* 
ntwk_impl) : aFIIJSbJDJUndqPzwinJ(layer) , dwFpvfypaTkJiYAULzFs(ntwk_impl) , 
RgALmBtPIZWDevjZBUHy(0.0) , RFQXHGHdWUKqrdBFLaiy(1.0) , QKDZxzoIvpFPYOFxhkKX(-1.0) , 
FCVkOsYRLlakBfSVryaq(0) { } MWCNNLayerImpl::~MWCNNLayerImpl() { 
for(std::map<int, cudnnTensorDescriptor_t*>::iterator it = 
lqYqXvaqiZxCnTfQwVBT.begin(); it != lqYqXvaqiZxCnTfQwVBT.end(); ++it) { 
delete it->second; it->second = 0; } } ITensor* 
MWCNNLayerImpl::getInputITensor(int inputIdx) { MWTensor* ipTensor = 
getLayer()->getInputTensor(inputIdx); assert(ipTensor); return 
getITensor(ipTensor); } ITensor* MWCNNLayerImpl::getITensor(MWTensor* tensor) { 
if (tensor->getOwner()->getImpl() == NULL) { return 
getITensor(tensor->getOwner()->getInputTensor(0)); } else { return 
tensor->getOwner()->getImpl()->getOpTensorPtr(tensor->getSourcePortIndex()); } 
} cudnnTensorDescriptor_t* MWCNNLayerImpl::getOutputDescriptor(int index) { 
std::map<int, cudnnTensorDescriptor_t*>::iterator it = 
lqYqXvaqiZxCnTfQwVBT.find(index); if (it == lqYqXvaqiZxCnTfQwVBT.end()) { 
cudnnTensorDescriptor_t* tmp = new cudnnTensorDescriptor_t; 
lqYqXvaqiZxCnTfQwVBT[index] = tmp; assert(tmp != 0); return tmp; } else { 
assert(it->second != 0); return it->second; } } void 
MWCNNLayerImpl::deallocateOutputData(){ for (int i = 0; i < 
getLayer()->getNumOutputs(); ++i){ MWTensor* opTensor = 
getLayer()->getOutputTensor(i); float* data = opTensor->getData<float>(); if 
(data) { CUDA_FREE_CALL(data); opTensor->setData((float*)NULL); } } } 
cudnnTensorDescriptor_t* MWCNNLayerImpl::getCuDNNDescriptor(MWTensor* tensor) { 
return tensor->getOwner()->getImpl()->getOutputDescriptor( 
tensor->getSourcePortIndex()); } int MWCNNLayerImpl::pluginEnqueueImpl(const 
void* const * , void** ){ assert(false); return 0; } 
MWPluginInterfaceImpl::MWPluginInterfaceImpl(MWCNNLayerImpl* 
PsSZzscVKwYLIATdyqkh) : m_cnnLayerImpl(PsSZzscVKwYLIATdyqkh){} Dims 
MWPluginInterfaceImpl::getOutputDimensions(int index, const Dims* , int ) { if 
(!m_cnnLayerImpl->dwFpvfypaTkJiYAULzFs->isSequenceNetwork){ int 
PiMNTwjpqwsGWomVWqdO = 
m_cnnLayerImpl->getLayer()->getOutputTensor(index)->getChannels(); int 
UGusHMMXcwaKODbaZkQs = 
m_cnnLayerImpl->getLayer()->getOutputTensor(index)->getHeight(); int 
znJVDnWdGXAXoBVlQhwT = 
m_cnnLayerImpl->getLayer()->getOutputTensor(index)->getWidth(); return 
DimsCHW(PiMNTwjpqwsGWomVWqdO, UGusHMMXcwaKODbaZkQs, znJVDnWdGXAXoBVlQhwT); }
#if (NV_TENSORRT_MAJOR >= 5)
 else{ int sPCEmfHYfjaRzyVvCKeA = 
m_cnnLayerImpl->getLayer()->getOutputTensor(index)->getSequenceLength(); int 
PiMNTwjpqwsGWomVWqdO = 
m_cnnLayerImpl->getLayer()->getOutputTensor(index)->getChannels(); int 
NhdIzzqqVxMjekDIWciw = 
m_cnnLayerImpl->getLayer()->getOutputTensor(index)->getBatchSize(); return 
Dims3(sPCEmfHYfjaRzyVvCKeA, NhdIzzqqVxMjekDIWciw, PiMNTwjpqwsGWomVWqdO); }
#endif 
 } void MWPluginInterfaceImpl::configure(const Dims* inputDims, int nbInputs, 
const Dims* outputDims, int nbOutputs, int ) { assert(inputDims->nbDims == 3);  
assert(outputDims->nbDims == 3);  assert(nbInputs == 
m_cnnLayerImpl->getLayer()->getNumInputs()); assert(nbOutputs == 
m_cnnLayerImpl->getLayer()->getNumOutputs()); } int 
MWPluginInterfaceImpl::getNbOutputs() const{ return 
m_cnnLayerImpl->getLayer()->getNumOutputs(); } int 
MWPluginInterfaceImpl::enqueue(int , const void* const* inputs, void** outputs, 
void* , cudaStream_t ) { m_cnnLayerImpl->pluginEnqueueImpl(inputs,outputs); 
return 0; } MWInputLayerImpl::MWInputLayerImpl(MWCNNLayer* layer, 
MWTargetNetworkImpl* ntwk_impl, int crKSAZwnyiinNFYODxoN, int ThkGOmtrxiMfUeOSxFsN, int 
wDZXrivCbTUzkLKxiPUh, int PIyXElJqMZoWKemWyTOa, int , const char* , int ) : 
MWCNNLayerImpl(layer, ntwk_impl) { MWTensor* opTensor = 
getLayer()->getOutputTensor(0); float * QHUGvHzeHXyFElIiOliL; 
CUDA_CALL(cudaMalloc((void**)&QHUGvHzeHXyFElIiOliL, sizeof(float) * ThkGOmtrxiMfUeOSxFsN * 
wDZXrivCbTUzkLKxiPUh * PIyXElJqMZoWKemWyTOa * crKSAZwnyiinNFYODxoN)); InputLayerITensor = 
dwFpvfypaTkJiYAULzFs->network->addInput( "data", DataType::kFLOAT, 
DimsCHW{PIyXElJqMZoWKemWyTOa, ThkGOmtrxiMfUeOSxFsN, wDZXrivCbTUzkLKxiPUh}); 
setOpTensorPtr(InputLayerITensor); opTensor->setData(QHUGvHzeHXyFElIiOliL); } void 
MWInputLayerImpl::cleanup() { for (int idx = 0; idx < 
aFIIJSbJDJUndqPzwinJ->getNumOutputs(); idx++) { float* data = 
aFIIJSbJDJUndqPzwinJ->getOutputTensor(idx)->getData<float>(); if (data) { 
CUDA_FREE_CALL(data); } } } MWReLULayerImpl::MWReLULayerImpl(MWCNNLayer* layer, 
MWTargetNetworkImpl* ntwk_impl, int , int ) : MWCNNLayerImpl(layer, ntwk_impl) 
, iReLULayer(0) { ITensor* prevLayerTensor = getInputITensor(0); iReLULayer = 
dwFpvfypaTkJiYAULzFs->network->addActivation(*prevLayerTensor, 
ActivationType::kRELU); iReLULayer->setName(getLayer()->getName().c_str()); 
setOpTensorPtr(iReLULayer->getOutput(0)); } 
MWNormLayerImpl::MWNormLayerImpl(MWCNNLayer* layer, MWTargetNetworkImpl* 
ntwk_impl, unsigned INKFbkrHldYkZFmALnfC,  double ATYqlAsSnRELrakAbCoK,  
double AjhVZuQXURJimwbnYqDF,  double EMtxAWxHxCcPIkaNDIHM, int ) : MWCNNLayerImpl(layer, 
ntwk_impl) { ITensor* prevLayerTensor = getInputITensor(0); iNormLayer = 
dwFpvfypaTkJiYAULzFs->network->addLRN(*prevLayerTensor, 
INKFbkrHldYkZFmALnfC, ATYqlAsSnRELrakAbCoK, AjhVZuQXURJimwbnYqDF, EMtxAWxHxCcPIkaNDIHM); 
iNormLayer->setName(getLayer()->getName().c_str()); 
setOpTensorPtr(iNormLayer->getOutput(0)); } void __global__ 
__launch_bounds__(1024) MWSetDyForBackPropImpl(float * RKrEonnJBdcnwoJXOHNM, const int 
gJJWRjXklapoEujuiRhJ) { for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < 
gJJWRjXklapoEujuiRhJ; i+= blockDim.x*gridDim.x) { RKrEonnJBdcnwoJXOHNM[i] = i+1; } } 
void __global__ __launch_bounds__(1024) doMWMaxPoolingLayerImpl(float * 
UdXKaaSyRlPmZWCVHODj, float * UROOthsHWeMcNycRifoq, const int 
EWUFPRDanwwTdrjmLomh) { for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < 
EWUFPRDanwwTdrjmLomh; i+= blockDim.x*gridDim.x) { if 
(static_cast<int>(UdXKaaSyRlPmZWCVHODj[i]) != 0){ 
UROOthsHWeMcNycRifoq[static_cast<int>(UdXKaaSyRlPmZWCVHODj[i])-1] = 
i; } } } MWMaxPoolingLayerImpl::MWMaxPoolingLayerImpl(MWCNNLayer* layer, 
MWTargetNetworkImpl* ntwk_impl, int HUlJqbQweWuEmHtzurnN,  int HXjVDkKHrBbbPfJOPKXp,  
int IGBjAMvMJXqrubGDtvyq,  int ILoEtPdZOaUSKUyteZTa, int GtFSKuDmLreppbjSISoU, int 
GbHRuweETkejIMGyqHDI,  int GmRRxuYauzGdhIlgciAT, int GrowsTaKrpHVUZdgZeJW, bool 
IggeMCRfncIoKgBcfyKF, int MW_mangled_, const std::vector<int>& ) : 
MWCNNLayerImpl(layer, ntwk_impl) , iMaxPoolingLayer(0) , 
HKStLBswJlYYprZPPGQx(0) , mJHKRLbEVeCWkBgMqjYL(0) , 
TwiaHttwApyaipMEKPSg(IggeMCRfncIoKgBcfyKF) { ITensor* prevLayerTensor = 
getInputITensor(0); if (!TwiaHttwApyaipMEKPSg && (GtFSKuDmLreppbjSISoU == 
GbHRuweETkejIMGyqHDI) && (GmRRxuYauzGdhIlgciAT == GrowsTaKrpHVUZdgZeJW)){ 
iMaxPoolingLayer = dwFpvfypaTkJiYAULzFs->network->addPooling( *prevLayerTensor, 
PoolingType::kMAX, DimsHW{HUlJqbQweWuEmHtzurnN, HXjVDkKHrBbbPfJOPKXp}); 
iMaxPoolingLayer->setStride(DimsHW{IGBjAMvMJXqrubGDtvyq, ILoEtPdZOaUSKUyteZTa}); 
iMaxPoolingLayer->setPadding(DimsHW{GtFSKuDmLreppbjSISoU, 
GmRRxuYauzGdhIlgciAT}); 
iMaxPoolingLayer->setName(getLayer()->getName().c_str()); 
setOpTensorPtr(iMaxPoolingLayer->getOutput(0)); } else{ 
pluginSetup(HUlJqbQweWuEmHtzurnN, HXjVDkKHrBbbPfJOPKXp, IGBjAMvMJXqrubGDtvyq, 
ILoEtPdZOaUSKUyteZTa, GtFSKuDmLreppbjSISoU, GmRRxuYauzGdhIlgciAT); 
mJHKRLbEVeCWkBgMqjYL = new MWPluginInterfaceImpl(this); HKStLBswJlYYprZPPGQx = 
dwFpvfypaTkJiYAULzFs->network->addPlugin(&prevLayerTensor, 1, 
*mJHKRLbEVeCWkBgMqjYL); setOpTensorPtr(HKStLBswJlYYprZPPGQx->getOutput(0),0); 
HKStLBswJlYYprZPPGQx->setName(getLayer()->getName().c_str()); if 
(TwiaHttwApyaipMEKPSg) setOpTensorPtr(HKStLBswJlYYprZPPGQx->getOutput(1),1); 
} } float* MWMaxPoolingLayerImpl::getIndexData() { return NULL; } void 
MWMaxPoolingLayerImpl::cleanup() { if (mJHKRLbEVeCWkBgMqjYL){ 
CUDNN_CALL(cudnnDestroyPoolingDescriptor(mMiQeYHoTbNUGLyNakei)); 
CUDNN_CALL(cudnnDestroyTensorDescriptor(*XOJRvKzQwSaZobhyUoOi)); 
CUDNN_CALL(cudnnDestroyTensorDescriptor(*jNxFsuLXTFYGOUlfRwLW)); } if 
(TwiaHttwApyaipMEKPSg) { 
CUDNN_CALL(cudnnDestroyTensorDescriptor(*jodrcfdVqEtXMUFJulgi)); 
CUDA_FREE_CALL(UdXKaaSyRlPmZWCVHODj); CUDA_FREE_CALL(RKrEonnJBdcnwoJXOHNM); } 
} void MWMaxPoolingLayerImpl::pluginSetup(int HUlJqbQweWuEmHtzurnN, int 
HXjVDkKHrBbbPfJOPKXp, int IGBjAMvMJXqrubGDtvyq, int ILoEtPdZOaUSKUyteZTa, int 
GtFSKuDmLreppbjSISoU, int GmRRxuYauzGdhIlgciAT){ MWTensor* ipTensor = 
getLayer()->getInputTensor();  
CUDNN_CALL(cudnnCreatePoolingDescriptor(&mMiQeYHoTbNUGLyNakei)); 
CUDNN_CALL(cudnnSetPooling2dDescriptor(mMiQeYHoTbNUGLyNakei, CUDNN_POOLING_MAX, 
CUDNN_NOT_PROPAGATE_NAN, HUlJqbQweWuEmHtzurnN, HXjVDkKHrBbbPfJOPKXp, 
GtFSKuDmLreppbjSISoU, GmRRxuYauzGdhIlgciAT, IGBjAMvMJXqrubGDtvyq, 
ILoEtPdZOaUSKUyteZTa)); XOJRvKzQwSaZobhyUoOi = new cudnnTensorDescriptor_t; 
CUDNN_CALL(cudnnCreateTensorDescriptor(XOJRvKzQwSaZobhyUoOi)); 
CUDNN_CALL(cudnnSetTensor4dDescriptor(*XOJRvKzQwSaZobhyUoOi, CUDNN_TENSOR_NCHW,  
CUDNN_DATA_FLOAT, ipTensor->getBatchSize(),  ipTensor->getChannels(),  
ipTensor->getHeight(),  ipTensor->getWidth()));  int crKSAZwnyiinNFYODxoN, 
PIyXElJqMZoWKemWyTOa, ThkGOmtrxiMfUeOSxFsN, wDZXrivCbTUzkLKxiPUh; 
CUDNN_CALL(cudnnGetPooling2dForwardOutputDim(mMiQeYHoTbNUGLyNakei, 
*XOJRvKzQwSaZobhyUoOi, &crKSAZwnyiinNFYODxoN ,&PIyXElJqMZoWKemWyTOa, &ThkGOmtrxiMfUeOSxFsN, 
&wDZXrivCbTUzkLKxiPUh)); ThkGOmtrxiMfUeOSxFsN = getLayer()->getOutputTensor(0)->getHeight(); 
wDZXrivCbTUzkLKxiPUh = getLayer()->getOutputTensor(0)->getWidth(); jNxFsuLXTFYGOUlfRwLW = 
new cudnnTensorDescriptor_t; 
CUDNN_CALL(cudnnCreateTensorDescriptor(jNxFsuLXTFYGOUlfRwLW)); 
CUDNN_CALL(cudnnSetTensor4dDescriptor(*jNxFsuLXTFYGOUlfRwLW, CUDNN_TENSOR_NCHW, 
CUDNN_DATA_FLOAT, crKSAZwnyiinNFYODxoN, PIyXElJqMZoWKemWyTOa, ThkGOmtrxiMfUeOSxFsN, wDZXrivCbTUzkLKxiPUh)); if 
(TwiaHttwApyaipMEKPSg){ jodrcfdVqEtXMUFJulgi = new cudnnTensorDescriptor_t; 
CUDNN_CALL(cudnnCreateTensorDescriptor(jodrcfdVqEtXMUFJulgi)); 
CUDNN_CALL(cudnnSetTensor4dDescriptor(*jodrcfdVqEtXMUFJulgi, CUDNN_TENSOR_NCHW, 
CUDNN_DATA_FLOAT, crKSAZwnyiinNFYODxoN, PIyXElJqMZoWKemWyTOa, ThkGOmtrxiMfUeOSxFsN, wDZXrivCbTUzkLKxiPUh)); 
assert((PIyXElJqMZoWKemWyTOa == ipTensor->getChannels()) && (crKSAZwnyiinNFYODxoN == 
ipTensor->getBatchSize()));  ekFKUFSJOBakwDuHENjA = 
(ipTensor->getHeight())*(ipTensor->getWidth())*(ipTensor->getChannels())*(ipTensor->getBatchSize()); 
CUDA_CALL(cudaMalloc((void**)&UdXKaaSyRlPmZWCVHODj, 
sizeof(float)*ekFKUFSJOBakwDuHENjA)); gJJWRjXklapoEujuiRhJ = 
wDZXrivCbTUzkLKxiPUh*ThkGOmtrxiMfUeOSxFsN*PIyXElJqMZoWKemWyTOa*crKSAZwnyiinNFYODxoN; 
CUDA_CALL(cudaMalloc((void**)&RKrEonnJBdcnwoJXOHNM, 
sizeof(float)*gJJWRjXklapoEujuiRhJ)); int vVyVzWKKaCvGClCSagOb = 
(gJJWRjXklapoEujuiRhJ < 1024) ? gJJWRjXklapoEujuiRhJ : 1024; int 
OuTwywxKeMgznElXdjGp = (gJJWRjXklapoEujuiRhJ + vVyVzWKKaCvGClCSagOb - 
1)/vVyVzWKKaCvGClCSagOb; 
MWSetDyForBackPropImpl<<<OuTwywxKeMgznElXdjGp, 
vVyVzWKKaCvGClCSagOb>>>( RKrEonnJBdcnwoJXOHNM, gJJWRjXklapoEujuiRhJ); } } int 
MWMaxPoolingLayerImpl::pluginEnqueueImpl(const void* const * inputs, void** 
outputs){ 
CUDNN_CALL(cudnnPoolingForward(*dwFpvfypaTkJiYAULzFs->getCudnnHandle(), 
mMiQeYHoTbNUGLyNakei, getOnePtr(), *XOJRvKzQwSaZobhyUoOi, (float*)inputs[0], 
getZeroPtr(), *jNxFsuLXTFYGOUlfRwLW, (float*)outputs[0])); if 
(TwiaHttwApyaipMEKPSg) { MWTensor* ipTensor = getLayer()->getInputTensor(); 
CUDNN_CALL(cudnnPoolingBackward(*dwFpvfypaTkJiYAULzFs->getCudnnHandle(), 
mMiQeYHoTbNUGLyNakei, getOnePtr(), *jNxFsuLXTFYGOUlfRwLW, (float*)outputs[0], 
*jNxFsuLXTFYGOUlfRwLW, RKrEonnJBdcnwoJXOHNM, *XOJRvKzQwSaZobhyUoOi, (float*)inputs[0], 
getZeroPtr(), *XOJRvKzQwSaZobhyUoOi, UdXKaaSyRlPmZWCVHODj)); int 
vVyVzWKKaCvGClCSagOb = (ekFKUFSJOBakwDuHENjA < 1024) ? ekFKUFSJOBakwDuHENjA : 
1024; int OuTwywxKeMgznElXdjGp = (ekFKUFSJOBakwDuHENjA + 
vVyVzWKKaCvGClCSagOb - 1)/vVyVzWKKaCvGClCSagOb; 
doMWMaxPoolingLayerImpl<<<OuTwywxKeMgznElXdjGp, 
vVyVzWKKaCvGClCSagOb>>>( UdXKaaSyRlPmZWCVHODj, 
(float*)outputs[1], ekFKUFSJOBakwDuHENjA); } return 0; } 
MWFCLayerImpl::MWFCLayerImpl(MWCNNLayer* layer, MWTargetNetworkImpl* ntwk_impl, 
int EpwuhXsRcwdqXSjBpUeO, const char* zdjrTcsHRcPcpoVFAMfT,  const 
char* OWgntZrUmlZXHAsNObcq, int ) : MWCNNLayerImpl(layer, ntwk_impl) , 
iFCLayer(0) { MWTensor* opTensor = getLayer()->getOutputTensor(0); MWTensor* 
ipTensor = getLayer()->getInputTensor(0); wDZXrivCbTUzkLKxiPUh = 
(float*)calloc(EpwuhXsRcwdqXSjBpUeO * opTensor->getChannels(), 
sizeof(float)); NSzdekOvRhMhRCXdWsdY = (float*)calloc(opTensor->getChannels(), 
sizeof(float)); int eWYFXrUazhqiEIscccda = EpwuhXsRcwdqXSjBpUeO * 
opTensor->getChannels();  loadWeights(eWYFXrUazhqiEIscccda, zdjrTcsHRcPcpoVFAMfT); 
loadBias(OWgntZrUmlZXHAsNObcq); ITensor* prevLayerITensor = getInputITensor(0); 
filt_weights.values = wDZXrivCbTUzkLKxiPUh; filt_weights.count = 
EpwuhXsRcwdqXSjBpUeO * opTensor->getChannels(); filt_weights.type = 
DataType::kFLOAT; filt_bias.values = NSzdekOvRhMhRCXdWsdY; filt_bias.count = 
opTensor->getChannels(); filt_bias.type = DataType::kFLOAT; if 
(!dwFpvfypaTkJiYAULzFs->isSequenceNetwork){ iFCLayer = 
dwFpvfypaTkJiYAULzFs->network->addFullyConnected( *prevLayerITensor, 
opTensor->getChannels(), filt_weights, filt_bias); 
iFCLayer->setName(getLayer()->getName().c_str()); 
setOpTensorPtr(iFCLayer->getOutput(0)); }
#if (NV_TENSORRT_MAJOR >= 5)
 else{ auto shuffleLayer = 
dwFpvfypaTkJiYAULzFs->network->addShuffle(*prevLayerITensor); 
assert(shuffleLayer); shuffleLayer->setFirstTranspose(Permutation{1, 0, 2}); 
auto fcwts = dwFpvfypaTkJiYAULzFs->network->addConstant(Dims3(1, 
opTensor->getChannels(), EpwuhXsRcwdqXSjBpUeO), filt_weights);
#if (NV_TENSORRT_MAJOR >= 5 && NV_TENSORRT_MINOR >= 1)
 auto matrixMultLayer = dwFpvfypaTkJiYAULzFs->network->addMatrixMultiply( 
*fcwts->getOutput(0), MatrixOperation::kNONE, *shuffleLayer->getOutput(0), MatrixOperation::kTRANSPOSE);
#else
 auto matrixMultLayer = dwFpvfypaTkJiYAULzFs->network->addMatrixMultiply( 
*fcwts->getOutput(0), false, *shuffleLayer->getOutput(0), true);
#endif
 assert(matrixMultLayer != nullptr); auto fcbias = 
dwFpvfypaTkJiYAULzFs->network->addConstant(Dims3(1, opTensor->getChannels(), 1), 
filt_bias); auto elementWiseLayer = 
dwFpvfypaTkJiYAULzFs->network->addElementWise(*matrixMultLayer->getOutput(0), 
*fcbias->getOutput(0), ElementWiseOperation::kSUM); assert(elementWiseLayer != 
nullptr); shuffleLayer = 
dwFpvfypaTkJiYAULzFs->network->addShuffle(*elementWiseLayer->getOutput(0)); 
assert(shuffleLayer); shuffleLayer->setFirstTranspose(Permutation{2, 0, 1}); 
setOpTensorPtr(shuffleLayer->getOutput(0)); }
#endif
 } void MWFCLayerImpl::loadWeights(int eWYFXrUazhqiEIscccda, const char* 
RuGYRQXjIMQJrbgoRUxZ) { MWFCLayer* fcLayer = 
static_cast<MWFCLayer*>(getLayer()); MWTensor* ipTensor = 
fcLayer->getInputTensor(0); FILE* SmibqCQPbtzycGEpwhpN = 
MWCNNLayer::openBinaryFile(RuGYRQXjIMQJrbgoRUxZ); assert(SmibqCQPbtzycGEpwhpN); 
call_fread(wDZXrivCbTUzkLKxiPUh, sizeof(float), eWYFXrUazhqiEIscccda, SmibqCQPbtzycGEpwhpN, 
RuGYRQXjIMQJrbgoRUxZ); if (ipTensor->getHeight() != 1 && ipTensor->getWidth() != 
1) { float* PAwKCndEJEByqwNZnPgb = (float*)malloc(sizeof(float) * 
ipTensor->getHeight() * ipTensor->getWidth()); for (int k = 0; k < 
eWYFXrUazhqiEIscccda / ipTensor->getHeight() / ipTensor->getWidth(); k++) { for (int 
i = 0; i < ipTensor->getHeight() * ipTensor->getWidth(); i++) { 
PAwKCndEJEByqwNZnPgb[i] = wDZXrivCbTUzkLKxiPUh[k * ipTensor->getHeight() * 
ipTensor->getWidth() + i]; } for (int j = 0; j < ipTensor->getHeight(); j++) 
for (int i = 0; i < ipTensor->getWidth(); i++) { wDZXrivCbTUzkLKxiPUh[k * 
ipTensor->getHeight() * ipTensor->getWidth() + j * ipTensor->getWidth() + i] = 
PAwKCndEJEByqwNZnPgb[j + i * ipTensor->getHeight()]; } } 
free(PAwKCndEJEByqwNZnPgb); } fclose(SmibqCQPbtzycGEpwhpN); } void 
MWFCLayerImpl::loadBias(const char* RuGYRQXjIMQJrbgoRUxZ) { MWTensor* opTensor = 
getLayer()->getOutputTensor(0); FILE* SmibqCQPbtzycGEpwhpN = 
MWCNNLayer::openBinaryFile(RuGYRQXjIMQJrbgoRUxZ); assert(SmibqCQPbtzycGEpwhpN); int 
eWYFXrUazhqiEIscccda = opTensor->getChannels();  call_fread(NSzdekOvRhMhRCXdWsdY, 
sizeof(float), eWYFXrUazhqiEIscccda, SmibqCQPbtzycGEpwhpN, RuGYRQXjIMQJrbgoRUxZ); 
fclose(SmibqCQPbtzycGEpwhpN); } void MWFCLayerImpl::cleanup() { free(wDZXrivCbTUzkLKxiPUh); 
free(NSzdekOvRhMhRCXdWsdY); } MWSoftmaxLayerImpl::MWSoftmaxLayerImpl(MWCNNLayer* layer, 
MWTargetNetworkImpl* ntwk_impl, int ) : MWCNNLayerImpl(layer, ntwk_impl) , 
iSoftmaxLayer(0) { MWTensor* opTensor = getLayer()->getOutputTensor(0); 
ITensor* prevLayerTensor = getInputITensor(0); if 
(!dwFpvfypaTkJiYAULzFs->isSequenceNetwork){ iSoftmaxLayer = 
dwFpvfypaTkJiYAULzFs->network->addSoftMax(*prevLayerTensor); }
#if (NV_TENSORRT_MAJOR >= 5) 
 else{ iSoftmaxLayer = 
dwFpvfypaTkJiYAULzFs->network->addSoftMax(*prevLayerTensor); 
iSoftmaxLayer->setAxes(1<<2); }
#endif
 iSoftmaxLayer->setName(getLayer()->getName().c_str()); 
setOpTensorPtr(iSoftmaxLayer->getOutput(0)); } 
MWOutputLayerImpl::MWOutputLayerImpl(MWCNNLayer* layer, MWTargetNetworkImpl* 
ntwk_impl, int ) : MWCNNLayerImpl(layer, ntwk_impl) { MWTensor* opTensor = 
getLayer()->getOutputTensor(0); float * QHUGvHzeHXyFElIiOliL; 
CUDA_CALL(cudaMalloc((void**)&QHUGvHzeHXyFElIiOliL, sizeof(float) * 
opTensor->getNumElements())); ITensor* prevLayerTensor = getInputITensor(0); 
setOpTensorPtr(prevLayerTensor); opTensor->setData(QHUGvHzeHXyFElIiOliL); } void 
MWOutputLayerImpl::cleanup() { for (int idx = 0; idx < 
aFIIJSbJDJUndqPzwinJ->getNumOutputs(); idx++) { float* data = 
aFIIJSbJDJUndqPzwinJ->getOutputTensor(idx)->getData<float>(); if (data) { 
CUDA_FREE_CALL(data); } } } 
MWAvgPoolingLayerImpl::MWAvgPoolingLayerImpl(MWCNNLayer* layer, 
MWTargetNetworkImpl* ntwk_impl, int HUlJqbQweWuEmHtzurnN,  int HXjVDkKHrBbbPfJOPKXp,  
int IGBjAMvMJXqrubGDtvyq,  int ILoEtPdZOaUSKUyteZTa,  int GtFSKuDmLreppbjSISoU,  
int GbHRuweETkejIMGyqHDI,  int GmRRxuYauzGdhIlgciAT, int GrowsTaKrpHVUZdgZeJW, 
int ) : MWCNNLayerImpl(layer, ntwk_impl) , iAvgPoolingLayer(0) { ITensor* 
prevLayerTensor = getInputITensor(0); if((GtFSKuDmLreppbjSISoU == 
GbHRuweETkejIMGyqHDI) && (GmRRxuYauzGdhIlgciAT == GrowsTaKrpHVUZdgZeJW)){  
iAvgPoolingLayer = dwFpvfypaTkJiYAULzFs->network->addPooling( *prevLayerTensor, 
PoolingType::kAVERAGE, DimsHW{HUlJqbQweWuEmHtzurnN, HXjVDkKHrBbbPfJOPKXp}); 
iAvgPoolingLayer->setPadding(DimsHW{GtFSKuDmLreppbjSISoU, 
GmRRxuYauzGdhIlgciAT}); } else { IPaddingLayer* iPaddingLayer = 
dwFpvfypaTkJiYAULzFs->network->addPadding( *prevLayerTensor, 
DimsHW{GtFSKuDmLreppbjSISoU,GmRRxuYauzGdhIlgciAT}, 
DimsHW{GbHRuweETkejIMGyqHDI,GrowsTaKrpHVUZdgZeJW}); ITensor* 
FCgChnBQLAUXyoWzTlfg = iPaddingLayer->getOutput(0); iAvgPoolingLayer = 
dwFpvfypaTkJiYAULzFs->network->addPooling( *FCgChnBQLAUXyoWzTlfg, 
PoolingType::kAVERAGE, DimsHW{HUlJqbQweWuEmHtzurnN, HXjVDkKHrBbbPfJOPKXp});  } 
iAvgPoolingLayer->setStride(DimsHW{IGBjAMvMJXqrubGDtvyq, ILoEtPdZOaUSKUyteZTa}); 
iAvgPoolingLayer->setAverageCountExcludesPadding(false); 
iAvgPoolingLayer->setName(getLayer()->getName().c_str()); 
setOpTensorPtr(iAvgPoolingLayer->getOutput(0)); }