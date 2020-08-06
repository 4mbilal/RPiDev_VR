#include "MWElementwiseAffineLayerImpl.hpp"
#include "MWTargetNetworkImpl.hpp"
#include "cnn_api.hpp"
#include "MWKernelHeaders.hpp"
 MWElementwiseAffineLayerImpl::MWElementwiseAffineLayerImpl(MWCNNLayer* layer, 
MWTargetNetworkImpl* ntwk_impl, int scale_H,  int scale_W,  int scale_C, int 
offset_H,  int offset_W,  int offset_C, bool isClipped,  int lowerbound,  int 
upperbound, const char* scale_file,  const char* offset_file, int ) : 
MWCNNLayerImpl(layer, ntwk_impl), pxmnUEWGnfCxJNuDkXAo(NULL), 
hTbrmZSPUORqMIprVbKk(NULL), rZyMIPooLjRiXLgSWDuw(scale_H), 
rfIckzovFcUSejaEKNlY(scale_W), qquNiJHQtfSLDMNCPIBJ(scale_C), 
hpOzCTZasBMYKoXVxMDZ(offset_H), hqbKXLMjsDxRQqyJEgbg(offset_W), 
hjXpmIaeqlKKSDDBSxtE(offset_C), ZqQxEyCjEixByRZYMkbj(isClipped), 
boZxhUbgnNUlJQXTuWwi(lowerbound), voqEJSkAwmNPuqzoiuom(upperbound), 
rwPhFWHcKnJsClVtebGW(nullptr), ikTyjLTPRBkBRlLSyxXG(nullptr), 
qJWXFXvcpbSwehmlTNru(0), HKStLBswJlYYprZPPGQx(0), mJHKRLbEVeCWkBgMqjYL(0) { 
loadScaleAndOffset(scale_file, offset_file); setLayerProperties(); bool 
isMatrix2d = (rZyMIPooLjRiXLgSWDuw > 1) && (rfIckzovFcUSejaEKNlY > 1) && 
(qquNiJHQtfSLDMNCPIBJ != WShIchjHSSzOoYZernlN); if ((!ZqQxEyCjEixByRZYMkbj) && 
(reGtUwUlPSwEenEBVIzH == hqVFaqkobRNLQNgtbaai ) && !isMatrix2d && 
(!dwFpvfypaTkJiYAULzFs->isSequenceNetwork)) { qeQuIDaHqnxGPDbPoQJF.values 
= rwPhFWHcKnJsClVtebGW; qeQuIDaHqnxGPDbPoQJF.count = reGtUwUlPSwEenEBVIzH; 
qeQuIDaHqnxGPDbPoQJF.type = DataType::kFLOAT; 
niGnnRufksTFnsUUxnCj.values = nullptr; niGnnRufksTFnsUUxnCj.count = 
0; niGnnRufksTFnsUUxnCj.type = DataType::kFLOAT; 
suFVgcuEVpCOrewbJfkB.values = ikTyjLTPRBkBRlLSyxXG; 
suFVgcuEVpCOrewbJfkB.count = hqVFaqkobRNLQNgtbaai; 
suFVgcuEVpCOrewbJfkB.type = DataType::kFLOAT; ITensor* prevLayerTensor = 
getInputITensor(0); ScaleMode mode; if (reGtUwUlPSwEenEBVIzH == 1) mode = 
ScaleMode::kUNIFORM; else if (YMNbgnUYZspjMLjwcIOS == 
reGtUwUlPSwEenEBVIzH) mode = ScaleMode::kELEMENTWISE; else if (rZyMIPooLjRiXLgSWDuw 
== 1 && rfIckzovFcUSejaEKNlY == 1 && reGtUwUlPSwEenEBVIzH == qquNiJHQtfSLDMNCPIBJ) 
mode = ScaleMode::kCHANNEL; qJWXFXvcpbSwehmlTNru = 
dwFpvfypaTkJiYAULzFs->network->addScale(*prevLayerTensor,  mode,  
suFVgcuEVpCOrewbJfkB, qeQuIDaHqnxGPDbPoQJF,  
niGnnRufksTFnsUUxnCj); assert(qJWXFXvcpbSwehmlTNru); 
qJWXFXvcpbSwehmlTNru->setName(getLayer()->getName().c_str()); 
setOpTensorPtr(qJWXFXvcpbSwehmlTNru->getOutput(0)); } else { ITensor* 
prevLayerTensor = getInputITensor(0); mJHKRLbEVeCWkBgMqjYL = new 
MWPluginInterfaceImpl(this); HKStLBswJlYYprZPPGQx = 
dwFpvfypaTkJiYAULzFs->network->addPlugin(&prevLayerTensor, 1, 
*mJHKRLbEVeCWkBgMqjYL); 
HKStLBswJlYYprZPPGQx->setName(getLayer()->getName().c_str()); 
setOpTensorPtr(HKStLBswJlYYprZPPGQx->getOutput(0)); } } void 
MWElementwiseAffineLayerImpl::loadScaleAndOffset(const char* 
sCDdEyIOjXBVHhcakBhd, const char* jLmklYtHcmTxayQTpmRw){ 
CUDA_CALL(cudaMalloc((void**)&pxmnUEWGnfCxJNuDkXAo, 
sizeof(float)*rZyMIPooLjRiXLgSWDuw*rfIckzovFcUSejaEKNlY*qquNiJHQtfSLDMNCPIBJ)); 
CUDA_CALL(cudaMalloc((void**)&hTbrmZSPUORqMIprVbKk, 
sizeof(float)*hpOzCTZasBMYKoXVxMDZ*hqbKXLMjsDxRQqyJEgbg*hjXpmIaeqlKKSDDBSxtE));  
loadScale(sCDdEyIOjXBVHhcakBhd); loadOffset(jLmklYtHcmTxayQTpmRw); } void 
MWElementwiseAffineLayerImpl::setLayerProperties(){ WawamKKnqecNqBXIyHIl = 
getLayer()->getInputTensor(0)->getHeight(); WbTBQxsNsCURmwRhNTAD = 
getLayer()->getInputTensor(0)->getWidth(); WShIchjHSSzOoYZernlN = 
getLayer()->getInputTensor(0)->getChannels(); YmfPcXPXNFZDznkzKZrl = 
WawamKKnqecNqBXIyHIl*WbTBQxsNsCURmwRhNTAD; YMNbgnUYZspjMLjwcIOS = 
YmfPcXPXNFZDznkzKZrl*WShIchjHSSzOoYZernlN; YDoginwuwFxabuYCVqpT = 
getLayer()->getInputTensor(0)->getNumElements(); reGtUwUlPSwEenEBVIzH = 
rZyMIPooLjRiXLgSWDuw * rfIckzovFcUSejaEKNlY * qquNiJHQtfSLDMNCPIBJ; 
hqVFaqkobRNLQNgtbaai = hpOzCTZasBMYKoXVxMDZ * hqbKXLMjsDxRQqyJEgbg * 
hjXpmIaeqlKKSDDBSxtE; assert(reGtUwUlPSwEenEBVIzH <= YDoginwuwFxabuYCVqpT); 
assert(hqVFaqkobRNLQNgtbaai <= YDoginwuwFxabuYCVqpT); } int 
MWElementwiseAffineLayerImpl::pluginEnqueueImpl(const void* const* inputs, 
void** outputs) { long int uTUuLVVebDakbPjXOQwp = ((YDoginwuwFxabuYCVqpT + 31) / 32) 
* 32; long int vVyVzWKKaCvGClCSagOb = (uTUuLVVebDakbPjXOQwp < 1024) ? 
uTUuLVVebDakbPjXOQwp : 1024; long int OuTwywxKeMgznElXdjGp = 
(YDoginwuwFxabuYCVqpT + vVyVzWKKaCvGClCSagOb - 1) / 
vVyVzWKKaCvGClCSagOb; if (reGtUwUlPSwEenEBVIzH == 1) { 
scale_scalar_kernel<<<OuTwywxKeMgznElXdjGp, vVyVzWKKaCvGClCSagOb>>>( 
(float*)inputs[0],  (float*)outputs[0], pxmnUEWGnfCxJNuDkXAo, 
YDoginwuwFxabuYCVqpT); } else if (rZyMIPooLjRiXLgSWDuw == 1 && rfIckzovFcUSejaEKNlY 
== 1 && reGtUwUlPSwEenEBVIzH > 1) { 
scale_vector_kernel<<<OuTwywxKeMgznElXdjGp, vVyVzWKKaCvGClCSagOb>>>( 
(float*)inputs[0],  (float*)outputs[0], pxmnUEWGnfCxJNuDkXAo, 
YmfPcXPXNFZDznkzKZrl, YMNbgnUYZspjMLjwcIOS, 
YDoginwuwFxabuYCVqpT); } else if (YMNbgnUYZspjMLjwcIOS == 
reGtUwUlPSwEenEBVIzH) {  scale_tensor3d_kernel<<<OuTwywxKeMgznElXdjGp, 
vVyVzWKKaCvGClCSagOb>>>( (float*)inputs[0],  (float*)outputs[0], 
pxmnUEWGnfCxJNuDkXAo, WbTBQxsNsCURmwRhNTAD, WawamKKnqecNqBXIyHIl,  
YmfPcXPXNFZDznkzKZrl,  YMNbgnUYZspjMLjwcIOS, 
YDoginwuwFxabuYCVqpT); } else { 
scale_matrix2d_kernel<<<OuTwywxKeMgznElXdjGp, 
vVyVzWKKaCvGClCSagOb>>>( (float*)inputs[0],  (float*)outputs[0], 
pxmnUEWGnfCxJNuDkXAo, WbTBQxsNsCURmwRhNTAD,  YmfPcXPXNFZDznkzKZrl,  
YMNbgnUYZspjMLjwcIOS, YDoginwuwFxabuYCVqpT); } if (hqVFaqkobRNLQNgtbaai 
== 1) { offset_scalar_kernel<<<OuTwywxKeMgznElXdjGp, 
vVyVzWKKaCvGClCSagOb>>>( (float*)outputs[0],  (float*)outputs[0], 
hTbrmZSPUORqMIprVbKk, YDoginwuwFxabuYCVqpT, ZqQxEyCjEixByRZYMkbj, 
boZxhUbgnNUlJQXTuWwi, voqEJSkAwmNPuqzoiuom); } else if (hpOzCTZasBMYKoXVxMDZ 
== 1 && hqbKXLMjsDxRQqyJEgbg == 1 && hqVFaqkobRNLQNgtbaai > 1) { 
offset_vector_kernel<<<OuTwywxKeMgznElXdjGp, vVyVzWKKaCvGClCSagOb>>>( 
(float*)outputs[0],  (float*)outputs[0], hTbrmZSPUORqMIprVbKk, 
YmfPcXPXNFZDznkzKZrl, YMNbgnUYZspjMLjwcIOS, 
YDoginwuwFxabuYCVqpT, ZqQxEyCjEixByRZYMkbj, boZxhUbgnNUlJQXTuWwi, 
voqEJSkAwmNPuqzoiuom); } else if (YMNbgnUYZspjMLjwcIOS == 
hqVFaqkobRNLQNgtbaai) { offset_tensor3d_kernel<<<OuTwywxKeMgznElXdjGp, 
vVyVzWKKaCvGClCSagOb>>>( (float*)outputs[0],  (float*)outputs[0], 
hTbrmZSPUORqMIprVbKk, WbTBQxsNsCURmwRhNTAD, WawamKKnqecNqBXIyHIl, 
YmfPcXPXNFZDznkzKZrl, YMNbgnUYZspjMLjwcIOS, 
YDoginwuwFxabuYCVqpT, ZqQxEyCjEixByRZYMkbj, boZxhUbgnNUlJQXTuWwi, 
voqEJSkAwmNPuqzoiuom); } else { 
offset_matrix2d_kernel<<<OuTwywxKeMgznElXdjGp, 
vVyVzWKKaCvGClCSagOb>>>( (float*)outputs[0],  (float*)outputs[0], 
hTbrmZSPUORqMIprVbKk, WbTBQxsNsCURmwRhNTAD, YmfPcXPXNFZDznkzKZrl, 
YMNbgnUYZspjMLjwcIOS, YDoginwuwFxabuYCVqpT, ZqQxEyCjEixByRZYMkbj, 
boZxhUbgnNUlJQXTuWwi, voqEJSkAwmNPuqzoiuom); } return 0; } void 
MWElementwiseAffineLayerImpl::loadScale(const char* sCDdEyIOjXBVHhcakBhd) { 
FILE* SmibqCQPbtzycGEpwhpN = MWCNNLayer::openBinaryFile(sCDdEyIOjXBVHhcakBhd); 
assert(SmibqCQPbtzycGEpwhpN); long int eWYFXrUazhqiEIscccda = 
rZyMIPooLjRiXLgSWDuw*rfIckzovFcUSejaEKNlY*qquNiJHQtfSLDMNCPIBJ; rwPhFWHcKnJsClVtebGW 
= MALLOC_CALL(sizeof(float)*eWYFXrUazhqiEIscccda); call_fread(rwPhFWHcKnJsClVtebGW, 
sizeof(float), eWYFXrUazhqiEIscccda, SmibqCQPbtzycGEpwhpN, sCDdEyIOjXBVHhcakBhd); 
CUDA_CALL(cudaMemcpy(pxmnUEWGnfCxJNuDkXAo, rwPhFWHcKnJsClVtebGW, 
sizeof(float)*eWYFXrUazhqiEIscccda, cudaMemcpyHostToDevice)); fclose(SmibqCQPbtzycGEpwhpN);  
} void MWElementwiseAffineLayerImpl::loadOffset(const char* 
jLmklYtHcmTxayQTpmRw) { FILE* SmibqCQPbtzycGEpwhpN = 
MWCNNLayer::openBinaryFile(jLmklYtHcmTxayQTpmRw); assert(SmibqCQPbtzycGEpwhpN); long 
int eWYFXrUazhqiEIscccda = 
hpOzCTZasBMYKoXVxMDZ*hqbKXLMjsDxRQqyJEgbg*hjXpmIaeqlKKSDDBSxtE; 
ikTyjLTPRBkBRlLSyxXG = MALLOC_CALL(sizeof(float)*eWYFXrUazhqiEIscccda); 
call_fread(ikTyjLTPRBkBRlLSyxXG, sizeof(float), eWYFXrUazhqiEIscccda, SmibqCQPbtzycGEpwhpN, 
jLmklYtHcmTxayQTpmRw); CUDA_CALL(cudaMemcpy(hTbrmZSPUORqMIprVbKk, 
ikTyjLTPRBkBRlLSyxXG, sizeof(float)*eWYFXrUazhqiEIscccda, cudaMemcpyHostToDevice)); 
fclose(SmibqCQPbtzycGEpwhpN);  } void MWElementwiseAffineLayerImpl::cleanup() { if 
(pxmnUEWGnfCxJNuDkXAo) { CUDA_FREE_CALL(pxmnUEWGnfCxJNuDkXAo); } if (hTbrmZSPUORqMIprVbKk) 
{ CUDA_FREE_CALL(hTbrmZSPUORqMIprVbKk); } if (rwPhFWHcKnJsClVtebGW) 
free(rwPhFWHcKnJsClVtebGW); if (ikTyjLTPRBkBRlLSyxXG) 
free(ikTyjLTPRBkBRlLSyxXG); }