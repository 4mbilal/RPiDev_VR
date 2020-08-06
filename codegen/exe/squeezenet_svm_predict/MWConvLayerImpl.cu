#include <cstdlib>
#include <cassert>
#include <stdio.h>
#include <cassert>
#include <iostream>
#include "MWConvLayerImpl.hpp"
#include "MWConvLayer.hpp"
#include "MWTargetNetworkImpl.hpp"
#include "cnn_api.hpp"
 MWConvLayerImpl::MWConvLayerImpl(MWCNNLayer* layer, MWTargetNetworkImpl* 
ntwk_impl, int filt_H, int filt_W, int numGrps, int numChnls, int numFilts, int 
IGBjAMvMJXqrubGDtvyq, int ILoEtPdZOaUSKUyteZTa, int GtFSKuDmLreppbjSISoU, int 
GbHRuweETkejIMGyqHDI, int GmRRxuYauzGdhIlgciAT, int GrowsTaKrpHVUZdgZeJW, int 
DytNxKWQcYUHaYuuACXS, int EBrSnWuJobWBIFNZLSZN, const char* 
zdjrTcsHRcPcpoVFAMfT, const char* OWgntZrUmlZXHAsNObcq, int ) : 
MWCNNLayerImpl(layer, ntwk_impl) , EFRxTzGDLCOxVeZLDhRL(filt_H) , 
EGsHUnogBQpOwCZJYeUd(filt_W) , EiBytenrthqoQrTnOFaK(numGrps) , 
wDZXrivCbTUzkLKxiPUh(NULL) , NSzdekOvRhMhRCXdWsdY(NULL) , ConvLayerT(0) { MWConvLayer* 
convLayer = static_cast<MWConvLayer*>(getLayer()); MWTensor* ipTensor = 
convLayer->getInputTensor(0); MWTensor* opTensor = 
convLayer->getOutputTensor(0); wDZXrivCbTUzkLKxiPUh = 
(float*)calloc(ipTensor->getChannels() / EiBytenrthqoQrTnOFaK * 
opTensor->getChannels() * EFRxTzGDLCOxVeZLDhRL * EGsHUnogBQpOwCZJYeUd, 
sizeof(float)); NSzdekOvRhMhRCXdWsdY = (float*)calloc(opTensor->getChannels(), 
sizeof(float)); loadWeights(zdjrTcsHRcPcpoVFAMfT); 
loadBias(OWgntZrUmlZXHAsNObcq); filt_weights.values = wDZXrivCbTUzkLKxiPUh; 
filt_weights.count = ipTensor->getChannels() / EiBytenrthqoQrTnOFaK * 
opTensor->getChannels() * EFRxTzGDLCOxVeZLDhRL * EGsHUnogBQpOwCZJYeUd; 
filt_weights.type = DataType::kFLOAT; filt_bias.values = NSzdekOvRhMhRCXdWsdY; 
filt_bias.count = opTensor->getChannels(); filt_bias.type = DataType::kFLOAT; 
ITensor* prevLayerTensor = getInputITensor(0); if((GtFSKuDmLreppbjSISoU == 
GbHRuweETkejIMGyqHDI) && (GmRRxuYauzGdhIlgciAT == GrowsTaKrpHVUZdgZeJW)){  
ConvLayerT = dwFpvfypaTkJiYAULzFs->network->addConvolution( *prevLayerTensor, 
opTensor->getChannels(), DimsHW{EFRxTzGDLCOxVeZLDhRL, 
EGsHUnogBQpOwCZJYeUd}, filt_weights, filt_bias); 
ConvLayerT->setPadding(DimsHW{GtFSKuDmLreppbjSISoU ,GmRRxuYauzGdhIlgciAT}); } 
else{ IPaddingLayer* iPaddingLayer = dwFpvfypaTkJiYAULzFs->network->addPadding( 
*prevLayerTensor, DimsHW{GtFSKuDmLreppbjSISoU,GmRRxuYauzGdhIlgciAT}, 
DimsHW{GbHRuweETkejIMGyqHDI,GrowsTaKrpHVUZdgZeJW}); ITensor* 
FCgChnBQLAUXyoWzTlfg = iPaddingLayer->getOutput(0); ConvLayerT = 
dwFpvfypaTkJiYAULzFs->network->addConvolution( *FCgChnBQLAUXyoWzTlfg, 
opTensor->getChannels(), DimsHW{EFRxTzGDLCOxVeZLDhRL, 
EGsHUnogBQpOwCZJYeUd}, filt_weights, filt_bias);  } 
ConvLayerT->setDilation(DimsHW{DytNxKWQcYUHaYuuACXS, 
EBrSnWuJobWBIFNZLSZN}); ConvLayerT->setStride(DimsHW{IGBjAMvMJXqrubGDtvyq, 
ILoEtPdZOaUSKUyteZTa}); ConvLayerT->setNbGroups(EiBytenrthqoQrTnOFaK); 
ConvLayerT->setName(getLayer()->getName().c_str()); 
setOpTensorPtr(ConvLayerT->getOutput(0)); } void MWConvLayerImpl::cleanup() { 
free(wDZXrivCbTUzkLKxiPUh); free(NSzdekOvRhMhRCXdWsdY); } void 
MWConvLayerImpl::loadWeights(const char* RuGYRQXjIMQJrbgoRUxZ) { MWConvLayer* 
convLayer = static_cast<MWConvLayer*>(getLayer()); FILE* SmibqCQPbtzycGEpwhpN = 
MWCNNLayer::openBinaryFile(RuGYRQXjIMQJrbgoRUxZ); assert(SmibqCQPbtzycGEpwhpN); int 
eWYFXrUazhqiEIscccda = convLayer->getInputTensor()->getChannels() / 
EiBytenrthqoQrTnOFaK * convLayer->getOutputTensor()->getChannels() * 
EFRxTzGDLCOxVeZLDhRL * EGsHUnogBQpOwCZJYeUd; call_fread(wDZXrivCbTUzkLKxiPUh, 
sizeof(float), eWYFXrUazhqiEIscccda, SmibqCQPbtzycGEpwhpN, RuGYRQXjIMQJrbgoRUxZ); 
fclose(SmibqCQPbtzycGEpwhpN); } void MWConvLayerImpl::loadBias(const char* 
RuGYRQXjIMQJrbgoRUxZ) { MWConvLayer* convLayer = 
static_cast<MWConvLayer*>(getLayer()); FILE* SmibqCQPbtzycGEpwhpN = 
MWCNNLayer::openBinaryFile(RuGYRQXjIMQJrbgoRUxZ); assert(SmibqCQPbtzycGEpwhpN); int 
eWYFXrUazhqiEIscccda = convLayer->getOutputTensor()->getChannels(); 
call_fread(NSzdekOvRhMhRCXdWsdY, sizeof(float), eWYFXrUazhqiEIscccda, SmibqCQPbtzycGEpwhpN, 
RuGYRQXjIMQJrbgoRUxZ); fclose(SmibqCQPbtzycGEpwhpN); }