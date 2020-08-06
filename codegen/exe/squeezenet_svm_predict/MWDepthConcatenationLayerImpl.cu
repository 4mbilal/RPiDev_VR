#include "MWDepthConcatenationLayerImpl.hpp"
#include "MWDepthConcatenationLayer.hpp"
#include "MWTargetNetworkImpl.hpp"
#include <vector> 
 MWDepthConcatenationLayerImpl::MWDepthConcatenationLayerImpl(MWCNNLayer* 
layer, MWTargetNetworkImpl* ntwk_impl, int outbufIdx) : MWCNNLayerImpl(layer, 
ntwk_impl){ std::vector<ITensor*> VwwbAYoMgONCiWkbFFyF; for (int i = 0; i < 
layer->getNumInputs(); i++){ 
VwwbAYoMgONCiWkbFFyF.push_back(getInputITensor(i)); } 
CZiiieLxAFTgpdhdjNUA = 
dwFpvfypaTkJiYAULzFs->network->addConcatenation(&VwwbAYoMgONCiWkbFFyF[0], 
static_cast<MWDepthConcatenationLayer*>(getLayer())->getNumInputs()); 
CZiiieLxAFTgpdhdjNUA->setName(getLayer()->getName().c_str()); 
setOpTensorPtr(CZiiieLxAFTgpdhdjNUA->getOutput(0)); return; } MWDepthConcatenationLayerImpl::~MWDepthConcatenationLayerImpl(){}