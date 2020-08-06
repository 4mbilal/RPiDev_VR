#include "MWTargetNetworkImpl.hpp"
#include "cnn_api.hpp"
#include "MWCNNLayerImpl.hpp"
#include <math.h>
#include <iostream>
#include <cassert>
#include <fstream>
#if INT8_ENABLED
#include <fstream>
#include <iterator>
#include "MWBatchStream.hpp"
#define XSTR(x) #x
#define STR(x) XSTR(x)
#endif
 using namespace nvinfer1; using namespace nvcaffeparser1; void 
CHECK(cudaError_t status) { if (status != 0) { std::cout << "Cuda failure: " << 
status; abort(); } } class Logger : public ILogger { void log(Severity 
severity, const char* msg) override { if (severity != Severity::kINFO){ 
std::cout << msg << std::endl; } if (MWCNNLayer::isDebuggingEnabled()){  if 
(severity == Severity::kINFO){ std::ofstream logfile; 
logfile.open("MW_TensorRT_log.txt" , std::ofstream::out | std::ofstream::app); 
logfile << msg <<"\n"; logfile.close(); } } } }; static Logger gLogger;
#if INT8_ENABLED
 std::string getFilePath(std:: string fileS, std::string &path) { char* 
usrDataPath = NULL; usrDataPath = getenv("USER_DL_DATA_PATH"); if(usrDataPath 
!= NULL) { path = usrDataPath; } else { path = STR(MW_DL_DATA_PATH); } path = 
path + "/tensorrt"; size_t fNamePos = fileS.find_last_of("/\\"); if(fNamePos != 
std::string::npos) { std::string fileN(fileS.substr(fNamePos)); fileS = path + 
fileN; } else { fileS = path + fileS; } return fileS; } std::string 
gvalidDatapath;  void getValidDataPath(const char* fileName, char 
*validDatapath) { FILE* fp = fopen(fileName, "rb"); std::string 
fileS(fileName); if (!fp) {
#ifdef MW_DL_DATA_PATH
 std::string path; fileS = getFilePath(fileS,path); fp = fopen(fileS.c_str(), 
"rb"); if(fp != NULL) { fclose(fp); gvalidDatapath = path; 
strcpy(validDatapath,fileS.c_str()); } else { strcpy(validDatapath,fileName); }
#else
 size_t pos = 0;
#if defined(_WIN32) || defined(_WIN64)
 char delim_unix[] = "/"; char delim_win[] = "\\"; while(((pos = 
fileS.find(delim_unix)) != std::string::npos) || ((pos = fileS.find(delim_win)) 
!= std::string::npos))
#else
 char delim_unix[] = "/"; while((pos = fileS.find(delim_unix)) != std::string::npos)
#endif
 { if (pos == (fileS.size() - 1)) { fileS = ""; break; } fileS = 
fileS.substr(pos+1); fp = fopen(fileS.c_str(), "rb"); if(fp != NULL) { 
fclose(fp); strcpy(validDatapath, fileS.c_str()); gvalidDatapath = 
fileS.substr(0,fileS.find_last_of("/\\")); break; } else{ strcpy(validDatapath, 
fileName); } }
#endif
 } else { fclose(fp); strcpy(validDatapath, fileName); gvalidDatapath 
=validDatapath; gvalidDatapath = 
gvalidDatapath.substr(0,gvalidDatapath.find_last_of("/\\")); } }
#endif
 void MWTargetNetworkImpl::setBatchSize(int aBatchSize){ batchSize = 
aBatchSize; } void MWTargetNetworkImpl::setIsSequenceNetwork(bool 
aIsSequenceNetwork){ isSequenceNetwork = aIsSequenceNetwork; } void 
MWTargetNetworkImpl::doInference(int batchSize) { const ICudaEngine& engine = 
context->getEngine(); cudaStream_t stream; CHECK(cudaStreamCreate(&stream)); if 
(this->isSequenceNetwork){ context->enqueue(1, m_buffers, stream, nullptr); } 
else{ context->enqueue(batchSize, m_buffers, stream, nullptr); } 
cudaStreamSynchronize(stream); cudaStreamDestroy(stream); } 
MWTargetNetworkImpl::MWTargetNetworkImpl() : network(0) , builder(0) , 
engine(0) , context(0) , PuqtLkzghXGwKBHOGAzf(0) , m_buffers(0) { } void 
MWTargetNetworkImpl::preSetup() { PuqtLkzghXGwKBHOGAzf = new cudnnHandle_t; 
cudnnCreate(PuqtLkzghXGwKBHOGAzf); builder = createInferBuilder(gLogger); } 
void MWTargetNetworkImpl::allocate(int, int) { network = 
builder->createNetwork(); } void MWTargetNetworkImpl::postSetup(MWCNNLayer* 
layers[], int numLayers, int layerIdxs[], int portIdxs[], int numOuts) { 
markOutputs(layers, layerIdxs, numOuts); std::map<int, std::pair<float*, 
std::string> > buffers; setupBuffers(layers, layerIdxs, portIdxs, numOuts, buffers);
#if INT8_ENABLED
 bool useINT8 = builder->platformHasFastInt8(); if(!useINT8){ char buffer[100]; int numElem = sprintf(buffer,"#### INT8 mode is not supported on GPU available on the current machine! ####\n"); throw std::runtime_error(buffer); } else{ builder->setInt8Mode(1); } int trainBatchCount=0;  while(1) { char filename[500]; char filename1[500]; sprintf(filename,"|>targetdir<|/tensorrt/batch%d",trainBatchCount++); getValidDataPath(filename,filename1); FILE *fp = fopen(filename1,"rb"); if(fp==NULL) { trainBatchCount-=1; break; } fclose(fp); } BatchStream calibrationStream(trainBatchCount); Int8EntropyCalibrator calibrator(calibrationStream, 0); builder->setAverageFindIterations(1); builder->setMinFindIterations(1); builder->setDebugSync(true); builder->setInt8Calibrator(&calibrator);
#endif
#if FP16_ENABLED
 bool useFp16 = builder->platformHasFastFp16(); if(useFp16){ builder->setFp16Mode(1); } else{ printf("#### FP16 mode is not supported on GPU available on the current machine. Falling back to FP32 ####\n"); }
#endif
 builder->setMaxBatchSize(batchSize); unsigned int wsize = 1 << 30; 
builder->setMaxWorkspaceSize(wsize); engine = 
builder->buildCudaEngine(*network); cudaError_t err = cudaGetLastError(); if 
(err != cudaSuccess) { builder->setMaxWorkspaceSize(1 << 26);  engine = 
builder->buildCudaEngine(*network);  cudaError_t err = cudaGetLastError(); if 
(err != cudaSuccess) { builder = 0; engine = 0; network = 0; CUDA_CALL(err); } 
}  context = engine->createExecutionContext();  m_buffers = (void**) new 
float*[buffers.size()]; for (std::map<int, std::pair<float*, std::string> 
>::iterator it = buffers.begin(); it != buffers.end(); ++it) { int 
binding_index = engine->getBindingIndex((it->second.second).c_str()); 
m_buffers[binding_index] = it->second.first; } network->destroy(); } void 
MWTargetNetworkImpl::markOutputs(MWCNNLayer* layers[], int layerIdxs[], int 
numOuts){ for (int k = 0; k < numOuts; k++) { int layerIdx = layerIdxs[k]; 
MWCNNLayer* layer = layers[layerIdx]; ITensor* itensor = 
MWCNNLayerImpl::getITensor(layer->getOutputTensor(0)); char layerIdxStr[20]; 
sprintf(layerIdxStr, "output%d", layerIdx); itensor->setName(layerIdxStr); 
network->markOutput(*itensor); } } void 
MWTargetNetworkImpl::setupBuffers(MWCNNLayer* layers[], int layerIdxs[], int 
portIdxs[], int numOuts, std::map<int, std::pair<float*, std::string> > & 
buffers) { float* buffer = getBuffer(layers[0], 0, 0); auto inputITensor = 
MWCNNLayerImpl::getITensor(layers[0]->getOutputTensor(0)); buffers[0] = 
std::make_pair(buffer, std::string(inputITensor->getName())); for(int k = 0; k 
< numOuts; k++) { int layerIdx = layerIdxs[k]; MWCNNLayer* layer = 
layers[layerIdx]; ITensor* itensor = 
MWCNNLayerImpl::getITensor(layer->getOutputTensor(0)); float* buffer = 
getBuffer(layer, 0, portIdxs[k]); buffers[k+1] = std::make_pair(buffer, 
std::string(itensor->getName())); } } float* 
MWTargetNetworkImpl::getBuffer(MWCNNLayer* layer, int layerIdx, int portIdx) { 
MWTensor* opTensor = layer->getOutputTensor(portIdx); float* data = 
opTensor->getData<float>(); if (!data) { CUDA_CALL(cudaMalloc((void**)&data, 
sizeof(float) * opTensor->getNumElements())); } opTensor->setData(data); return 
data; } cudnnHandle_t* MWTargetNetworkImpl::getCudnnHandle() { return 
PuqtLkzghXGwKBHOGAzf; } void MWTargetNetworkImpl::deallocate() { if 
(m_buffers) { delete[] m_buffers; m_buffers = 0; } if (cudaFree(0) != 
cudaErrorCudartUnloading) { if (context) { context->destroy(); context = 0; } 
if (engine) { engine->destroy(); engine = 0; } } } void 
MWTargetNetworkImpl::cleanup() { if (builder) { builder->destroy(); builder = 
0; } if (PuqtLkzghXGwKBHOGAzf) { cudnnDestroy(*PuqtLkzghXGwKBHOGAzf); 
delete PuqtLkzghXGwKBHOGAzf; PuqtLkzghXGwKBHOGAzf = 0; } } 
MWTargetNetworkImpl::~MWTargetNetworkImpl() { }