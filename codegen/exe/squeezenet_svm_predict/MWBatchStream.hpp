#ifndef MWBATCH_STREAM_HPP
#define MWBATCH_STREAM_HPP

#include <vector>
#include <assert.h>
#include <algorithm>
#include "NvInfer.h"

/* This file contains the classes ,used to parse the claibration data set .
 * Parsed data is used by TensorRT for creating the Calibration Table 
 * which is then used for int8 execution*/

extern void getValidDataPath(const char* fileName, char *validDatapath);
extern void CHECK(cudaError_t status);
extern std::string gvalidDatapath;
using namespace nvinfer1;


class BatchStream{

  public:

    BatchStream(int maxBatches)
        : mNumCalibrationBatches(maxBatches)
        , mCalibrationBatchCount(0)
    {
	char filename[500],filename1[500];

        // start with first batch
        sprintf(filename," |>targetdir<|/tensorrt/batch0");

	getValidDataPath(filename,filename1);
        FILE *file = fopen(filename1,"rb");
        if (file == NULL) {
            printf("Unable to open file\n");
            exit(1);
        }   

        int d[4];
        if (fread(d, sizeof(int), 4, file) != 4) {
            throw std::runtime_error("Unexpected number of bytes read from " + std::string(filename1));
        }

        fclose(file);
        DimsNCHW mDims = DimsNCHW{ d[0], d[1], d[2], d[3] };
        mBatchSize = mDims.n();        
        mImageSize = mDims.c()*mDims.h()*mDims.w();

        // allocate memory for data buffer
        mBatchData = new float[mBatchSize * mImageSize];
             
    }

    ~BatchStream(){

        if(mBatchData){
            delete[] mBatchData;
            mBatchData = 0;
        }
    }

    float* getBatchPtr(){
        return mBatchData;
    }

    int getBatchSize() const{

        return mBatchSize;
    }        

    bool next(){

        if (mCalibrationBatchCount == mNumCalibrationBatches)
            return false;
        else
            return true;
    }

    float* getBatch()
    {

        char filename[500],filename1[500];
       
        sprintf(filename," |>targetdir<|/tensorrt/batch");
        
        std::string inputFileName = filename + std::to_string(mCalibrationBatchCount);
        
        getValidDataPath(inputFileName.c_str(),filename1);

        FILE *file = fopen(filename1,"rb");
        if (file == NULL) {
            printf("Unable to open file %s \n", filename1);
            exit(1);
        }   

        if (mCalibrationBatchCount == 0){
            int status = fseek(file, sizeof(int) * 4, SEEK_SET);
            assert(status == 0);
        }

	size_t readInputCount = fread(getBatchPtr(), sizeof(float), mBatchSize*mImageSize, file);     
        fclose(file);

        mCalibrationBatchCount++;

        return getBatchPtr();
        
  
    }

    long int getImageSize(){
        return mImageSize;
    }


  private:
    int mBatchSize;
    int mCalibrationBatchCount;
    int mNumCalibrationBatches;
    long int mImageSize;
    float* mBatchData;
};


#if (NV_TENSORRT_MAJOR >= 5 && NV_TENSORRT_MINOR >= 1)
class Int8EntropyCalibrator : public IInt8EntropyCalibrator2 {    
#else
class Int8EntropyCalibrator : public IInt8EntropyCalibrator {
#endif        
  public:
    Int8EntropyCalibrator(BatchStream& stream, int firstBatch, bool readCache = true)
        : mStream(stream)
        , mReadCache(readCache)
    {
        mInputCount = mStream.getBatchSize() *mStream.getImageSize();
        CHECK(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
    }

    virtual ~Int8EntropyCalibrator() {
	CHECK(cudaFree(mDeviceInput));
    }

    int getBatchSize() const override {
        return mStream.getBatchSize();
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) override
    {
        if (!mStream.next()) {
            return false;
        }

        CHECK(cudaMemcpy(mDeviceInput, mStream.getBatch(), mInputCount * sizeof(float), cudaMemcpyHostToDevice));
        assert(!strcmp(names[0], "data"));
        bindings[0] = mDeviceInput;
        return true;
    }

    const void* readCalibrationCache(size_t& length) override {
        mCalibrationCache.clear();

        gvalidDatapath.append("/");
        
        gvalidDatapath.append("CalibrationTable");
        std::ifstream input(gvalidDatapath.c_str(), std::ios::binary);
        input >> std::noskipws;
        if (mReadCache && input.good()) {
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
                      std::back_inserter(mCalibrationCache));
        }

        length = mCalibrationCache.size();
        return length ? &mCalibrationCache[0] : nullptr;
    }

    void writeCalibrationCache(const void* cache, size_t length) override {

        gvalidDatapath.append("/");

        gvalidDatapath.append("CalibrationTable");
        std::ofstream output(gvalidDatapath.c_str(), std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }

  private:
    BatchStream& mStream;
    bool mReadCache{ true };

    size_t mInputCount;
    void* mDeviceInput{ nullptr };
    std::vector<char> mCalibrationCache;
};
#endif
