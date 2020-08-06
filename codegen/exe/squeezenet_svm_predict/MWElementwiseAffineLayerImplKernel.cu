#include "MWKernelHeaders.hpp"
#include <math.h>
#include <stdio.h>
 void __global__ __launch_bounds__(1024) scale_scalar_kernel(float* 
inputBuffer, float* outputBuffer, float* omxlPZbBePZdWaJOBUUG, long int 
YGiQICncmsGZkNUyiQyg) {  for (int idx = blockDim.x * blockIdx.x + threadIdx.x; 
idx < YGiQICncmsGZkNUyiQyg; idx += blockDim.x * gridDim.x) {  outputBuffer[idx] 
= omxlPZbBePZdWaJOBUUG[0]*inputBuffer[idx]; } } void __global__ 
__launch_bounds__(1024) scale_vector_kernel(float* inputBuffer, float* 
outputBuffer, float* omxlPZbBePZdWaJOBUUG, long int YNmJhGSUszJKxsodxiuV, 
long int YNDVziqpDddiXQKYZZhX, long int YGiQICncmsGZkNUyiQyg) {  for 
(int idx = blockDim.x * blockIdx.x + threadIdx.x; idx < YGiQICncmsGZkNUyiQyg; 
idx += blockDim.x * gridDim.x) { int dAGMlbhOYuZqhuDGCqih = 
idx/YNDVziqpDddiXQKYZZhX; long int EfvWctmlsWAPsxXgdKWf = 
idx-(YNDVziqpDddiXQKYZZhX*dAGMlbhOYuZqhuDGCqih); int KHjdvykTFbUxdfZTFbqy = 
static_cast<int>(EfvWctmlsWAPsxXgdKWf / YNmJhGSUszJKxsodxiuV); 
outputBuffer[idx] = omxlPZbBePZdWaJOBUUG[KHjdvykTFbUxdfZTFbqy]*inputBuffer[idx]; } } void 
__global__ __launch_bounds__(1024) scale_matrix2d_kernel(float* inputBuffer, 
float* outputBuffer, float* omxlPZbBePZdWaJOBUUG, long int YOWMnLKOMqAODXiVNoGy, long 
int YNmJhGSUszJKxsodxiuV, long int YNDVziqpDddiXQKYZZhX, long 
int YGiQICncmsGZkNUyiQyg) {  for (int idx = blockDim.x * blockIdx.x + 
threadIdx.x; idx < YGiQICncmsGZkNUyiQyg; idx += blockDim.x * gridDim.x) { int 
dAGMlbhOYuZqhuDGCqih = idx/YNDVziqpDddiXQKYZZhX; long int EfvWctmlsWAPsxXgdKWf 
= idx-(YNDVziqpDddiXQKYZZhX*dAGMlbhOYuZqhuDGCqih); int KHjdvykTFbUxdfZTFbqy = 
static_cast<int>(EfvWctmlsWAPsxXgdKWf / YNmJhGSUszJKxsodxiuV); long 
int EvebzoroiuKkIxwjkGnD = EfvWctmlsWAPsxXgdKWf - 
(YNmJhGSUszJKxsodxiuV*KHjdvykTFbUxdfZTFbqy); int vIWQzNvYZSuxmOTVDFhU = 
static_cast<int>(EvebzoroiuKkIxwjkGnD % YOWMnLKOMqAODXiVNoGy); int 
RAtlBpdedvgxUsgDTsch = static_cast<int>(EvebzoroiuKkIxwjkGnD / YOWMnLKOMqAODXiVNoGy); 
outputBuffer[idx] = 
omxlPZbBePZdWaJOBUUG[vIWQzNvYZSuxmOTVDFhU+YOWMnLKOMqAODXiVNoGy*RAtlBpdedvgxUsgDTsch]*inputBuffer[idx]; 
} } void __global__ __launch_bounds__(1024) scale_tensor3d_kernel(float* 
inputBuffer, float* outputBuffer, float* omxlPZbBePZdWaJOBUUG, long int 
YOWMnLKOMqAODXiVNoGy, long int YCSvyQZBWMDYQXHtyVai, long int 
YNmJhGSUszJKxsodxiuV, long int YNDVziqpDddiXQKYZZhX, long int 
YGiQICncmsGZkNUyiQyg) {  for (int idx = blockDim.x * blockIdx.x + threadIdx.x; 
idx < YGiQICncmsGZkNUyiQyg; idx += blockDim.x * gridDim.x) { int dAGMlbhOYuZqhuDGCqih = 
idx/YNDVziqpDddiXQKYZZhX; long int EfvWctmlsWAPsxXgdKWf = 
idx-(YNDVziqpDddiXQKYZZhX*dAGMlbhOYuZqhuDGCqih); int KHjdvykTFbUxdfZTFbqy = 
static_cast<int>(EfvWctmlsWAPsxXgdKWf / YNmJhGSUszJKxsodxiuV); long 
int EvebzoroiuKkIxwjkGnD = EfvWctmlsWAPsxXgdKWf - 
(YNmJhGSUszJKxsodxiuV*KHjdvykTFbUxdfZTFbqy); int vIWQzNvYZSuxmOTVDFhU = 
static_cast<int>(EvebzoroiuKkIxwjkGnD % YOWMnLKOMqAODXiVNoGy); int 
RAtlBpdedvgxUsgDTsch = static_cast<int>(EvebzoroiuKkIxwjkGnD / YOWMnLKOMqAODXiVNoGy); 
outputBuffer[idx] = 
omxlPZbBePZdWaJOBUUG[vIWQzNvYZSuxmOTVDFhU+YOWMnLKOMqAODXiVNoGy*(RAtlBpdedvgxUsgDTsch+YCSvyQZBWMDYQXHtyVai*KHjdvykTFbUxdfZTFbqy)]*inputBuffer[idx]; 
} }  void __global__ __launch_bounds__(1024) offset_scalar_kernel(float* 
inputBuffer, float* outputBuffer, float* gCYwEfkibolsgZAumsuW, long int 
YGiQICncmsGZkNUyiQyg, bool ZKjSVYDDjACizBkGbqBq, int bOrQjJTNlssnrexxbHdi, int 
unSXtdjDjpysqxmbIiPv) {  for (int idx = blockDim.x * blockIdx.x + threadIdx.x; 
idx < YGiQICncmsGZkNUyiQyg; idx += blockDim.x * gridDim.x) { outputBuffer[idx] 
= inputBuffer[idx] + gCYwEfkibolsgZAumsuW[0]; if (ZKjSVYDDjACizBkGbqBq){ 
outputBuffer[idx] = outputBuffer[idx] > unSXtdjDjpysqxmbIiPv ? 
unSXtdjDjpysqxmbIiPv : outputBuffer[idx]; outputBuffer[idx] = 
outputBuffer[idx] < bOrQjJTNlssnrexxbHdi ? bOrQjJTNlssnrexxbHdi : 
outputBuffer[idx]; } } } void __global__ __launch_bounds__(1024) 
offset_vector_kernel(float* inputBuffer, float* outputBuffer, float* 
gCYwEfkibolsgZAumsuW,  long int YNmJhGSUszJKxsodxiuV, long int 
YNDVziqpDddiXQKYZZhX, long int YGiQICncmsGZkNUyiQyg, bool 
ZKjSVYDDjACizBkGbqBq, int bOrQjJTNlssnrexxbHdi, int unSXtdjDjpysqxmbIiPv) {  
for (int idx = blockDim.x * blockIdx.x + threadIdx.x; idx < 
YGiQICncmsGZkNUyiQyg; idx += blockDim.x * gridDim.x) { int dAGMlbhOYuZqhuDGCqih = 
idx/YNDVziqpDddiXQKYZZhX; long int EfvWctmlsWAPsxXgdKWf = 
idx-(YNDVziqpDddiXQKYZZhX*dAGMlbhOYuZqhuDGCqih); int KHjdvykTFbUxdfZTFbqy = 
static_cast<int>(EfvWctmlsWAPsxXgdKWf / YNmJhGSUszJKxsodxiuV); 
outputBuffer[idx] = inputBuffer[idx] + gCYwEfkibolsgZAumsuW[KHjdvykTFbUxdfZTFbqy]; if 
(ZKjSVYDDjACizBkGbqBq){ outputBuffer[idx] = outputBuffer[idx] > 
unSXtdjDjpysqxmbIiPv ? unSXtdjDjpysqxmbIiPv : outputBuffer[idx]; 
outputBuffer[idx] = outputBuffer[idx] < bOrQjJTNlssnrexxbHdi ? 
bOrQjJTNlssnrexxbHdi : outputBuffer[idx]; } } } void __global__ 
__launch_bounds__(1024) offset_matrix2d_kernel(float* inputBuffer, float* 
outputBuffer, float* gCYwEfkibolsgZAumsuW, long int YOWMnLKOMqAODXiVNoGy, long int 
YNmJhGSUszJKxsodxiuV, long int YNDVziqpDddiXQKYZZhX, long int 
YGiQICncmsGZkNUyiQyg, bool ZKjSVYDDjACizBkGbqBq, int bOrQjJTNlssnrexxbHdi, int 
unSXtdjDjpysqxmbIiPv) {  for (int idx = blockDim.x * blockIdx.x + threadIdx.x; 
idx < YGiQICncmsGZkNUyiQyg; idx += blockDim.x * gridDim.x) { int dAGMlbhOYuZqhuDGCqih = 
idx/YNDVziqpDddiXQKYZZhX; long int EfvWctmlsWAPsxXgdKWf = 
idx-(YNDVziqpDddiXQKYZZhX*dAGMlbhOYuZqhuDGCqih); int KHjdvykTFbUxdfZTFbqy = 
static_cast<int>(EfvWctmlsWAPsxXgdKWf / YNmJhGSUszJKxsodxiuV); long 
int EvebzoroiuKkIxwjkGnD = EfvWctmlsWAPsxXgdKWf - 
(YNmJhGSUszJKxsodxiuV*KHjdvykTFbUxdfZTFbqy); int vIWQzNvYZSuxmOTVDFhU = 
static_cast<int>(EvebzoroiuKkIxwjkGnD % YOWMnLKOMqAODXiVNoGy); int 
RAtlBpdedvgxUsgDTsch = static_cast<int>(EvebzoroiuKkIxwjkGnD / YOWMnLKOMqAODXiVNoGy); 
outputBuffer[idx] = inputBuffer[idx] + 
gCYwEfkibolsgZAumsuW[vIWQzNvYZSuxmOTVDFhU+YOWMnLKOMqAODXiVNoGy*RAtlBpdedvgxUsgDTsch]; if 
(ZKjSVYDDjACizBkGbqBq){ outputBuffer[idx] = outputBuffer[idx] > 
unSXtdjDjpysqxmbIiPv ? unSXtdjDjpysqxmbIiPv : outputBuffer[idx]; 
outputBuffer[idx] = outputBuffer[idx] < bOrQjJTNlssnrexxbHdi ? 
bOrQjJTNlssnrexxbHdi : outputBuffer[idx]; } } } void __global__ 
__launch_bounds__(1024) offset_tensor3d_kernel(float* inputBuffer, float* 
outputBuffer, float* gCYwEfkibolsgZAumsuW,  long int YOWMnLKOMqAODXiVNoGy, long int 
YCSvyQZBWMDYQXHtyVai, long int YNmJhGSUszJKxsodxiuV, long int 
YNDVziqpDddiXQKYZZhX, long int YGiQICncmsGZkNUyiQyg, bool 
ZKjSVYDDjACizBkGbqBq, int bOrQjJTNlssnrexxbHdi, int unSXtdjDjpysqxmbIiPv) {  
for (int idx = blockDim.x * blockIdx.x + threadIdx.x; idx < 
YGiQICncmsGZkNUyiQyg; idx += blockDim.x * gridDim.x) { int dAGMlbhOYuZqhuDGCqih = 
idx/YNDVziqpDddiXQKYZZhX; long int EfvWctmlsWAPsxXgdKWf = 
idx-(YNDVziqpDddiXQKYZZhX*dAGMlbhOYuZqhuDGCqih); int KHjdvykTFbUxdfZTFbqy = 
static_cast<int>(EfvWctmlsWAPsxXgdKWf / YNmJhGSUszJKxsodxiuV); long 
int EvebzoroiuKkIxwjkGnD = EfvWctmlsWAPsxXgdKWf - 
(YNmJhGSUszJKxsodxiuV*KHjdvykTFbUxdfZTFbqy); int vIWQzNvYZSuxmOTVDFhU = 
static_cast<int>(EvebzoroiuKkIxwjkGnD % YOWMnLKOMqAODXiVNoGy); int 
RAtlBpdedvgxUsgDTsch = static_cast<int>(EvebzoroiuKkIxwjkGnD / YOWMnLKOMqAODXiVNoGy); 
outputBuffer[idx] = inputBuffer[idx] + 
gCYwEfkibolsgZAumsuW[vIWQzNvYZSuxmOTVDFhU+YOWMnLKOMqAODXiVNoGy*(RAtlBpdedvgxUsgDTsch+YCSvyQZBWMDYQXHtyVai*KHjdvykTFbUxdfZTFbqy)]; 
if (ZKjSVYDDjACizBkGbqBq){ outputBuffer[idx] = outputBuffer[idx] > 
unSXtdjDjpysqxmbIiPv ? unSXtdjDjpysqxmbIiPv : outputBuffer[idx]; 
outputBuffer[idx] = outputBuffer[idx] < bOrQjJTNlssnrexxbHdi ? 
bOrQjJTNlssnrexxbHdi : outputBuffer[idx]; } } } 