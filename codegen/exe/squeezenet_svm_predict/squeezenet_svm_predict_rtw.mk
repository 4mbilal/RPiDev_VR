###########################################################################
## Makefile generated for MATLAB file/project 'squeezenet_svm_predict'. 
## 
## Makefile     : squeezenet_svm_predict_rtw.mk
## Generated on : Thu Aug 06 04:34:00 2020
## MATLAB Coder version: 5.0 (R2020a)
## 
## Build Info:
## 
## Final product: $(MATLAB_WORKSPACE)/D/RnD/Frameworks/Matlab/ML/CNN/AI_NEOM/RemoteReality/squeezenet_svm_predict.elf
## Product type : executable
## 
###########################################################################

###########################################################################
## MACROS
###########################################################################

# Macro Descriptions:
# PRODUCT_NAME            Name of the system to build
# MAKEFILE                Name of this makefile

PRODUCT_NAME              = squeezenet_svm_predict
MAKEFILE                  = squeezenet_svm_predict_rtw.mk
MATLAB_ROOT               = $(MATLAB_WORKSPACE)/C/Program_Files/Polyspace/R2020a
MATLAB_BIN                = $(MATLAB_WORKSPACE)/C/Program_Files/Polyspace/R2020a/bin
MATLAB_ARCH_BIN           = $(MATLAB_BIN)/win64
MASTER_ANCHOR_DIR         = 
START_DIR                 = $(MATLAB_WORKSPACE)/D/RnD/Frameworks/Matlab/ML/CNN/AI_NEOM/RemoteReality/codegen/exe/squeezenet_svm_predict
TGT_FCN_LIB               = ISO_C++
SOLVER_OBJ                = 
CLASSIC_INTERFACE         = 0
MODEL_HAS_DYNAMICALLY_LOADED_SFCNS = 
RELATIVE_PATH_TO_ANCHOR   = .
C_STANDARD_OPTS           = 
CPP_STANDARD_OPTS         = 

###########################################################################
## TOOLCHAIN SPECIFICATIONS
###########################################################################

# Toolchain Name:          NVCC for NVIDIA Embedded Processors
# Supported Version(s):    
# ToolchainInfo Version:   2020a
# Specification Revision:  1.0
# 

#-----------
# MACROS
#-----------

CCOUTPUTFLAG = --output_file=
LDOUTPUTFLAG = --output_file=

TOOLCHAIN_SRCS = 
TOOLCHAIN_INCS = 
TOOLCHAIN_LIBS = -lm -lm

#------------------------
# BUILD TOOL COMMANDS
#------------------------

# C Compiler: NVCC for NVIDIA Embedded Processors1.0 NVIDIA CUDA C Compiler Driver
CC = nvcc

# Linker: NVCC for NVIDIA Embedded Processors1.0 NVIDIA CUDA C Linker
LD = nvcc

# C++ Compiler: NVCC for NVIDIA Embedded Processors1.0 NVIDIA CUDA C++ Compiler Driver
CPP = nvcc

# C++ Linker: NVCC for NVIDIA Embedded Processors1.0 NVIDIA CUDA C++ Linker
CPP_LD = nvcc

# Archiver: NVCC for NVIDIA Embedded Processors1.0 Archiver
AR = ar

# MEX Tool: MEX Tool
MEX_PATH = $(MATLAB_ARCH_BIN)
MEX = $(MEX_PATH)/mex

# Download: Download
DOWNLOAD =

# Execute: Execute
EXECUTE = $(PRODUCT)

# Builder: Make Tool
MAKE = make


#-------------------------
# Directives/Utilities
#-------------------------

CDEBUG              = -g -G
C_OUTPUT_FLAG       = -o
LDDEBUG             = -g -G
OUTPUT_FLAG         = -o
CPPDEBUG            = -g -G
CPP_OUTPUT_FLAG     = -o
CPPLDDEBUG          = -g -G
OUTPUT_FLAG         = -o
ARDEBUG             =
STATICLIB_OUTPUT_FLAG =
MEX_DEBUG           = -g
RM                  =
ECHO                = echo
MV                  =
RUN                 =

#--------------------------------------
# "Faster Runs" Build Configuration
#--------------------------------------

ARFLAGS              = -ruvs
CFLAGS               = -rdc=true -Xcudafe "--diag_suppress=unsigned_compare_with_zero" \
                       -c \
                       -Xcompiler -MMD,-MP \
                       -O2
CPPFLAGS             = -rdc=true -Xcudafe "--diag_suppress=unsigned_compare_with_zero" \
                       -c \
                       -Xcompiler -MMD,-MP \
                       -O2
CPP_LDFLAGS          = -lm -lrt -ldl \
                       -Xlinker -rpath,/usr/lib32 -Xnvlink -w -lcudart -lcuda -Wno-deprecated-gpu-targets
CPP_SHAREDLIB_LDFLAGS  = -shared  \
                         -lm -lrt -ldl \
                         -Xlinker -rpath,/usr/lib32 -Xnvlink -w -lcudart -lcuda -Wno-deprecated-gpu-targets
DOWNLOAD_FLAGS       =
EXECUTE_FLAGS        =
LDFLAGS              = -lm -lrt -ldl \
                       -Xlinker -rpath,/usr/lib32 -Xnvlink -w -lcudart -lcuda -Wno-deprecated-gpu-targets
MEX_CPPFLAGS         =
MEX_CPPLDFLAGS       =
MEX_CFLAGS           =
MEX_LDFLAGS          =
MAKE_FLAGS           = -f $(MAKEFILE)
SHAREDLIB_LDFLAGS    = -shared  \
                       -lm -lrt -ldl \
                       -Xlinker -rpath,/usr/lib32 -Xnvlink -w -lcudart -lcuda -Wno-deprecated-gpu-targets



###########################################################################
## OUTPUT INFO
###########################################################################

PRODUCT = $(MATLAB_WORKSPACE)/D/RnD/Frameworks/Matlab/ML/CNN/AI_NEOM/RemoteReality/squeezenet_svm_predict.elf
PRODUCT_TYPE = "executable"
BUILD_TYPE = "Executable"

###########################################################################
## INCLUDE PATHS
###########################################################################

INCLUDES_BUILDINFO = -I$(START_DIR) -I$(MATLAB_WORKSPACE)/D/RnD/Frameworks/Matlab/ML/CNN/AI_NEOM/RemoteReality -I$(MATLAB_WORKSPACE)/C/ProgramData/MATLAB/SupportPackages/R2020a/toolbox/target/supportpackages/nvidia/include -I$(MATLAB_ROOT)/extern/include

INCLUDES = $(INCLUDES_BUILDINFO)

###########################################################################
## DEFINES
###########################################################################

DEFINES_ = -DMW_CUDA_ARCH=350 -D__MW_TARGET_USE_HARDWARE_RESOURCES_H__ -DARM_PROJECT -D_USE_TARGET_UDP_ -D_RUNONTARGETHARDWARE_BUILD_ -DSTACK_SIZE=200000 -DMODEL=squeezenet_svm_predict -DMW_DL_DATA_PATH="$(START_DIR)" -DMW_SCHED_OTHER=1
DEFINES_CUSTOM = 
DEFINES_SKIPFORSIL = -DARM_PROJECT -D_USE_TARGET_UDP_ -D_RUNONTARGETHARDWARE_BUILD_ -DSTACK_SIZE=200000
DEFINES_STANDARD = -DMODEL=squeezenet_svm_predict

DEFINES = $(DEFINES_) $(DEFINES_CUSTOM) $(DEFINES_SKIPFORSIL) $(DEFINES_STANDARD)

###########################################################################
## SOURCE FILES
###########################################################################

SRCS = $(MATLAB_WORKSPACE)/MWConvLayer.cpp $(MATLAB_WORKSPACE)/MWDepthConcatenationLayer.cpp $(MATLAB_WORKSPACE)/MWElementwiseAffineLayer.cpp $(MATLAB_WORKSPACE)/cnn_api.cpp $(MATLAB_WORKSPACE)/MWElementwiseAffineLayerImplKernel.cu $(MATLAB_WORKSPACE)/MWCNNLayerImpl.cu $(MATLAB_WORKSPACE)/MWConvLayerImpl.cu $(MATLAB_WORKSPACE)/MWDepthConcatenationLayerImpl.cu $(MATLAB_WORKSPACE)/MWElementwiseAffineLayerImpl.cu $(MATLAB_WORKSPACE)/MWTargetNetworkImpl.cu $(START_DIR)/rt_nonfinite.cpp $(START_DIR)/rtGetNaN.cpp $(START_DIR)/rtGetInf.cpp $(MATLAB_WORKSPACE)/D/RnD/Frameworks/Matlab/ML/CNN/AI_NEOM/RemoteReality/main_rr.cpp $(MATLAB_WORKSPACE)/D/RnD/Frameworks/Matlab/ML/CNN/AI_NEOM/RemoteReality/MJPEGWriter.cpp $(START_DIR)/squeezenet_svm_predict_data.cu $(START_DIR)/squeezenet_svm_predict_initialize.cu $(START_DIR)/squeezenet_svm_predict_terminate.cu $(START_DIR)/squeezenet_svm_predict.cu $(START_DIR)/DeepLearningNetwork.cu $(START_DIR)/activations.cu $(START_DIR)/CompactClassificationECOC.cu $(MATLAB_WORKSPACE)/MWCudaDimUtility.cu

ALL_SRCS = $(SRCS)

###########################################################################
## OBJECTS
###########################################################################

OBJS = MWConvLayer.o MWDepthConcatenationLayer.o MWElementwiseAffineLayer.o cnn_api.o MWElementwiseAffineLayerImplKernel.o MWCNNLayerImpl.o MWConvLayerImpl.o MWDepthConcatenationLayerImpl.o MWElementwiseAffineLayerImpl.o MWTargetNetworkImpl.o rt_nonfinite.o rtGetNaN.o rtGetInf.o main_rr.o MJPEGWriter.o squeezenet_svm_predict_data.o squeezenet_svm_predict_initialize.o squeezenet_svm_predict_terminate.o squeezenet_svm_predict.o DeepLearningNetwork.o activations.o CompactClassificationECOC.o MWCudaDimUtility.o

ALL_OBJS = $(OBJS)

###########################################################################
## PREBUILT OBJECT FILES
###########################################################################

PREBUILT_OBJS = 

###########################################################################
## LIBRARIES
###########################################################################

LIBS = 

###########################################################################
## SYSTEM LIBRARIES
###########################################################################

SYSTEM_LIBS =  -lm -lstdc++ -lcufft -lcublas -lcusolver

###########################################################################
## ADDITIONAL TOOLCHAIN FLAGS
###########################################################################

#---------------
# C Compiler
#---------------

CFLAGS_ = `pkg-config --cflags --libs opencv4` -lpthread
CFLAGS_CU_OPTS = -std=c++11 -Xcompiler -std=c++11 -arch sm_35 
CFLAGS_BASIC = $(DEFINES) $(INCLUDES)

CFLAGS += $(CFLAGS_) $(CFLAGS_CU_OPTS) $(CFLAGS_BASIC)

#-----------------
# C++ Compiler
#-----------------

CPPFLAGS_ = `pkg-config --cflags --libs opencv4` -lpthread
CPPFLAGS_CU_OPTS = -std=c++11 -Xcompiler -std=c++11 -arch sm_35 
CPPFLAGS_BASIC = $(DEFINES) $(INCLUDES)

CPPFLAGS += $(CPPFLAGS_) $(CPPFLAGS_CU_OPTS) $(CPPFLAGS_BASIC)

#---------------
# C++ Linker
#---------------

CPP_LDFLAGS_ = `pkg-config --cflags --libs opencv4` -lpthread -lcudnn -lnvinfer_plugin -lnvinfer -lcudart -arch sm_35 

CPP_LDFLAGS += $(CPP_LDFLAGS_)

#------------------------------
# C++ Shared Library Linker
#------------------------------

CPP_SHAREDLIB_LDFLAGS_ = `pkg-config --cflags --libs opencv4` -lpthread -lcudnn -lnvinfer_plugin -lnvinfer -lcudart -arch sm_35 

CPP_SHAREDLIB_LDFLAGS += $(CPP_SHAREDLIB_LDFLAGS_)

#-----------
# Linker
#-----------

LDFLAGS_ = `pkg-config --cflags --libs opencv4` -lpthread -lcudnn -lnvinfer_plugin -lnvinfer -lcudart -arch sm_35 

LDFLAGS += $(LDFLAGS_)

#--------------------------
# Shared Library Linker
#--------------------------

SHAREDLIB_LDFLAGS_ = `pkg-config --cflags --libs opencv4` -lpthread -lcudnn -lnvinfer_plugin -lnvinfer -lcudart -arch sm_35 

SHAREDLIB_LDFLAGS += $(SHAREDLIB_LDFLAGS_)

###########################################################################
## INLINED COMMANDS
###########################################################################


DERIVED_SRCS = $(subst .o,.dep,$(OBJS))

build:

%.dep:



-include codertarget_assembly_flags.mk
-include *.dep


###########################################################################
## PHONY TARGETS
###########################################################################

.PHONY : all build buildobj clean info prebuild download execute


all : build
	echo "### Successfully generated all binary outputs."


build : prebuild $(PRODUCT)


buildobj : prebuild $(OBJS) $(PREBUILT_OBJS)
	echo "### Successfully generated all binary outputs."


prebuild : 


download : $(PRODUCT)


execute : download
	echo "### Invoking postbuild tool "Execute" ..."
	$(EXECUTE) $(EXECUTE_FLAGS)
	echo "### Done invoking postbuild tool."


###########################################################################
## FINAL TARGET
###########################################################################

#-------------------------------------------
# Create a standalone executable            
#-------------------------------------------

$(PRODUCT) : $(OBJS) $(PREBUILT_OBJS)
	echo "### Creating standalone executable "$(PRODUCT)" ..."
	$(CPP_LD) $(CPP_LDFLAGS) -o $(PRODUCT) $(OBJS) $(SYSTEM_LIBS) $(TOOLCHAIN_LIBS)
	echo "### Created: $(PRODUCT)"


###########################################################################
## INTERMEDIATE TARGETS
###########################################################################

#---------------------
# SOURCE-TO-OBJECT
#---------------------

%.o : %.c
	$(CC) $(CFLAGS) -o $@ $<


%.o : %.c
	$(CC) $(CFLAGS) -o $@ $<


%.o : %.cpp
	$(CPP) $(CPPFLAGS) -o $@ $<


%.o : %.cu
	$(CPP) $(CPPFLAGS) -o $@ $<


%.o : $(RELATIVE_PATH_TO_ANCHOR)/%.c
	$(CC) $(CFLAGS) -o $@ $<


%.o : $(RELATIVE_PATH_TO_ANCHOR)/%.c
	$(CC) $(CFLAGS) -o $@ $<


%.o : $(RELATIVE_PATH_TO_ANCHOR)/%.cpp
	$(CPP) $(CPPFLAGS) -o $@ $<


%.o : $(RELATIVE_PATH_TO_ANCHOR)/%.cu
	$(CPP) $(CPPFLAGS) -o $@ $<


%.o : $(START_DIR)/%.c
	$(CC) $(CFLAGS) -o $@ $<


%.o : $(START_DIR)/%.c
	$(CC) $(CFLAGS) -o $@ $<


%.o : $(START_DIR)/%.cpp
	$(CPP) $(CPPFLAGS) -o $@ $<


%.o : $(START_DIR)/%.cu
	$(CPP) $(CPPFLAGS) -o $@ $<


%.o : $(MATLAB_WORKSPACE)/D/RnD/Frameworks/Matlab/ML/CNN/AI_NEOM/RemoteReality/%.c
	$(CC) $(CFLAGS) -o $@ $<


%.o : $(MATLAB_WORKSPACE)/D/RnD/Frameworks/Matlab/ML/CNN/AI_NEOM/RemoteReality/%.c
	$(CC) $(CFLAGS) -o $@ $<


%.o : $(MATLAB_WORKSPACE)/D/RnD/Frameworks/Matlab/ML/CNN/AI_NEOM/RemoteReality/%.cpp
	$(CPP) $(CPPFLAGS) -o $@ $<


%.o : $(MATLAB_WORKSPACE)/D/RnD/Frameworks/Matlab/ML/CNN/AI_NEOM/RemoteReality/%.cu
	$(CPP) $(CPPFLAGS) -o $@ $<


rt_nonfinite.o : $(START_DIR)/rt_nonfinite.cpp
	$(CPP) $(CPPFLAGS) -o $@ $<


rtGetNaN.o : $(START_DIR)/rtGetNaN.cpp
	$(CPP) $(CPPFLAGS) -o $@ $<


rtGetInf.o : $(START_DIR)/rtGetInf.cpp
	$(CPP) $(CPPFLAGS) -o $@ $<


main_rr.o : $(MATLAB_WORKSPACE)/D/RnD/Frameworks/Matlab/ML/CNN/AI_NEOM/RemoteReality/main_rr.cpp
	$(CPP) $(CPPFLAGS) -o $@ $<


MJPEGWriter.o : $(MATLAB_WORKSPACE)/D/RnD/Frameworks/Matlab/ML/CNN/AI_NEOM/RemoteReality/MJPEGWriter.cpp
	$(CPP) $(CPPFLAGS) -o $@ $<


squeezenet_svm_predict_data.o : $(START_DIR)/squeezenet_svm_predict_data.cu
	$(CPP) $(CPPFLAGS) -o $@ $<


squeezenet_svm_predict_initialize.o : $(START_DIR)/squeezenet_svm_predict_initialize.cu
	$(CPP) $(CPPFLAGS) -o $@ $<


squeezenet_svm_predict_terminate.o : $(START_DIR)/squeezenet_svm_predict_terminate.cu
	$(CPP) $(CPPFLAGS) -o $@ $<


squeezenet_svm_predict.o : $(START_DIR)/squeezenet_svm_predict.cu
	$(CPP) $(CPPFLAGS) -o $@ $<


DeepLearningNetwork.o : $(START_DIR)/DeepLearningNetwork.cu
	$(CPP) $(CPPFLAGS) -o $@ $<


activations.o : $(START_DIR)/activations.cu
	$(CPP) $(CPPFLAGS) -o $@ $<


CompactClassificationECOC.o : $(START_DIR)/CompactClassificationECOC.cu
	$(CPP) $(CPPFLAGS) -o $@ $<


###########################################################################
## DEPENDENCIES
###########################################################################

$(ALL_OBJS) : rtw_proj.tmw $(MAKEFILE)


###########################################################################
## MISCELLANEOUS TARGETS
###########################################################################

info : 
	echo "### PRODUCT = $(PRODUCT)"
	echo "### PRODUCT_TYPE = $(PRODUCT_TYPE)"
	echo "### BUILD_TYPE = $(BUILD_TYPE)"
	echo "### INCLUDES = $(INCLUDES)"
	echo "### DEFINES = $(DEFINES)"
	echo "### ALL_SRCS = $(ALL_SRCS)"
	echo "### ALL_OBJS = $(ALL_OBJS)"
	echo "### LIBS = $(LIBS)"
	echo "### MODELREF_LIBS = $(MODELREF_LIBS)"
	echo "### SYSTEM_LIBS = $(SYSTEM_LIBS)"
	echo "### TOOLCHAIN_LIBS = $(TOOLCHAIN_LIBS)"
	echo "### CFLAGS = $(CFLAGS)"
	echo "### LDFLAGS = $(LDFLAGS)"
	echo "### SHAREDLIB_LDFLAGS = $(SHAREDLIB_LDFLAGS)"
	echo "### CPPFLAGS = $(CPPFLAGS)"
	echo "### CPP_LDFLAGS = $(CPP_LDFLAGS)"
	echo "### CPP_SHAREDLIB_LDFLAGS = $(CPP_SHAREDLIB_LDFLAGS)"
	echo "### ARFLAGS = $(ARFLAGS)"
	echo "### MEX_CFLAGS = $(MEX_CFLAGS)"
	echo "### MEX_CPPFLAGS = $(MEX_CPPFLAGS)"
	echo "### MEX_LDFLAGS = $(MEX_LDFLAGS)"
	echo "### MEX_CPPLDFLAGS = $(MEX_CPPLDFLAGS)"
	echo "### DOWNLOAD_FLAGS = $(DOWNLOAD_FLAGS)"
	echo "### EXECUTE_FLAGS = $(EXECUTE_FLAGS)"
	echo "### MAKE_FLAGS = $(MAKE_FLAGS)"


clean : 
	$(ECHO) "### Deleting all derived files..."
	$(RM) $(PRODUCT)
	$(RM) $(ALL_OBJS)
	$(RM) *.c.dep
	$(RM) *.cpp.dep .cu.dep
	$(ECHO) "### Deleted all derived files."


