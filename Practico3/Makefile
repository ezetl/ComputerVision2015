BIN			=	harris_corners

# Flags
CUDA_INSTALL_PATH	= /opt/cuda
CUDASDK_INSTALL_PATH	= /opt/cudasdk
CUDA_LIB_PATH		= $(CUDA_INSTALL_PATH)/lib64
CUDASDK_LIB_PATH	= $(CUDASDK_INSTALL_PATH)/C/lib
INCLUDES		+= -I. -I$(CUDA_INSTALL_PATH)/include -I$(CUDASDK_INSTALL_PATH)/C/common/inc
LIBS			= -L$(CUDA_LIB_PATH) -L$(CUDASDK_LIB_PATH)
LDFLAGS			= -lopencv_core -lopencv_imgproc -lopencv_highgui -lcudart -std=c++0x 
NVCC_FLAGS		= 


# Compilers
NVCC		= $(CUDA_INSTALL_PATH)/bin/nvcc
GCC			= g++
LINKER		= g++

# Files
CU_SOURCES		= kernel.cu 
CU_OBJS			= $(patsubst %.cu, %.o, $(CU_SOURCES))
C_SOURCES		= DisplayImageCuda.cpp


# Rules
$(BIN): clean $(C_OBJS) $(CU_OBJS)
	$(LINKER) $(LDFLAGS) $(C_SOURCES) -o $(BIN) $(CU_OBJS) $(INCLUDES) $(LIBS)

$(CU_OBJS): $(CU_SOURCES)
	$(NVCC) -c $(CU_SOURCES) $(INCLUDES)
	
run: $(BIN)
	LD_LIBRARY_PATH=$(CUDA_LIB_PATH) ./$(BIN)

clean:
	rm -f $(BIN) *.o *.out
