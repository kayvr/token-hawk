CC := emcc
CXX := emcc
LD := emcc
BUILD ?= debug
ifeq ($(BUILD),debug)
    CXXFLAGS := -std=c++20 -Wall -Wextra -fwasm-exceptions -Wno-unused-function -O0 -g
    CCFLAGS := -Wall -Wextra -Wno-unused-function -O0 -g
else ifeq ($(BUILD),release)
    CXXFLAGS := -std=c++20 -Wall -Wextra -fwasm-exceptions -Wno-unused-function -O3
    CCFLAGS := -Wall -Wextra -Wno-unused-function -O3
endif
DEPFLAGS := -MMD -MP
OBJDIR := obj
SRCDIR := ../

MKDIR_P=mkdir -p
$(shell mkdir -p $(OBJDIR) >/dev/null)

CPP_SRC := $(wildcard $(SRCDIR)/*.cpp)
CPP_SRC += $(wildcard $(SRCDIR)/web/*.cpp)

C_SRC := $(wildcard $(SRCDIR)/*.c)

CXX_OBJ := $(CPP_SRC:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
CC_OBJ := $(C_SRC:$(SRCDIR)/%.c=$(OBJDIR)/%.o)

DEP := $(CXX_OBJ:%.o=%.d) $(CC_OBJ:%.o=%.d)

LDFLAGS := -s WASM=1 -s USE_WEBGPU=1 -s ALLOW_MEMORY_GROWTH=1 -s ENVIRONMENT=web -s TEXTDECODER=2 --output_eol=linux -fwasm-exceptions -s EXPORTED_FUNCTIONS="['_main', '_malloc', '_free']" -s EXPORTED_RUNTIME_METHODS="['intArrayFromString', 'setValue', 'ccall', 'stringToUTF8']" -s TOTAL_MEMORY=2147483648 -s MAXIMUM_MEMORY=4gb
TARGET := output

# Default target
all: live/index.html

$(info CXX_OBJ is $(CXX_OBJ))
$(info CC_OBJ is $(CC_OBJ))

live/index.html: $(CXX_OBJ) $(CC_OBJ)
	$(MKDIR_P) ./live
	$(LD) $(CXX_OBJ) $(CC_OBJ) $(LDFLAGS) -o ./live/index.html

# Compile C++ files
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

# Compile C files
$(OBJDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(@D)
	$(CC) $(CCFLAGS) $(DEPFLAGS) -c $< -o $@

# Include the dependency files
-include $(DEP)

# Clean up
.PHONY: clean
clean:
	rm -rf $(OBJDIR) $(TARGET) $(DEP)
