# Compiler and flags
CXX := g++
CXXFLAGS := -Wextra -Wall -pedantic -O2 -std=c++17 -L/usr/lib64/root -lGui -lCore -lImt -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lROOTVecOps -lTree -lTreePlayer -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread -lMultiProc -lROOTDataFrame -pthread -lm -ldl -rdynamic -pthread -std=c++17 -m64 -I/usr/include/root

CXXFLAGS += -g

# Directories
SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin
SCRIPT_DIR := scripts

# File extensions
SRC_EXT := cpp
OBJ_EXT := o

# Source files and object files
SRCS := $(wildcard $(SRC_DIR)/*.$(SRC_EXT))
OBJS := $(patsubst $(SRC_DIR)/%.$(SRC_EXT), $(OBJ_DIR)/%.$(OBJ_EXT), $(SRCS))

# Scripts and binaries
SCRIPTS := $(wildcard $(SCRIPT_DIR)/*.$(SRC_EXT))
BINS := $(patsubst $(SCRIPT_DIR)/%.$(SRC_EXT), $(BIN_DIR)/%, $(SCRIPTS))


all: $(BINS)


# Rule to build binaries
$(BIN_DIR)/%: $(SCRIPT_DIR)/%.$(SRC_EXT) $(OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< $(OBJS) -o $@.out

# Rule to build object files
$(OBJ_DIR)/%.$(OBJ_EXT): $(SRC_DIR)/%.$(SRC_EXT) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create directories if they don't exist
$(OBJ_DIR) $(BIN_DIR):
	mkdir -p $@

# Clean target
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# Phony targets
.PHONY: all clean