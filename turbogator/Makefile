CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17 -I./src -MMD -MP -march=native

OPT_FLAGS = -O3
ASAN_FLAGS = -O0 -g -fsanitize=address
DBG_FLAGS = -O0 -g

SRC_DIR = src
BUILD_DIR = build

ALL_SRCS = $(wildcard $(SRC_DIR)/*.cpp $(SRC_DIR)/*/*.cpp)
COMMON_SRCS = $(filter-out $(SRC_DIR)/main.cpp $(SRC_DIR)/validate_main.cpp, $(ALL_SRCS))

BENCH_SRCS = $(COMMON_SRCS) $(SRC_DIR)/main.cpp
VALIDATE_SRCS = $(COMMON_SRCS) $(SRC_DIR)/validate_main.cpp

OPT_OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/opt/%.o, $(BENCH_SRCS))
ASAN_OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/asan/%.o, $(VALIDATE_SRCS))
DBG_OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/dbg/%.o, $(BENCH_SRCS))

all: benchmark validate valgrind assembly


# default benchmark for running and perf
$(BUILD_DIR)/opt/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(OPT_FLAGS) -c $< -o $@

benchmark: $(OPT_OBJS)
	$(CXX) $(CXXFLAGS) $(OPT_FLAGS) $^ -o $(BUILD_DIR)/bench_kernel

# validate with asan
$(BUILD_DIR)/asan/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(ASAN_FLAGS) -c $< -o $@

validate: $(ASAN_OBJS)
	$(CXX) $(CXXFLAGS) $(ASAN_FLAGS) $^ -o $(BUILD_DIR)/validate_kernel

# valgrind for callgrind
$(BUILD_DIR)/dbg/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(DBG_FLAGS) -c $< -o $@

valgrind: $(DBG_OBJS)
	$(CXX) $(CXXFLAGS) $(DBG_FLAGS) $^ -o $(BUILD_DIR)/bench_kernel_dbg

# for llvm-mca 
assembly:
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPT_FLAGS) -S $(SRC_DIR)/main.cpp -o $(BUILD_DIR)/bench_kernel.s

clean:
	rm -rf $(BUILD_DIR) metrics.json perf_metrics.json llvm_mca.txt callgrind.out.*

.PHONY: all benchmark validate valgrind assembly clean

-include $(wildcard $(BUILD_DIR)/*/*.d) $(wildcard $(BUILD_DIR)/*/*/*.d)