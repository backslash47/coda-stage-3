# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/lib/python3.5/dist-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /usr/local/lib/python3.5/dist-packages/cmake/data/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ubuntu/tutorial_1/cuda-fixnum

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ubuntu/tutorial_1/cuda-fixnum/build

# Include any dependencies generated for this target.
include cuda-fixnum/CMakeFiles/cuda_bench.dir/depend.make

# Include the progress variables for this target.
include cuda-fixnum/CMakeFiles/cuda_bench.dir/progress.make

# Include the compile flags for this target's objects.
include cuda-fixnum/CMakeFiles/cuda_bench.dir/flags.make

cuda-fixnum/CMakeFiles/cuda_bench.dir/bench/bench.cu.o: cuda-fixnum/CMakeFiles/cuda_bench.dir/flags.make
cuda-fixnum/CMakeFiles/cuda_bench.dir/bench/bench.cu.o: ../cuda-fixnum/bench/bench.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/tutorial_1/cuda-fixnum/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object cuda-fixnum/CMakeFiles/cuda_bench.dir/bench/bench.cu.o"
	cd /home/ubuntu/tutorial_1/cuda-fixnum/build/cuda-fixnum && /usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/ubuntu/tutorial_1/cuda-fixnum/cuda-fixnum/bench/bench.cu -o CMakeFiles/cuda_bench.dir/bench/bench.cu.o

cuda-fixnum/CMakeFiles/cuda_bench.dir/bench/bench.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cuda_bench.dir/bench/bench.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

cuda-fixnum/CMakeFiles/cuda_bench.dir/bench/bench.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cuda_bench.dir/bench/bench.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target cuda_bench
cuda_bench_OBJECTS = \
"CMakeFiles/cuda_bench.dir/bench/bench.cu.o"

# External object files for target cuda_bench
cuda_bench_EXTERNAL_OBJECTS =

cuda-fixnum/CMakeFiles/cuda_bench.dir/cmake_device_link.o: cuda-fixnum/CMakeFiles/cuda_bench.dir/bench/bench.cu.o
cuda-fixnum/CMakeFiles/cuda_bench.dir/cmake_device_link.o: cuda-fixnum/CMakeFiles/cuda_bench.dir/build.make
cuda-fixnum/CMakeFiles/cuda_bench.dir/cmake_device_link.o: cuda-fixnum/CMakeFiles/cuda_bench.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ubuntu/tutorial_1/cuda-fixnum/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/cuda_bench.dir/cmake_device_link.o"
	cd /home/ubuntu/tutorial_1/cuda-fixnum/build/cuda-fixnum && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cuda_bench.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
cuda-fixnum/CMakeFiles/cuda_bench.dir/build: cuda-fixnum/CMakeFiles/cuda_bench.dir/cmake_device_link.o

.PHONY : cuda-fixnum/CMakeFiles/cuda_bench.dir/build

# Object files for target cuda_bench
cuda_bench_OBJECTS = \
"CMakeFiles/cuda_bench.dir/bench/bench.cu.o"

# External object files for target cuda_bench
cuda_bench_EXTERNAL_OBJECTS =

cuda-fixnum/bin/cuda_bench: cuda-fixnum/CMakeFiles/cuda_bench.dir/bench/bench.cu.o
cuda-fixnum/bin/cuda_bench: cuda-fixnum/CMakeFiles/cuda_bench.dir/build.make
cuda-fixnum/bin/cuda_bench: cuda-fixnum/CMakeFiles/cuda_bench.dir/cmake_device_link.o
cuda-fixnum/bin/cuda_bench: cuda-fixnum/CMakeFiles/cuda_bench.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ubuntu/tutorial_1/cuda-fixnum/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable bin/cuda_bench"
	cd /home/ubuntu/tutorial_1/cuda-fixnum/build/cuda-fixnum && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cuda_bench.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
cuda-fixnum/CMakeFiles/cuda_bench.dir/build: cuda-fixnum/bin/cuda_bench

.PHONY : cuda-fixnum/CMakeFiles/cuda_bench.dir/build

cuda-fixnum/CMakeFiles/cuda_bench.dir/clean:
	cd /home/ubuntu/tutorial_1/cuda-fixnum/build/cuda-fixnum && $(CMAKE_COMMAND) -P CMakeFiles/cuda_bench.dir/cmake_clean.cmake
.PHONY : cuda-fixnum/CMakeFiles/cuda_bench.dir/clean

cuda-fixnum/CMakeFiles/cuda_bench.dir/depend:
	cd /home/ubuntu/tutorial_1/cuda-fixnum/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/tutorial_1/cuda-fixnum /home/ubuntu/tutorial_1/cuda-fixnum/cuda-fixnum /home/ubuntu/tutorial_1/cuda-fixnum/build /home/ubuntu/tutorial_1/cuda-fixnum/build/cuda-fixnum /home/ubuntu/tutorial_1/cuda-fixnum/build/cuda-fixnum/CMakeFiles/cuda_bench.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : cuda-fixnum/CMakeFiles/cuda_bench.dir/depend

