CXX = g++
CXXFLAGS = -O3 -Wall -Wextra -std=c++20 -march=native

all: out/simd-shuffle out/leetcode-921

clean:
	rm out/simd-shuffle

run-simd-shuffle: out/simd-shuffle
	./out/simd-shuffle

run-leetcode-921: out/leetcode-921
	./out/leetcode-921

out/simd-shuffle: src/simd-shuffle.cpp src/common.hpp
	$(CXX) $(CXXFLAGS) -o out/simd-shuffle src/simd-shuffle.cpp

out/leetcode-921: src/leetcode-921.cpp src/common.hpp
	$(CXX) $(CXXFLAGS) -o out/leetcode-921 src/leetcode-921.cpp
