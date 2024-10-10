CXX = g++
CXXFLAGS = -O3 -Wall -Wextra -std=c++20 -march=x86-64-v3 -mtune=native -Iinclude
# CXXFLAGS = -O3 -Wall -Wextra -std=c++20 -march=native -mtune=native -Iinclude

all: out/leetcode-2696 out/leetcode-921

clean:
	rm out/*
	rm obj/*

run-leetcode-2696: out/leetcode-2696
	@ "./$<"

run-leetcode-921: out/leetcode-921
	@ "./$<"

obj/nanobench.o: src/nanobench.cpp include/nanobench.h
	$(CXX) $(CXXFLAGS) -c -o "$@" "$<"

out/leetcode-2696: src/leetcode-2696.cpp src/common.hpp
	$(CXX) $(CXXFLAGS) -o "$@" "$<"

out/leetcode-921: src/leetcode-921.cpp obj/nanobench.o src/common.hpp
	$(CXX) $(CXXFLAGS) -o "$@" obj/nanobench.o "$<"
