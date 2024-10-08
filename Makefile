
all: out/simd-shuffle

clean:
	rm out/simd-shuffle

run-simd-shuffle: out/simd-shuffle
	./out/simd-shuffle

out/simd-shuffle: src/simd-shuffle.cpp
	g++ -o out/simd-shuffle -O3 -Wall -Wextra -std=c++20 -march=native src/simd-shuffle.cpp
