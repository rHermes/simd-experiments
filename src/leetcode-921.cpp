#include "common.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <string>

#include <immintrin.h>

#define VERBOSE 0

inline int
solveNormal(std::string_view input);
inline int
solveSIMD_SSE4(std::string_view input);

//
template<typename Gen>
void
generateInputBernoulli(Gen&& gen, std::span<char> output, const std::size_t N, const float p)
{
  if (output.size() < N)
    throw std::runtime_error("Output buffer is not big enough for requested input");

  std::bernoulli_distribution dist(p);
  for (std::size_t i = 0; i < N; i++) {
    output[i] = dist(gen) ? '(' : ')';
  }
}

int
main()
{
  // We are going to try to solve leetcode 921.
  std::array<char, 4096> bufferRaw;
  std::span<char, 4096> buffer(bufferRaw);

  std::ranlux48 rng(std::random_device{}());

  for (int i = 0; i < 40; i++) {
    generateInputBernoulli(rng, buffer, 32, 0.5);
    std::string_view view{ buffer.data(), 32 };

    const auto expected = solveNormal(view);
    const auto simdCalc = solveSIMD_SSE4(view);

    if (expected != simdCalc) {
      std::cout << "Expected: " << std::setw(3) << expected << ", got: " << std::setw(3) << simdCalc << " for: " << view
                << "\n";
			break;
    }
  }

  return 0;
}

inline int
solveSIMD_SSE4(std::string_view inputString)
{
  // we just knnow that these are good.
  const int N = inputString.size();

  int ans = 0;
  int balance = 0;

  for (int i = 0; i < N; i += 16) {
    const auto chunk = _mm_loadu_si128(reinterpret_cast<__m128i const*>(inputString.data() + i));
    const auto rightBraces = _mm_sub_epi8(chunk, _mm_set1_epi8(')'));
#ifdef VERBOSE
		std::cout << "Input string: [" << inputString.substr(i, 16) << "]\n";
		std::cout << "Chunk:        [" << m128toHex(rightBraces, true) << "]\n";
#endif

    // We want
  }

  return ans + balance;
}

inline int
solveNormal(std::string_view input)
{
  int ans = 0;
  int bal = 0;
  for (const auto c : input) {
    if (c == '(') {
      bal++;
    } else {
      bal--;
      if (bal < 0) {
        ans++;
        bal = 0;
      }
    }
  }

  return ans + bal;
}
