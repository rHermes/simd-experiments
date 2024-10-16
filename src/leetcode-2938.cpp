// The point of this file is to solve 2938.

#include "common.hpp"

#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <array>
#include <print>
#include <random>

#include <immintrin.h>

#include <nanobench.h>

namespace p2938 {

inline long long
solveScalar(std::string_view input);

inline long long
solveSIMD_SSE4_v1(std::string_view input);

inline long long
solveSIMD_AVX2_v1(std::string_view input);



inline long long
solveSIMD_SSE4(std::string_view input)
{
  return solveSIMD_SSE4_v1(input);
};

inline long long
solveSIMD_AVX2(std::string_view input)
{
  return solveSIMD_AVX2_v1(input);
}

template<typename Gen>
void
generateInputBernoulli(Gen&& gen, std::span<char> output, const float p)
{
  std::bernoulli_distribution dist(p);
  for (auto& out : output) {
    out = dist(gen) ? '1' : '0';
  }
}

template<typename Gen>
std::vector<std::string>
generateBernoulliTestset(Gen&& gen, const std::size_t size, const std::size_t inputLen, const float p)
{
  std::vector<std::string> strings(size);
  for (auto& s : strings) {
    s.resize(inputLen);
    generateInputBernoulli(gen, { s.data(), inputLen }, p);
  }

  return strings;
}

template<typename Rng>
void
runBernoulliTest(Rng&& rng, const std::string& desc, const std::size_t size, const std::size_t inputLen, const float p)
{
  ankerl::nanobench::Bench b;

  const auto testSet = generateBernoulliTestset(rng, size, inputLen, p);

  b.title(desc).warmup(10).batch(size * inputLen).unit("byte");

  const auto runTest = [&](const std::string& title, const auto& solver) {
    b.run(title, [&]() {
      for (const auto& s : testSet) {
        const auto r = solver(s);
        ankerl::nanobench::doNotOptimizeAway(r);
      }
    });
  };

  b.relative(true);

  runTest("Scalar", solveScalar);

  runTest("SSE4_v1", solveSIMD_SSE4_v1);
  runTest("AVX2_v1", solveSIMD_AVX2_v1);
}

template<typename Rng>
bool
sanityCheck(Rng&& rng)
{
  const auto testSet = generateBernoulliTestset(rng, 100, 100, 0.5);

  for (const auto& tc : testSet) {
    const auto expected = solveScalar(tc);
    const auto sse4 = solveSIMD_SSE4(tc);
    const auto avx2 = solveSIMD_AVX2(tc);

    if (sse4 != expected && avx2 != expected) {
      std::print("We expected {}, but sse4 got: {}, and avx2: {}\n", expected, sse4, avx2);
      return false;
    }

    if (sse4 != expected) {
      std::print("We expected {}, but sse4 got: {}\n", expected, sse4);
      return false;
    }

    if (avx2 != expected) {
      std::print("We expected {}, but avx2 got: {}\n", expected, avx2);
      return false;
    }
  }

  return true;
}

long long
solveScalar(std::string_view inputString)
{
  const int N = inputString.size();
  long long ans = 0;
  long long spot = 0;
  for (int i = 0; i < N; i++) {
    if (inputString[i] == '0') {
      ans += i - spot;
      spot++;
    }
  }
  return ans;
}

long long
solveSIMD_SSE4_v1(std::string_view inputString)
{
  const int N = inputString.size();

  const auto ALL_ZERO = _mm_setzero_si128();
  const auto ALL_ONE = _mm_set1_epi8(0x01);
  const auto ALL_SET = _mm_set1_epi8(-1);
  
  const auto POSITIONS = _mm_set_epi8(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);


  int i = 0;
  for (; i + 16 <= N; i += 16) {
    const auto chunk = _mm_loadu_si128(reinterpret_cast<__m128i const*>(inputString.data() + i));

    // Ok, so how do we want to do this. We are after all 0s here.
    // ok, we have selected all zeros now
    const auto zeroSpots = _mm_sub_epi8(chunk, _mm_set1_epi8('1'));
    const auto indexes = _mm_and_si128(zeroSpots, POSITIONS);

    // Let's just figure out how much it costs to move everything one behind us.
    const auto revPsa = simd::calcReverseRunningSum<8>(indexes);
    const auto cntPsaNeg = simd::calcReverseRunningSum<8>(zeroSpots);
    
    // count is now the negative of how many there are.
    const auto cntPsa = simd::negate<8>(cntPsaNeg);

    // ok, let's think here. We now know two things. How many positons are taken up.
  }

  return 0;
}

long long
solveSIMD_AVX2_v1(std::string_view inputString)
{
  return -1;
}

} // namespace p2938

int
main()
{
  ankerl::nanobench::Rng rng(10);


  p2938::sanityCheck(rng);

  /*
  ankerl::nanobench::Bench b;

  p2938::runBernoulliTest(rng, "Short 50% strings", 1000, 10, 0.5);
  p2938::runBernoulliTest(rng, "Mid 50% strings", 1000, 100, 0.5);
  p2938::runBernoulliTest(rng, "Long 50% strings", 1000, 1000, 0.5);
  p2938::runBernoulliTest(rng, "Very long 50% strings", 1000, 10000, 0.5);
  */
  return 0;
}