// The point of this file is to solve 2938.

#include "common.hpp"

#include <array>
#include <print>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <immintrin.h>

#include <nanobench.h>

namespace p2938 {

inline long long
solveScalar(std::string_view input);

inline long long
solveSIMD_SSE4_v1(std::string_view input);

inline long long
solveSIMD_SSE4_v2(std::string_view input);

inline long long
solveSIMD_AVX2_v1(std::string_view input);

inline long long
solveSIMD_SSE4(std::string_view input)
{
  return solveSIMD_SSE4_v2(input);
};

inline long long
solveSIMD_AVX2(std::string_view input)
{
  // Until later, when we implement this:
  // return solveScalar(input);
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

  b.title(desc).warmup(10).batch(size * inputLen).unit("byte").minEpochIterations(10);

  const auto runTest = [&](const std::string& title, const auto& solver) {
    b.run(title, [&]() {
      for (const auto& s : testSet) {
        const auto r = solver(s);
        ankerl::nanobench::doNotOptimizeAway(r);
      }
    });
  };

  b.relative(true);

  // runTest("Scalar", solveScalar);

  runTest("SSE4_v1", solveSIMD_SSE4_v1);
  runTest("SSE4_v2", solveSIMD_SSE4_v2);

  runTest("AVX2_v1", solveSIMD_AVX2_v1);
}

template<typename Rng>
bool
sanityCheck(Rng&& rng)
{
  // const auto testSet = generateBernoulliTestset(rng, 100, 100, 0.5);
  const auto testSet = generateBernoulliTestset(rng, 100, 100, 0.5);
  // const auto testSet = generateBernoulliTestset(rng, 100, 65, 0.5);
  // const auto testSet = generateBernoulliTestset(rng, 100, 32, 0.5);
  // const auto testSet = generateBernoulliTestset(rng, 100, 14, 0.5);

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
  const auto POSITIONS = _mm_set_epi8(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);

  // I = oldPos + lag, and in the beginning we have I = -1, so lag must be -1.
  auto lag = _mm_set_epi32(0, 0, 0, -1);
  auto update = _mm_setzero_si128();

  int i = 0;
  for (; i + 16 <= N; i += 16) {
    const auto chunk = _mm_loadu_si128(reinterpret_cast<__m128i const*>(inputString.data() + i));

    // Ok, so how do we want to do this. We are after all 0s here.
    // ok, we have selected all zeros now
    const auto zeroSpots = _mm_sub_epi8(chunk, _mm_set1_epi8('1'));
    // const auto zeroOnes = _mm_xor_si128(chunk, _mm_set1_epi8('1'));

    const auto indexes = _mm_and_si128(zeroSpots, POSITIONS);

    // Let's just figure out how much it costs to move everything one behind us.
    const auto revPsa = simd::calcReverseRunningSum<8>(indexes);
    const auto revZeroCount = simd::calcReverseRunningSum<8>(zeroSpots);
    // ok, so let's think here then.

    // count is now the negative of how many there are.
    // const auto cntPsa = simd::negate<8>(cntPsaNeg);

    // So now then, we I = oldPos + lag
    // oldPos = I - lag

    // ok, let's think here. We now know two things. How many positons are taken up. and the sum of all those
    // positions.
    // So all positions are relative to a spot 1 before the first element, we will call this number I.
    // a = I+1, b = I+2, ...
    //
    // Now we now have sum of the positons that contain a zero. So let's say we have a string where b and d is 0.
    // update = (b - oldPos) + (d - (oldPos+1))
    // update = (I + 2 - I + lag) + (I + 4 - I + lag - 1)
    // update = (2 + lag) + (4 + (lag-1))
    // update = psa + (lag + (lag-1))
    // update = psa + (numZeros*lag - TriangleNumber(numZeros-1))
    // update = psa - (TriangleNumber(numZeros-1) - numZeros*lag)

    // Add the reverse psa to update.
    update = _mm_add_epi64(update, _mm_cvtepu8_epi32(revPsa));

    // Now what to take away.
    const auto zeroCount = _mm_cvtepi8_epi32(revZeroCount);

    const auto posZeroCount = simd::negate<32>(zeroCount);
    auto triang = _mm_mul_epi32(posZeroCount, posZeroCount);
    triang = _mm_sub_epi32(triang, posZeroCount);
    triang = _mm_srli_epi32(triang, 1);

    // So this is T(n-1)

    auto tmpLag = _mm_mul_epi32(lag, zeroCount);
    tmpLag = _mm_add_epi64(_mm_cvtepi32_epi64(triang), tmpLag);

    update = _mm_sub_epi64(update, tmpLag);

    // Ok, now then only thing left to do is to update the lag. It will have increased with 1 for each non zero.
    // That means that the revCount can be used for this, just in reverse. If we have 16 of them, then it's going to
    // be 0. if we have 15 it's going to be 1. So we just do 16 + negCount.
    lag = _mm_add_epi32(lag, zeroCount);
    lag = _mm_add_epi32(lag, _mm_set1_epi32(16));
  }

  long long ans = _mm_cvtsi128_si64(update);
  long long spot = (static_cast<long long>(i) - 1 - _mm_cvtsi128_si32(lag));

  // std::print("We got out of the loop with: {} and {}\n", ans, spot);

  // now we just need to extract the end.
  for (; i < N; i++) {
    if (inputString[i] == '0') {
      ans += i - spot;
      spot++;
    }
  }

  return ans;
}

long long
solveSIMD_SSE4_v2(std::string_view inputString)
{
  const int N = inputString.size();

  const auto ONLY_FIRST_BYTE = _mm_set_epi32(0, 0, 0, 0xFF);

  const auto POSITIONS = _mm_set_epi8(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);

  // I = oldPos + lag, and in the beginning we have I = -1, so lag must be -1.
  auto lag = _mm_set_epi32(0, 0, 0, -1);
  auto update = _mm_setzero_si128();

  int i = 0;
  for (; i + 16 <= N; i += 16) {
    const auto chunk = _mm_loadu_si128(reinterpret_cast<__m128i const*>(inputString.data() + i));

    // Ok, so how do we want to do this. We are after all 0s here.
    // ok, we have selected all zeros now
    const auto zeroSpots = _mm_sub_epi8(chunk, _mm_set1_epi8('1'));
    // const auto zeroOnes = _mm_xor_si128(chunk, _mm_set1_epi8('1'));

    const auto indexes = _mm_and_si128(zeroSpots, POSITIONS);

    // Let's just figure out how much it costs to move everything one behind us.
    const auto revPsa = simd::calcReverseRunningSum<8>(indexes);
    const auto revZeroCount = simd::negate<8>(simd::calcReverseRunningSum<8>(zeroSpots));

    const auto psa = _mm_and_si128(revPsa, ONLY_FIRST_BYTE);
    const auto zeroCount = _mm_and_si128(revZeroCount, ONLY_FIRST_BYTE);

    // Add the reverse psa to update.
    update = _mm_add_epi64(update, psa);

    // Now what to take away.

    auto triang = _mm_mul_epi32(zeroCount, zeroCount);
    triang = _mm_sub_epi32(triang, zeroCount);
    triang = _mm_srli_epi32(triang, 1);

    // So this is T(n-1)
    update = _mm_sub_epi64(update, triang);

    auto tmpLag = _mm_mul_epi32(lag, zeroCount);
    update = _mm_add_epi64(update, tmpLag);

    // Ok, now then only thing left to do is to update the lag. It will have increased with 1 for each non zero.
    // That means that the revCount can be used for this, just in reverse. If we have 16 of them, then it's going to
    // be 0. if we have 15 it's going to be 1. So we just do 16 + negCount.
    lag = _mm_sub_epi32(lag, zeroCount);
    lag = _mm_add_epi32(lag, _mm_set1_epi32(16));
  }

  long long ans = _mm_cvtsi128_si64(update);
  long long spot = (i - 1 - _mm_cvtsi128_si32(lag));

  // std::print("We got out of the loop with: {} and {}\n", ans, spot);

  // now we just need to extract the end.
  for (; i < N; i++) {
    if (inputString[i] == '0') {
      ans += i - spot;
      spot++;
    }
  }

  return ans;
}

long long
solveSIMD_AVX2_v1(std::string_view inputString)
{
  // Direct copy of SSE4_v2 ish
  const int N = inputString.size();

  // We will later be bumping the upper ones by cnt*16.
  // which we can do 2*2*2*2 = leftshift4.

  // so you might notice here that we are reusing 1-16. The reason for this
  // is that if we used the proper 1-32, the upper parts could overflow. We need to handle this, so instead we use 16
  // both places and add the 16 shifts later.
  const auto POSITIONS = _mm256_set_epi8(
    16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);

  const auto ONLY_FIRST_BYTES = _mm256_set_epi32(0, 0, 0, 0xFF, 0, 0, 0, 0xFF);

  // I = oldPos + lag, and in the beginning we have I = -1, so lag must be -1.
  auto lag = _mm_set_epi32(0, 0, 0, -1);
  auto update = _mm_setzero_si128();

  int i = 0;
  for (; i + 32 <= N; i += 32) {
    const auto chunk = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(inputString.data() + i));

    // Ok, so how do we want to do this. We are after all 0s here.
    // ok, we have selected all zeros now
    const auto zeroSpots = _mm256_sub_epi8(chunk, _mm256_set1_epi8('1'));
    const auto indexes = _mm256_and_si256(zeroSpots, POSITIONS);

    // Let's just figure out how much it costs to move everything one behind us.
    const auto revPsa = simd::calcReverseRunningSum<8>(indexes);
    const auto revZeroCount = simd::calcReverseRunningSum<8>(zeroSpots);

    // We are zeroing out all other parts of the array, effectivly extending them.
    const auto psa = _mm256_and_si256(revPsa, ONLY_FIRST_BYTES);
    const auto zeroCount = _mm256_and_si256(simd::negate<8>(revZeroCount), ONLY_FIRST_BYTES);

    // Ok now we have two goals here. We need to add the upper parts.
    const auto lowerPsa = simd::extractLowerLane(psa);
    const auto upperPsa = simd::extractUpperLane(psa);

    // Add the reverse psa to update.
    update = _mm_add_epi64(update, lowerPsa);
    update = _mm_add_epi64(update, upperPsa);

    // Now we need to add upper
    const auto lowerZP = simd::extractLowerLane(zeroCount);
    const auto upperZP = simd::extractUpperLane(zeroCount);

    // OK, now let's add that to update.
    // THis is the same as adding by upperZP*16
    update = _mm_add_epi64(update, _mm_slli_epi64(upperZP, 4));

    // we want to add t
    const auto combinedZP = _mm_add_epi64(lowerZP, upperZP);

    // Compute the n-1 triangle number.
    auto triang = _mm_mul_epu32(combinedZP, combinedZP);
    triang = _mm_sub_epi32(triang, combinedZP);
    triang = _mm_srli_epi32(triang, 1);

    update = _mm_sub_epi64(update, triang);


    auto tmpLag = _mm_mul_epi32(lag, combinedZP);
    update = _mm_add_epi64(update, tmpLag);

    // Ok, now then only thing left to do is to update the lag. It will have increased with 1 for each non zero.
    // That means that the revCount can be used for this, just in reverse. If we have 32 of them, then it's going to
    // be 0. if we have 31 it's going to be 1. So we just do 32 + negCount.
    lag = _mm_sub_epi32(lag, combinedZP);
    lag = _mm_add_epi32(lag, _mm_set1_epi32(32));
  }

  long long ans = _mm_cvtsi128_si64(update);
  long long spot = (static_cast<long long>(i) - 1 - _mm_cvtsi128_si32(lag));

  // now we just need to extract the end.
  for (; i < N; i++) {
    if (inputString[i] == '0') {
      ans += i - spot;
      spot++;
    }
  }

  return ans;
}

} // namespace p2938

int
main()
{
  ankerl::nanobench::Rng rng(10);

  p2938::sanityCheck(rng);

  ankerl::nanobench::Bench b;

  p2938::runBernoulliTest(rng, "Short 50% strings", 1000, 10, 0.5);
  p2938::runBernoulliTest(rng, "Mid 50% strings", 1000, 100, 0.5);
  p2938::runBernoulliTest(rng, "Long 50% strings", 1000, 1000, 0.5);
  p2938::runBernoulliTest(rng, "Very long 50% strings", 1000, 10000, 0.5);


  return 0;
}