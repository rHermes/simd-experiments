#include "common.hpp"

#include <nanobench.h>

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

inline int
solveSIMD_SSE4_v1(std::string_view input);

inline int
solveSIMD_SSE4_v2(std::string_view input);

inline int
solveSIMD_AVX2(std::string_view input);

inline int
solveSIMD_AVX2_v1(std::string_view input);

inline int
solveSIMD_AVX2_v2(std::string_view input);

inline int
solveSIMD_AVX2_v3(std::string_view input);

inline int
solveSIMD_AVX2_v4(std::string_view input);

inline int
solveSIMD_AVX2_v5(std::string_view input);

inline int
solveSIMD_AVX2_v6(std::string_view input);

inline int
solveSIMD_AVX2_v7(std::string_view input);

inline int
solveSIMD_AVX2_v8(std::string_view input);

//
template<typename Gen>
void
generateInputBernoulli(Gen&& gen, std::span<char> output, const float p)
{
  std::bernoulli_distribution dist(p);
  for (std::size_t i = 0; i < output.size(); i++) {
    output[i] = dist(gen) ? '(' : ')';
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
runBernoulliTest(Rng&& rng, const std::string& title, const std::size_t size, const std::size_t inputLen, const float p)
{
  ankerl::nanobench::Bench b;

  const auto testSet = generateBernoulliTestset(rng, size, inputLen, p);

  b.title(title).warmup(10).batch(size * inputLen).unit("byte");

  /* b.relative(true).run("Normal", [&]() { */
  /*   for (const auto& s : testSet) { */
  /*     const auto r = solveNormal(s); */
  /*     ankerl::nanobench::doNotOptimizeAway(r); */
  /*   } */
  /* }); */

  b.relative(true).run("SSE4_v1", [&]() {
    for (const auto& s : testSet) {
      const auto r = solveSIMD_SSE4_v1(s);
      ankerl::nanobench::doNotOptimizeAway(r);
    }
  });

  b.run("SSE4_v2", [&]() {
    for (const auto& s : testSet) {
      const auto r = solveSIMD_SSE4_v2(s);
      ankerl::nanobench::doNotOptimizeAway(r);
    }
  });

  b.run("AVX2_v1", [&]() {
    for (const auto& s : testSet) {
      const auto r = solveSIMD_AVX2_v1(s);
      ankerl::nanobench::doNotOptimizeAway(r);
    }
  });

  b.run("AVX2_v2", [&]() {
    for (const auto& s : testSet) {
      const auto r = solveSIMD_AVX2_v2(s);
      ankerl::nanobench::doNotOptimizeAway(r);
    }
  });
	
	// Best performer so far
  b.run("AVX2_v3", [&]() {
    for (const auto& s : testSet) {
      const auto r = solveSIMD_AVX2_v3(s);
      ankerl::nanobench::doNotOptimizeAway(r);
    }
  });

  b.run("AVX2_v4", [&]() {
    for (const auto& s : testSet) {
      const auto r = solveSIMD_AVX2_v4(s);
      ankerl::nanobench::doNotOptimizeAway(r);
    }
  });

  b.run("AVX2_v5", [&]() {
    for (const auto& s : testSet) {
      const auto r = solveSIMD_AVX2_v5(s);
      ankerl::nanobench::doNotOptimizeAway(r);
    }
  });

  b.run("AVX2_v6", [&]() {
    for (const auto& s : testSet) {
      const auto r = solveSIMD_AVX2_v6(s);
      ankerl::nanobench::doNotOptimizeAway(r);
    }
  });

  /* b.run("AVX2_V7", [&]() { */
  /*   for (const auto& s : testSet) { */
  /*     const auto r = solveSIMD_AVX2_v7(s); */
  /*     ankerl::nanobench::doNotOptimizeAway(r); */
  /*   } */
  /* }); */

  b.run("AVX2_v8", [&]() {
    for (const auto& s : testSet) {
      const auto r = solveSIMD_AVX2_v8(s);
      ankerl::nanobench::doNotOptimizeAway(r);
    }
  });
}

template<typename Rng>
bool
sanityCheck(Rng&& rng)
{
  /* const auto testSet = generateBernoulliTestset(rng, 100, 100, 0.5); */
  const auto testSet = generateBernoulliTestset(rng, 100, 64, 0.5);

  for (const auto& tc : testSet) {
    const auto expected = solveNormal(tc);
    const auto sse4 = solveSIMD_SSE4(tc);
    const auto avx2 = solveSIMD_AVX2(tc);

    if (sse4 != expected && avx2 != expected) {
      std::cout << "We expected: " << expected << ", but sse4 got: " << sse4 << ", and avx512: " << avx2
                << ", for tc: " << tc << "\n";
      return false;
    }

    if (sse4 != expected) {
      std::cout << "We expected: " << expected << ", but sse4 got: " << sse4 << ", for: " << tc << "\n";
      return false;
    }

    if (avx2 != expected) {
      std::cout << "We expected: " << expected << ", and avx2: " << avx2 << ", for: " << tc << "\n";
      return false;
    }
  }

  return true;
}

int
main()
{
  ankerl::nanobench::Rng rng(10);

  sanityCheck(rng);

  ankerl::nanobench::Bench b;

  runBernoulliTest(rng, "Short 50% strings", 1000, 10, 0.5);
  runBernoulliTest(rng, "Mid 50% strings", 1000, 100, 0.5);
  runBernoulliTest(rng, "Long 50% strings", 1000, 1000, 0.5);
  runBernoulliTest(rng, "Very long 50% strings", 1000, 10000, 0.5);

  return 0;
}

inline int
solveSIMD_AVX2(std::string_view inputString)
{
  /* return solveSIMD_AVX2_v3(inputString); */
  return solveSIMD_AVX2_v8(inputString);
}

inline int
solveSIMD_SSE4(std::string_view inputString)
{
  return solveSIMD_SSE4_v2(inputString);
}

inline int
solveSIMD_AVX2_v7(std::string_view inputString)
{
  // Ok, so we are going to try something different now, which is to store the information to an array.
  // These are fixed size for now, to avoid extra overhead.
  std::array<std::int8_t, 1000> psaOut;
  std::array<std::int8_t, 1000> minOut;

  const int N = inputString.size();

  const auto ALL_ZERO = _mm256_setzero_si256();
  const auto ALL_ONE = _mm256_set1_epi8(0x01);
  const auto ALL_SET = _mm256_set1_epi8(0xFF);

  /* auto ans = _mm256_setzero_si256(); */
  /* auto balance = _mm256_setzero_si256(); */
  auto psaOutPtr = psaOut.data();
  auto minOutPtr = minOut.data();

  int i = 0;
  for (; i + 32 <= N; i += 32) {
    const auto chunk = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(inputString.data() + i));
    const auto leftBraces = _mm256_sub_epi8(chunk, _mm256_set1_epi8(')'));
    const auto rightBraces = _mm256_andnot_si256(leftBraces, ALL_SET);
    const auto valueChunk = _mm256_or_si256(rightBraces, ALL_ONE);

    auto psa = _mm256_add_epi8(valueChunk, _mm256_bslli_epi128(valueChunk, 1));
    psa = _mm256_add_epi8(psa, _mm256_bslli_epi128(psa, 2));
    psa = _mm256_add_epi8(psa, _mm256_bslli_epi128(psa, 4));
    psa = _mm256_add_epi8(psa, _mm256_bslli_epi128(psa, 8));

    auto min = _mm256_min_epi8(psa, _mm256_bslli_epi128(psa, 1));
    min = _mm256_min_epi8(min, _mm256_bslli_epi128(min, 2));
    min = _mm256_min_epi8(min, _mm256_bslli_epi128(min, 4));
    min = _mm256_min_epi8(min, _mm256_bslli_epi128(min, 8));

    // ok, let's swap min and psa.
    const auto flippedPSA = _mm256_permute2x128_si256(psa, psa, 0x01);
    const auto flippedMin = _mm256_permute2x128_si256(min, min, 0x01);

    // ok, now we add these together.
    const auto finalPsa = _mm256_add_epi8(flippedPSA, psa);
    const auto finalMin = _mm256_add_epi8(flippedMin, psa);

    // ok, so now we just have to compare.
    const auto totalMin = _mm256_min_epi8(min, finalMin);
    const auto negMin = _mm256_sub_epi8(ALL_ZERO, totalMin);

    // now then, we are going to revesre the totalMin, since we need that.
    // ok, so this is nice, we can actually get both of them like we want by shifting 15 to the right.
    const auto onlyPSA = _mm256_bsrli_epi128(finalPsa, 15);
    const auto onlyMin = _mm256_bsrli_epi128(negMin, 15);

    *(psaOutPtr++) = _mm256_extract_epi8(onlyPSA, 0);
    *(minOutPtr++) = _mm256_extract_epi8(onlyMin, 0);

    /* _mm256_storeu_si256(reinterpret_cast<__m256i*>(psaOutPtr), onlyPSA); */
    /* _mm256_storeu_si256(reinterpret_cast<__m256i*>(minOutPtr), onlyMin); */

    /* psaOutPtr++; */
    /* minOutPtr++; */
  }

  /* int scalarBalance = _mm256_cvtsi256_si32(balance); */
  /* int scalarAns = _mm256_cvtsi256_si32(ans); */
  int scalarBalance = psaOut[i - 1];
  int scalarAns = minOut[i - 1];

  for (; i < N; i++) {
    if (inputString[i] == '(') {
      scalarBalance++;
    } else {
      scalarBalance--;
      if (scalarBalance < 0) {
        scalarAns++;
        scalarBalance = 0;
      }
    }
  }

  return scalarAns + scalarBalance;
}

inline int
solveSIMD_AVX2_v6(std::string_view inputString)
{
  // Ok, here we try to unroll the loop manually, to be 2 long.
  // Since v5 which tried to be smart at the end didn't work out good,
  // we are now simplefying it by just doubling up v4
  const int N = inputString.size();

  const auto ALL_ZERO = _mm256_setzero_si256();
  const auto ALL_ONE = _mm256_set1_epi8(0x01);
  const auto ALL_SET = _mm256_set1_epi8(0xFF);

  auto ans = _mm256_setzero_si256();
  auto balance = _mm256_setzero_si256();

  int i = 0;
  for (; i + 64 <= N; i += 64) {
    const auto chunk1 = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(inputString.data() + i));
    const auto chunk2 = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(inputString.data() + 32 + i));

    const auto leftBraces1 = _mm256_sub_epi8(chunk1, _mm256_set1_epi8(')'));
    const auto leftBraces2 = _mm256_sub_epi8(chunk2, _mm256_set1_epi8(')'));

    const auto rightBraces1 = _mm256_andnot_si256(leftBraces1, ALL_SET);
    const auto rightBraces2 = _mm256_andnot_si256(leftBraces2, ALL_SET);

    const auto valueChunk1 = _mm256_or_si256(rightBraces1, ALL_ONE);
    const auto valueChunk2 = _mm256_or_si256(rightBraces2, ALL_ONE);

    auto psa1 = _mm256_add_epi8(valueChunk1, _mm256_bslli_epi128(valueChunk1, 1));
    psa1 = _mm256_add_epi8(psa1, _mm256_bslli_epi128(psa1, 2));
    psa1 = _mm256_add_epi8(psa1, _mm256_bslli_epi128(psa1, 4));
    psa1 = _mm256_add_epi8(psa1, _mm256_bslli_epi128(psa1, 8));

    auto psa2 = _mm256_add_epi8(valueChunk2, _mm256_bslli_epi128(valueChunk2, 1));
    psa2 = _mm256_add_epi8(psa2, _mm256_bslli_epi128(psa2, 2));
    psa2 = _mm256_add_epi8(psa2, _mm256_bslli_epi128(psa2, 4));
    psa2 = _mm256_add_epi8(psa2, _mm256_bslli_epi128(psa2, 8));

    auto min1 = _mm256_min_epi8(psa1, _mm256_bslli_epi128(psa1, 1));
    min1 = _mm256_min_epi8(min1, _mm256_bslli_epi128(min1, 2));
    min1 = _mm256_min_epi8(min1, _mm256_bslli_epi128(min1, 4));
    min1 = _mm256_min_epi8(min1, _mm256_bslli_epi128(min1, 8));

    auto min2 = _mm256_min_epi8(psa2, _mm256_bslli_epi128(psa2, 1));
    min2 = _mm256_min_epi8(min2, _mm256_bslli_epi128(min2, 2));
    min2 = _mm256_min_epi8(min2, _mm256_bslli_epi128(min2, 4));
    min2 = _mm256_min_epi8(min2, _mm256_bslli_epi128(min2, 8));

    // ok, let's swap min and psa.
    const auto flippedPSA1 = _mm256_permute2x128_si256(psa1, psa1, 0x01);
    const auto flippedMin1 = _mm256_permute2x128_si256(min1, min1, 0x01);

    const auto flippedPSA2 = _mm256_permute2x128_si256(psa2, psa2, 0x01);
    const auto flippedMin2 = _mm256_permute2x128_si256(min2, min2, 0x01);

    // ok, now we add these together.
    const auto finalPsa1 = _mm256_add_epi8(flippedPSA1, psa1);
    const auto finalMin1 = _mm256_add_epi8(flippedMin1, psa1);

    const auto finalPsa2 = _mm256_add_epi8(flippedPSA2, psa2);
    const auto finalMin2 = _mm256_add_epi8(flippedMin2, psa2);

    // ok, so now we just have to compare.
    const auto totalMin1 = _mm256_min_epi8(min1, finalMin1);
    const auto negMin1 = _mm256_sub_epi8(ALL_ZERO, totalMin1);

    const auto totalMin2 = _mm256_min_epi8(min2, finalMin2);
    const auto negMin2 = _mm256_sub_epi8(ALL_ZERO, totalMin2);

    // now then, we are going to revesre the totalMin, since we need that.
    // ok, so this is nice, we can actually get both of them like we want by shifting 15 to the right.
    const auto onlyPSA1 = _mm256_bsrli_epi128(finalPsa1, 15);
    const auto onlyMin1 = _mm256_bsrli_epi128(negMin1, 15);

    const auto onlyPSA2 = _mm256_bsrli_epi128(finalPsa2, 15);
    const auto onlyMin2 = _mm256_bsrli_epi128(negMin2, 15);

    // Now then, we are going to convert this.
    const auto lowOnlyPSA1 = _mm256_castsi256_si128(onlyPSA1);
    const auto lowOnlyMin1 = _mm256_castsi256_si128(onlyMin1);

    const auto lowOnlyPSA2 = _mm256_castsi256_si128(onlyPSA2);
    const auto lowOnlyMin2 = _mm256_castsi256_si128(onlyMin2);

    // We need to use 32, because there there is no max_epi64
    const auto onlyPSA_32_1 = _mm256_cvtepi8_epi32(lowOnlyPSA1);
    const auto onlyMin_32_1 = _mm256_cvtepi8_epi32(lowOnlyMin1);

    const auto onlyPSA_32_2 = _mm256_cvtepi8_epi32(lowOnlyPSA2);
    const auto onlyMin_32_2 = _mm256_cvtepi8_epi32(lowOnlyMin2);

    const auto padding1 = _mm256_max_epi32(_mm256_sub_epi32(onlyMin_32_1, balance), ALL_ZERO);
    const auto tempAdd1 = _mm256_add_epi32(padding1, onlyPSA_32_1);

    balance = _mm256_add_epi32(balance, tempAdd1);
    ans = _mm256_add_epi32(ans, padding1);

    const auto padding2 = _mm256_max_epi32(_mm256_sub_epi32(onlyMin_32_2, balance), ALL_ZERO);
    const auto tempAdd2 = _mm256_add_epi32(padding2, onlyPSA_32_2);

    balance = _mm256_add_epi32(balance, tempAdd2);
    ans = _mm256_add_epi32(ans, padding2);
  }

  int scalarBalance = _mm256_cvtsi256_si32(balance);
  int scalarAns = _mm256_cvtsi256_si32(ans);

  for (; i < N; i++) {
    if (inputString[i] == '(') {
      scalarBalance++;
    } else {
      scalarBalance--;
      if (scalarBalance < 0) {
        scalarAns++;
        scalarBalance = 0;
      }
    }
  }

  return scalarAns + scalarBalance;
}

inline int
solveSIMD_AVX2_v5(std::string_view inputString)
{
  // Ok, here we try to unroll the loop manually, to be 2 long.
  const int N = inputString.size();

  const auto ALL_ZERO = _mm256_setzero_si256();
  const auto ALL_ONE = _mm256_set1_epi8(0x01);
  const auto ALL_SET = _mm256_set1_epi8(0xFF);

  auto ans = _mm256_setzero_si256();
  auto balance = _mm256_setzero_si256();

  int i = 0;
  for (; i + 64 <= N; i += 64) {
    const auto chunk1 = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(inputString.data() + i));
    const auto chunk2 = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(inputString.data() + 32 + i));

    const auto leftBraces1 = _mm256_sub_epi8(chunk1, _mm256_set1_epi8(')'));
    const auto leftBraces2 = _mm256_sub_epi8(chunk2, _mm256_set1_epi8(')'));

    const auto rightBraces1 = _mm256_andnot_si256(leftBraces1, ALL_SET);
    const auto rightBraces2 = _mm256_andnot_si256(leftBraces2, ALL_SET);

    const auto valueChunk1 = _mm256_or_si256(rightBraces1, ALL_ONE);
    const auto valueChunk2 = _mm256_or_si256(rightBraces2, ALL_ONE);

    auto psa1 = _mm256_add_epi8(valueChunk1, _mm256_bslli_epi128(valueChunk1, 1));
    psa1 = _mm256_add_epi8(psa1, _mm256_bslli_epi128(psa1, 2));
    psa1 = _mm256_add_epi8(psa1, _mm256_bslli_epi128(psa1, 4));
    psa1 = _mm256_add_epi8(psa1, _mm256_bslli_epi128(psa1, 8));

    auto psa2 = _mm256_add_epi8(valueChunk2, _mm256_bslli_epi128(valueChunk2, 1));
    psa2 = _mm256_add_epi8(psa2, _mm256_bslli_epi128(psa2, 2));
    psa2 = _mm256_add_epi8(psa2, _mm256_bslli_epi128(psa2, 4));
    psa2 = _mm256_add_epi8(psa2, _mm256_bslli_epi128(psa2, 8));

    auto min1 = _mm256_min_epi8(psa1, _mm256_bslli_epi128(psa1, 1));
    min1 = _mm256_min_epi8(min1, _mm256_bslli_epi128(min1, 2));
    min1 = _mm256_min_epi8(min1, _mm256_bslli_epi128(min1, 4));
    min1 = _mm256_min_epi8(min1, _mm256_bslli_epi128(min1, 8));

    auto min2 = _mm256_min_epi8(psa2, _mm256_bslli_epi128(psa2, 1));
    min2 = _mm256_min_epi8(min2, _mm256_bslli_epi128(min2, 2));
    min2 = _mm256_min_epi8(min2, _mm256_bslli_epi128(min2, 4));
    min2 = _mm256_min_epi8(min2, _mm256_bslli_epi128(min2, 8));

    // ok, lets get these into a single lane of a

    // ok, let's see what we can do.
    // Let's see here then if we can do that.
    // ok, now let's see here.
    //
    // Let's just create 4 here.
    const auto onlyPsa1 = _mm256_permute4x64_epi64(psa1, 0b00'00'11'01);
    const auto onlyPsa2 = _mm256_permute4x64_epi64(psa2, 0b00'00'11'01);

    // onlyPsa1 = [.., .., psa1_2_hi, psa1_1_hi]

    const auto onlyMin1 = _mm256_permute4x64_epi64(min1, 0b00'00'11'01);
    const auto onlyMin2 = _mm256_permute4x64_epi64(min2, 0b00'00'11'01);

    // We now transfer this.
    const auto MERGE_LANES_LOWER =
      _mm256_set_epi64x(0x8080808080808080, 0x808080800F078080, 0x8080808080808080, 0x8080808080800F07);
    const auto MERGE_LANES_UPPER =
      _mm256_set_epi64x(0x8080808080808080, 0x0F07808080808080, 0x8080808080808080, 0x80800F0780808080);

    auto mergePsa = _mm256_permute2x128_si256(onlyPsa2, onlyPsa1, 0x02);
    mergePsa = _mm256_shuffle_epi8(mergePsa, MERGE_LANES_LOWER);

    auto mergeMin = _mm256_permute2x128_si256(onlyMin2, onlyMin1, 0x02);
    mergeMin = _mm256_shuffle_epi8(mergeMin, MERGE_LANES_UPPER);

    // ok, in this case, we have [compressed min2, compressed min1]

    // Ok we have them now, how do we combine them?
    // We make the upper part zero.
    mergePsa = _mm256_or_si256(mergePsa, _mm256_permute2x128_si256(mergePsa, mergePsa, 0x81));
    mergeMin = _mm256_or_si256(mergeMin, _mm256_permute2x128_si256(mergeMin, mergeMin, 0x81));

    // ok, now then we join them
    const auto mergeAll = _mm256_or_si256(mergePsa, mergeMin);
    const auto lowerMergeAll = _mm256_castsi256_si128(mergeAll);

    const auto signExtended = _mm256_cvtepi8_epi32(lowerMergeAll);

    // ok, let's now create the combined PSA
    auto mush = _mm256_add_epi32(signExtended, _mm256_bslli_epi128(signExtended, 4));
    mush = _mm256_add_epi32(mush, _mm256_bslli_epi128(mush, 8));

    const auto swappedMush = _mm256_permute2x128_si256(mush, mush, 0x01);
    const auto shiftedMush = _mm256_bslli_epi128(swappedMush, 4);
    const auto minPosAdj = _mm256_add_epi32(signExtended, shiftedMush);

    auto minPos = _mm256_min_epi32(minPosAdj, _mm256_bslli_epi128(minPosAdj, 4));
    minPos = _mm256_min_epi32(minPos, _mm256_bslli_epi128(minPos, 8));

    const auto negatedMin = _mm256_sub_epi32(ALL_ZERO, minPos);

    // Ah we have to change the way around.
    const auto padding = _mm256_max_epi32(_mm256_sub_epi32(negatedMin, balance), ALL_ZERO);
    const auto tempAdd = _mm256_add_epi32(swappedMush, padding);

    balance = _mm256_add_epi32(balance, tempAdd);
    ans = _mm256_add_epi32(ans, padding);
  }

  int scalarBalance = _mm256_extract_epi32(balance, 7);
  int scalarAns = _mm256_extract_epi32(ans, 7);

  /* int scalarBalance = _mm256_cvtsi256_si32(balance); */
  /* int scalarAns = _mm256_cvtsi256_si32(ans); */

  for (; i < N; i++) {
    if (inputString[i] == '(') {
      scalarBalance++;
    } else {
      scalarBalance--;
      if (scalarBalance < 0) {
        scalarAns++;
        scalarBalance = 0;
      }
    }
  }

  return scalarAns + scalarBalance;
}

inline int
solveSIMD_SSE4_v1(std::string_view inputString)
{
  const int N = inputString.size();

  int ans = 0;
  int balance = 0;

  const auto ALL_ONE = _mm_set1_epi8(0x01);
  const auto ALL_SET = _mm_set1_epi8(0xFF);

  int i = 0;
  for (; i + 16 <= N; i += 16) {
    const auto chunk = _mm_loadu_si128(reinterpret_cast<__m128i const*>(inputString.data() + i));
    const auto leftBraces = _mm_sub_epi8(chunk, _mm_set1_epi8(')'));
    const auto rightBraces = _mm_andnot_si128(leftBraces, ALL_SET);
    const auto valueChunk = _mm_or_si128(rightBraces, ALL_ONE);

    auto psa = _mm_add_epi8(valueChunk, _mm_bslli_si128(valueChunk, 1));
    psa = _mm_add_epi8(psa, _mm_bslli_si128(psa, 2));
    psa = _mm_add_epi8(psa, _mm_bslli_si128(psa, 4));
    psa = _mm_add_epi8(psa, _mm_bslli_si128(psa, 8));

    // Now if I have more than 16 in balance,  I can skip the next bit,
    // but I don't, since it's better to just pipeline everything.
    auto min = _mm_min_epi8(psa, _mm_bslli_si128(psa, 1));
    min = _mm_min_epi8(min, _mm_bslli_si128(min, 2));
    min = _mm_min_epi8(min, _mm_bslli_si128(min, 4));
    min = _mm_min_epi8(min, _mm_bslli_si128(min, 8));

    const std::int8_t finPsa = static_cast<std::int8_t>(_mm_extract_epi8(psa, 15));
    const std::int8_t minVal = -static_cast<std::int8_t>(_mm_extract_epi8(min, 15));

    const int padding = std::max(0, minVal - balance);
    balance += finPsa + padding;
    ans += padding;
  }

  for (; i < N; i++) {
    if (inputString[i] == '(') {
      balance++;
    } else {
      balance--;
      if (balance < 0) {
        ans++;
        balance = 0;
      }
    }
  }

  return ans + balance;
}

inline int
solveSIMD_SSE4_v2(std::string_view inputString)
{
  const int N = inputString.size();

  // Ok, so this version is an experiment, where we keep a running count of the min and psa
  // instead and handle the output at the end.

  const auto ALL_ONE = _mm_set1_epi8(0x01);
  const auto ALL_SET = _mm_set1_epi8(0xFF);
  const auto ALL_ZERO = _mm_setzero_si128();

  // 32 bit running sum min
  auto runningMin = ALL_ZERO;
  auto runningSum = ALL_ZERO;

  int i = 0;
  for (; i + 16 <= N; i += 16) {
    const auto chunk = _mm_loadu_si128(reinterpret_cast<__m128i const*>(inputString.data() + i));
    const auto leftBraces = _mm_sub_epi8(chunk, _mm_set1_epi8(')'));
    const auto rightBraces = _mm_andnot_si128(leftBraces, ALL_SET);
    const auto valueChunk = _mm_or_si128(rightBraces, ALL_ONE);

    auto psa = _mm_add_epi8(valueChunk, _mm_bslli_si128(valueChunk, 1));
    psa = _mm_add_epi8(psa, _mm_bslli_si128(psa, 2));
    psa = _mm_add_epi8(psa, _mm_bslli_si128(psa, 4));
    psa = _mm_add_epi8(psa, _mm_bslli_si128(psa, 8));

    // Now if I have more than 16 in balance,  I can skip the next bit,
    // but I don't, since it's better to just pipeline everything.
    auto min = _mm_min_epi8(psa, _mm_bslli_si128(psa, 1));
    min = _mm_min_epi8(min, _mm_bslli_si128(min, 2));
    min = _mm_min_epi8(min, _mm_bslli_si128(min, 4));
    min = _mm_min_epi8(min, _mm_bslli_si128(min, 8));

    const auto extendedMin = _mm_cvtepi8_epi32(_mm_bsrli_si128(min, 15));
    const auto extendedPsa = _mm_cvtepi8_epi32(_mm_bsrli_si128(psa, 15));

    // Either prevPsa + curMin is a new minimum, or it's not.
    runningMin = _mm_min_epi32(runningMin, _mm_add_epi32(runningSum, extendedMin));
    runningSum = _mm_add_epi32(runningSum, extendedPsa);
  }

  std::int32_t scalarMin = _mm_extract_epi32(runningMin, 0);
  std::int32_t scalarSum = _mm_extract_epi32(runningSum, 0);

  /* std::cout << "Coming out we have: min = " << scalarMin << ", sum: " << scalarSum << "\n"; */

  for (; i < N; i++) {
    if (inputString[i] == '(') {
      scalarSum++;
    } else {
      scalarSum--;
      scalarMin = std::min(scalarMin, scalarSum);
    }
  }

  /* std::cout << "Coming out final we have: min = " << scalarMin << ", sum: " << scalarSum << "\n"; */
  return scalarSum - 2 * scalarMin;
}

inline int
solveSIMD_AVX2_v8(std::string_view inputString)
{
  // So this is is basically just the AVX256 version of SSE4_v2
  const int N = inputString.size();

  const auto ALL_ZERO = _mm256_setzero_si256();
  const auto ALL_ONE = _mm256_set1_epi8(0x01);
  const auto ALL_SET = _mm256_set1_epi8(0xFF);

  auto runningSum = ALL_ZERO;
  /* auto runningSum = _mm256_set1_epi32(0); */
  auto runningMin = _mm256_set1_epi32(0);

  int i = 0;
  for (; i + 32 <= N; i += 32) {
    const auto chunk = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(inputString.data() + i));
    const auto leftBraces = _mm256_sub_epi8(chunk, _mm256_set1_epi8(')'));
    const auto rightBraces = _mm256_andnot_si256(leftBraces, ALL_SET);
    const auto valueChunk = _mm256_or_si256(rightBraces, ALL_ONE);

    auto psa = _mm256_add_epi8(valueChunk, _mm256_bslli_epi128(valueChunk, 1));
    psa = _mm256_add_epi8(psa, _mm256_bslli_epi128(psa, 2));
    psa = _mm256_add_epi8(psa, _mm256_bslli_epi128(psa, 4));
    psa = _mm256_add_epi8(psa, _mm256_bslli_epi128(psa, 8));

    auto min = _mm256_min_epi8(psa, _mm256_bslli_epi128(psa, 1));
    min = _mm256_min_epi8(min, _mm256_bslli_epi128(min, 2));
    min = _mm256_min_epi8(min, _mm256_bslli_epi128(min, 4));
    min = _mm256_min_epi8(min, _mm256_bslli_epi128(min, 8));

    // ok, let's swap min and psa.
    const auto flippedPSA = _mm256_permute2x128_si256(psa, psa, 0x01);
    const auto flippedMin = _mm256_permute2x128_si256(min, min, 0x01);

    // ok, now we add these together.
    const auto finalPsa = _mm256_add_epi8(flippedPSA, psa);
    const auto finalMin = _mm256_add_epi8(flippedMin, psa);

    // ok, so now we just have to compare.
    const auto totalMin = _mm256_min_epi8(min, finalMin);

    // We have them in 15 now, but we need them lower to find this.
    auto extendedMin = _mm256_cvtepi8_epi32(_mm256_castsi256_si128(_mm256_bsrli_epi128(totalMin, 15)));
    auto extendedPsa = _mm256_cvtepi8_epi32(_mm256_castsi256_si128(_mm256_bsrli_epi128(finalPsa, 15)));

    // Either prevPsa + curMin is a new minimum, or it's not.
    runningMin = _mm256_min_epi32(runningMin, _mm256_add_epi32(runningSum, extendedMin));
    runningSum = _mm256_add_epi32(runningSum, extendedPsa);
  }

  std::int32_t scalarMin = _mm256_extract_epi32(runningMin, 0);
  std::int32_t scalarSum = _mm256_extract_epi32(runningSum, 0);

  for (; i < N; i++) {
    if (inputString[i] == '(') {
      scalarSum++;
    } else {
      scalarSum--;
      scalarMin = std::min(scalarMin, scalarSum);
    }
  }

  return scalarSum - 2 * scalarMin;
}

inline int
solveSIMD_AVX2_v4(std::string_view inputString)
{
  // In this one, we drop a bunch of the manipulation, because it's really not needed I realize now.
  const int N = inputString.size();

  const auto ALL_ZERO = _mm256_setzero_si256();
  const auto ALL_ONE = _mm256_set1_epi8(0x01);
  const auto ALL_SET = _mm256_set1_epi8(0xFF);

  auto ans = _mm256_setzero_si256();
  auto balance = _mm256_setzero_si256();

  int i = 0;
  for (; i + 32 <= N; i += 32) {
    const auto chunk = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(inputString.data() + i));
    const auto leftBraces = _mm256_sub_epi8(chunk, _mm256_set1_epi8(')'));
    const auto rightBraces = _mm256_andnot_si256(leftBraces, ALL_SET);
    const auto valueChunk = _mm256_or_si256(rightBraces, ALL_ONE);

    auto psa = _mm256_add_epi8(valueChunk, _mm256_bslli_epi128(valueChunk, 1));
    psa = _mm256_add_epi8(psa, _mm256_bslli_epi128(psa, 2));
    psa = _mm256_add_epi8(psa, _mm256_bslli_epi128(psa, 4));
    psa = _mm256_add_epi8(psa, _mm256_bslli_epi128(psa, 8));

    auto min = _mm256_min_epi8(psa, _mm256_bslli_epi128(psa, 1));
    min = _mm256_min_epi8(min, _mm256_bslli_epi128(min, 2));
    min = _mm256_min_epi8(min, _mm256_bslli_epi128(min, 4));
    min = _mm256_min_epi8(min, _mm256_bslli_epi128(min, 8));

    // ok, let's swap min and psa.
    const auto flippedPSA = _mm256_permute2x128_si256(psa, psa, 0x01);
    const auto flippedMin = _mm256_permute2x128_si256(min, min, 0x01);

    // ok, now we add these together.
    const auto finalPsa = _mm256_add_epi8(flippedPSA, psa);
    const auto finalMin = _mm256_add_epi8(flippedMin, psa);

    // ok, so now we just have to compare.
    const auto totalMin = _mm256_min_epi8(min, finalMin);
    const auto negMin = _mm256_sub_epi8(ALL_ZERO, totalMin);

    // now then, we are going to revesre the totalMin, since we need that.
    // ok, so this is nice, we can actually get both of them like we want by shifting 15 to the right.
    const auto onlyPSA = _mm256_bsrli_epi128(finalPsa, 15);
    const auto onlyMin = _mm256_bsrli_epi128(negMin, 15);

    // Now then, we are going to convert this.
    const auto lowOnlyPSA = _mm256_castsi256_si128(onlyPSA);
    const auto lowOnlyMin = _mm256_castsi256_si128(onlyMin);

    // We need to use 32, because there there is no max_epi64
    const auto onlyPSA_32 = _mm256_cvtepi8_epi32(lowOnlyPSA);
    const auto onlyMin_32 = _mm256_cvtepi8_epi32(lowOnlyMin);

    const auto padding = _mm256_max_epi32(_mm256_sub_epi32(onlyMin_32, balance), ALL_ZERO);
    const auto tempAdd = _mm256_add_epi32(padding, onlyPSA_32);

    balance = _mm256_add_epi32(balance, tempAdd);
    ans = _mm256_add_epi32(ans, padding);
  }

  int scalarBalance = _mm256_cvtsi256_si32(balance);
  int scalarAns = _mm256_cvtsi256_si32(ans);

  for (; i < N; i++) {
    if (inputString[i] == '(') {
      scalarBalance++;
    } else {
      scalarBalance--;
      if (scalarBalance < 0) {
        scalarAns++;
        scalarBalance = 0;
      }
    }
  }

  return scalarAns + scalarBalance;
}

inline int
solveSIMD_AVX2_v1(std::string_view inputString)
{
  // we just knnow that these are good.
  const int N = inputString.size();

  int ans = 0;
  int balance = 0;

  const auto ALL_ONE = _mm256_set1_epi8(0x01);
  const auto ALL_SET = _mm256_set1_epi8(0xFF);

  int i = 0;
  for (; i + 32 <= N; i += 32) {
    const auto chunk = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(inputString.data() + i));
    const auto leftBraces = _mm256_sub_epi8(chunk, _mm256_set1_epi8(')'));
    const auto rightBraces = _mm256_andnot_si256(leftBraces, ALL_SET);
    const auto valueChunk = _mm256_or_si256(rightBraces, ALL_ONE);

    auto psa = _mm256_add_epi8(valueChunk, _mm256_bslli_epi128(valueChunk, 1));
    psa = _mm256_add_epi8(psa, _mm256_bslli_epi128(psa, 2));
    psa = _mm256_add_epi8(psa, _mm256_bslli_epi128(psa, 4));
    psa = _mm256_add_epi8(psa, _mm256_bslli_epi128(psa, 8));

    // Now if I have more than 16 in balance,  I can skip the next bit,
    // but I don't, since it's better to just pipeline everything.
    auto min = _mm256_min_epi8(psa, _mm256_bslli_epi128(psa, 1));
    min = _mm256_min_epi8(min, _mm256_bslli_epi128(min, 2));
    min = _mm256_min_epi8(min, _mm256_bslli_epi128(min, 4));
    min = _mm256_min_epi8(min, _mm256_bslli_epi128(min, 8));

    const std::int8_t finPsa1 = static_cast<std::int8_t>(_mm256_extract_epi8(psa, 15));
    const std::int8_t minVal1 = -static_cast<std::int8_t>(_mm256_extract_epi8(min, 15));

    const std::int8_t finPsa2 = static_cast<std::int8_t>(_mm256_extract_epi8(psa, 31));
    const std::int8_t minVal2 = -static_cast<std::int8_t>(_mm256_extract_epi8(min, 31));

    const int padding1 = std::max(0, minVal1 - balance);
    balance += finPsa1 + padding1;
    ans += padding1;

    const int padding2 = std::max(0, minVal2 - balance);
    balance += finPsa2 + padding2;
    ans += padding2;
  }

  for (; i < N; i++) {
    if (inputString[i] == '(') {
      balance++;
    } else {
      balance--;
      if (balance < 0) {
        ans++;
        balance = 0;
      }
    }
  }

  return ans + balance;
}

inline int
solveSIMD_AVX2_v2(std::string_view inputString)
{
  // we just knnow that these are good.
  const int N = inputString.size();

  int ans = 0;
  int balance = 0;

  const auto ALL_ONE = _mm256_set1_epi8(0x01);
  const auto ALL_SET = _mm256_set1_epi8(0xFF);

  int i = 0;
  for (; i + 32 <= N; i += 32) {
    const auto chunk = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(inputString.data() + i));
    const auto leftBraces = _mm256_sub_epi8(chunk, _mm256_set1_epi8(')'));
    const auto rightBraces = _mm256_andnot_si256(leftBraces, ALL_SET);
    const auto valueChunk = _mm256_or_si256(rightBraces, ALL_ONE);

    auto psa = _mm256_add_epi8(valueChunk, _mm256_bslli_epi128(valueChunk, 1));
    psa = _mm256_add_epi8(psa, _mm256_bslli_epi128(psa, 2));
    psa = _mm256_add_epi8(psa, _mm256_bslli_epi128(psa, 4));
    psa = _mm256_add_epi8(psa, _mm256_bslli_epi128(psa, 8));

    // Now if I have more than 16 in balance,  I can skip the next bit,
    // but I don't, since it's better to just pipeline everything.
    auto min = _mm256_min_epi8(psa, _mm256_bslli_epi128(psa, 1));
    min = _mm256_min_epi8(min, _mm256_bslli_epi128(min, 2));
    min = _mm256_min_epi8(min, _mm256_bslli_epi128(min, 4));
    min = _mm256_min_epi8(min, _mm256_bslli_epi128(min, 8));

    // We are going to create our favroite now.
    const auto shuffleMask = _mm256_set_m128i(_mm_set1_epi8(0xFF), _mm_set1_epi8(0x0F));
    const auto shuffledRev = _mm256_shuffle_epi8(psa, shuffleMask);

    const auto shuffledNorm = _mm256_permute4x64_epi64(shuffledRev, 0b00'01'10'11);

    const auto finalPsa = _mm256_add_epi8(psa, shuffledNorm);
    const auto finalMin = _mm256_add_epi8(min, shuffledNorm);

    // ok, all minimums are now in the lower part.
    const auto shuffledMin = _mm256_permute4x64_epi64(finalMin, 0b00'10'11'01);
    const auto totalMin = _mm256_min_epi8(_mm256_bsrli_epi128(shuffledMin, 15), _mm256_bsrli_epi128(shuffledMin, 7));

    const std::int8_t finPsa = static_cast<std::int8_t>(_mm256_extract_epi8(finalPsa, 31));
    const std::int8_t minVal = -static_cast<std::int8_t>(_mm256_extract_epi8(totalMin, 0));

    const int padding = std::max(0, minVal - balance);
    balance += finPsa + padding;
    ans += padding;
  }

  for (; i < N; i++) {
    if (inputString[i] == '(') {
      balance++;
    } else {
      balance--;
      if (balance < 0) {
        ans++;
        balance = 0;
      }
    }
  }

  return ans + balance;
}

inline int
solveSIMD_AVX2_v3(std::string_view inputString)
{
  // In this one, we drop a bunch of the manipulation, because it's really not needed I realize now.
  const int N = inputString.size();

  const auto ALL_ZERO = _mm256_setzero_si256();
  const auto ALL_ONE = _mm256_set1_epi8(0x01);
  const auto ALL_SET = _mm256_set1_epi8(0xFF);

  int ans = 0;
  int balance = 0;

  int i = 0;
  for (; i + 32 <= N; i += 32) {
    const auto chunk = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(inputString.data() + i));
    const auto leftBraces = _mm256_sub_epi8(chunk, _mm256_set1_epi8(')'));
    const auto rightBraces = _mm256_andnot_si256(leftBraces, ALL_SET);
    const auto valueChunk = _mm256_or_si256(rightBraces, ALL_ONE);

    auto psa = _mm256_add_epi8(valueChunk, _mm256_bslli_epi128(valueChunk, 1));
    psa = _mm256_add_epi8(psa, _mm256_bslli_epi128(psa, 2));
    psa = _mm256_add_epi8(psa, _mm256_bslli_epi128(psa, 4));
    psa = _mm256_add_epi8(psa, _mm256_bslli_epi128(psa, 8));

    auto min = _mm256_min_epi8(psa, _mm256_bslli_epi128(psa, 1));
    min = _mm256_min_epi8(min, _mm256_bslli_epi128(min, 2));
    min = _mm256_min_epi8(min, _mm256_bslli_epi128(min, 4));
    min = _mm256_min_epi8(min, _mm256_bslli_epi128(min, 8));

    // ok, let's swap min and psa.
    const auto flippedPSA = _mm256_permute2x128_si256(psa, psa, 0x01);
    const auto flippedMin = _mm256_permute2x128_si256(min, min, 0x01);

    // ok, now we add these together.
    const auto finalPsa = _mm256_add_epi8(flippedPSA, psa);
    const auto finalMin = _mm256_add_epi8(flippedMin, psa);

    // ok, so now we just have to compare.
    const auto totalMin = _mm256_min_epi8(min, finalMin);
    const auto negMin = _mm256_sub_epi8(ALL_ZERO, totalMin);

    // now then, we are going to revesre the totalMin, since we need that.

    const std::int8_t finPsa = static_cast<std::int8_t>(_mm256_extract_epi8(finalPsa, 15));
    const std::int8_t minVal = static_cast<std::int8_t>(_mm256_extract_epi8(negMin, 15));

    const int padding = std::max(0, minVal - balance);
    balance += finPsa + padding;
    ans += padding;
  }

  for (; i < N; i++) {
    if (inputString[i] == '(') {
      balance++;
    } else {
      balance--;
      if (balance < 0) {
        ans++;
        balance = 0;
      }
    }
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
