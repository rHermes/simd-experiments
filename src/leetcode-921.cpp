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
solveSIMD_AVX2(std::string_view input);

inline int
solveSIMD_AVX2_v2(std::string_view input);

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

  b.title(title).warmup(10).relative(true);

  b.run("Normal", [&]() {
    for (const auto& s : testSet) {
      const auto r = solveNormal(s);
      ankerl::nanobench::doNotOptimizeAway(r);
    }
  });

  b.run("SSE4", [&]() {
    for (const auto& s : testSet) {
      const auto r = solveSIMD_SSE4(s);
      ankerl::nanobench::doNotOptimizeAway(r);
    }
  });

  b.run("AVX2", [&]() {
    for (const auto& s : testSet) {
      const auto r = solveSIMD_AVX2(s);
      ankerl::nanobench::doNotOptimizeAway(r);
    }
  });

  b.run("AVX2_V2", [&]() {
    for (const auto& s : testSet) {
      const auto r = solveSIMD_AVX2_v2(s);
      ankerl::nanobench::doNotOptimizeAway(r);
    }
  });
}

template<typename Rng>
bool
sanityCheck(Rng&& rng)
{
  const auto testSet = generateBernoulliTestset(rng, 100, 100, 0.5);

  for (const auto& tc : testSet) {
    const auto expected = solveNormal(tc);
    const auto sse4 = solveSIMD_SSE4(tc);
    const auto avx2 = solveSIMD_AVX2_v2(tc);

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

  /* b.title("Leetcode 921 solving").warmup(10).minEpochIterations(1000).relative(true); */
  /* b.title("Leetcode 921 solving").warmup(10).relative(true); */

  runBernoulliTest(rng, "Short 50% strings", 1000, 10, 0.5);
  runBernoulliTest(rng, "Mid 50% strings", 1000, 100, 0.5);
  runBernoulliTest(rng, "Long 50% strings", 1000, 1000, 0.5);

  return 0;
}

inline int
solveSIMD_SSE4(std::string_view inputString)
{
  // we just knnow that these are good.
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
solveSIMD_AVX2(std::string_view inputString)
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
		// This is now all the psa in the first, we just need to 
		//


		/* const auto shuffledNorm = _mm256_permute2x128_si256(shuffledRev, _mm256_setzero_si256(), 0x01); */
		const auto shuffledNorm = _mm256_permute4x64_epi64(shuffledRev, 0b00'01'10'11);


		const auto finalPsa = _mm256_add_epi8(psa, shuffledNorm);
		const auto finalMin = _mm256_add_epi8(min, shuffledNorm);

		/* const auto maxedMin = _mm256_shuffle_epi8(finalMin, _mm256_set1_epi8(0x0F)); */
	
		/* std::cout << "GRID:  [" << std::string_view(inputString.data() + i, 16) << " " << std::string_view(inputString.data()+ i + 16, 16) << "]\n"; */
		/* std::cout << "PSA: [" << m256toHex(psa, true) << "]\n"; */
		/* std::cout << "FSA: [" << m256toHex(finalPsa, true) << "]\n"; */

		// ok, all minimums are now in the lower part.
		const auto shuffledMin = _mm256_permute4x64_epi64(finalMin, 0b00'10'11'01);
		const auto totalMin = _mm256_min_epi8(_mm256_bsrli_epi128(shuffledMin, 15), _mm256_bsrli_epi128(shuffledMin, 7));

		/* std::cout << "PRE MIN:      [" << m256toHex(min, true) << "]\n"; */
		/* std::cout << "FINAL MIN:    [" << m256toHex(finalMin, true) << "]\n"; */
		/* std::cout << "MAXED MIN:    [" << m256toHex(maxedMin, true) << "]\n"; */
		/* std::cout << "SHUFFLED MIN: [" << m256toHex(shuffledMin, true) << "]\n"; */
		/* std::cout << "TOTAL MIN:    [" << m256toHex(totalMin, true) << "]\n"; */
		

    const std::int8_t finPsa = static_cast<std::int8_t>(_mm256_extract_epi8(finalPsa, 31));
    const std::int8_t minVal = -static_cast<std::int8_t>(_mm256_extract_epi8(totalMin, 0));

		/* std::cout << "final PSA: " << static_cast<int>(finPsa) << ", minVal: " << static_cast<int>(minVal) << "\n"; */

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
