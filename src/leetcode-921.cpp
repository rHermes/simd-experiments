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
}

int
main()
{
  ankerl::nanobench::Bench b;

  /* b.title("Leetcode 921 solving").warmup(10).minEpochIterations(1000).relative(true); */
  b.title("Leetcode 921 solving").warmup(10).relative(true);

  ankerl::nanobench::Rng rng(10);
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
