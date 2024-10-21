// The point of this file is to solve 2938.

#include <rimd/all.hpp>

#include <array>
#include <print>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <immintrin.h>

#include <nanobench.h>

namespace p67 {

inline std::string
solveScalar(std::string_view input1, std::string_view input2);

inline std::string
solveSIMD_SSE4_v1(std::string_view input1, std::string_view input2);

inline std::string
solveSIMD_AVX2_v1(std::string_view input1, std::string_view input2);

inline std::string
solveSIMD_SSE4(std::string_view input1, std::string_view input2)
{
  return solveSIMD_SSE4_v1(input1, input2);
};

inline std::string
solveSIMD_AVX2(std::string_view input1, std::string_view input2)
{
  // Until later, when we implement this:
  return solveScalar(input1, input2);
  // return solveSIMD_AVX2_v1(input1, input2);
}

template<typename Gen>
int
generateInputBernoulli(Gen&& gen, std::span<char> output, const float p)
{
  std::bernoulli_distribution dist(p);
  int outIdx = 0;
  for (int i = 0; i < output.size(); i++) {
    bool putOne = dist(gen);
    if (putOne || outIdx != 0) {
      output[outIdx++] = putOne ? '1' : '0';
    }
  }

  if (outIdx == 0) {
    output[outIdx++] = '0';
  }

  return outIdx;
}

template<typename Gen>
std::vector<std::pair<std::string, std::string>>
generateBernoulliTestset(Gen&& gen, const std::size_t size, const std::size_t inputLen, const float p)
{
  std::vector<std::pair<std::string, std::string>> strings(size);
  for (auto& [s1, s2] : strings) {
    s1.resize_and_overwrite(
      inputLen, [&](char* ptr, size_t sz) { return generateInputBernoulli(gen, std::span<char>(ptr, inputLen), p); });
    s2.resize_and_overwrite(
      inputLen, [&](char* ptr, size_t sz) { return generateInputBernoulli(gen, std::span<char>(ptr, inputLen), p); });
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
      for (const auto& [s1, s2] : testSet) {
        const auto r = solver(s1, s2);
        ankerl::nanobench::doNotOptimizeAway(r);
      }
    });
  };

  b.relative(true);

  runTest("Scalar", solveScalar);

  runTest("SSE4_v1", solveSIMD_SSE4_v1);

  // runTest("AVX2_v1", solveSIMD_AVX2_v1);
}

template<typename Rng>
bool
sanityCheck(Rng&& rng)
{
  // const auto testSet = generateBernoulliTestset(rng, 100, 100, 0.5);
  // const auto testSet = generateBernoulliTestset(rng, 100, 100, 0.3);
  const auto testSet = generateBernoulliTestset(rng, 100, 25, 0.3);
  // const auto testSet = generateBernoulliTestset(rng, 100, 65, 0.5);
  // const auto testSet = generateBernoulliTestset(rng, 100, 32, 0.5);
  // const auto testSet = generateBernoulliTestset(rng, 100, 14, 0.5);

  int i = 0;
  for (const auto& [tc1, tc2] : testSet) {
    i++;
    const auto expected = solveScalar(tc1, tc2);
    const auto sse4 = solveSIMD_SSE4(tc1, tc2);
    const auto avx2 = solveSIMD_AVX2(tc1, tc2);

    if (sse4 != expected && avx2 != expected) {
      std::print("We expected {}, but sse4 got: {}, and avx2: {}\n", expected, sse4, avx2);
      return false;
    }

    if (sse4 != expected) {
      std::print("A: {:>26}\nB: {:>26}\nC: {:>26}\nD: {:>26}\n", tc1, tc2, sse4, expected);
      std::print("We expected {}, but sse4 got: {} on test case {}\n", expected, sse4, i);
      return false;
    }

    if (avx2 != expected) {
      std::print("We expected {}, but avx2 got: {}\n", expected, avx2);
      return false;
    }
  }

  return true;
}

std::string
solveScalar(std::string_view inputString1, std::string_view inputString2)
{
  if (inputString1.size() < inputString2.size())
    std::swap(inputString1, inputString2);

  const int N1 = inputString1.size();
  const int N2 = inputString2.size();

  std::string out;
  out.resize_and_overwrite(N1 + 1, [&](char* const ptr, size_t sz) {
    // ok now then how do I do this. How do I know if it is going to overflow? Answer is I really don't, so we just
    // write this in reverse and hope for the best.
    bool carry = false;

    auto readPtr1 = inputString1.data() + N1 - 1;
    auto readPtr2 = inputString2.data() + N2 - 1;

    auto endptr = ptr + N1;
    // This is just a binary carry adder.
    while (inputString2.data() <= readPtr2) {
      const bool b1 = *readPtr1-- == '1';
      const bool b2 = *readPtr2-- == '1';
      const bool thisOut = (b1 != b2) != carry;

      *endptr-- = thisOut ? '1' : '0';

      carry = (b1 && b2) || ((b1 != b2) && carry);
    }

    // now then we just process the s1 string.
    while (inputString1.data() <= readPtr1) {
      const bool b1 = *readPtr1-- == '1';
      const bool thisOut = b1 != carry;
      *endptr-- = thisOut ? '1' : '0';
      carry = b1 && carry;
    }

    if (carry) {
      // We add a one then reverse, otherwise we don't
      *endptr-- = '1';
      return N1 + 1;
    } else {
      std::ranges::copy(ptr + 1, ptr + N1 + 1, ptr);
      return N1;
    }
  });

  return out;
}

std::string
solveSIMD_SSE4_v1(std::string_view input1, std::string_view input2)
{
  if (input1.size() < input2.size())
    std::swap(input1, input2);

  const int N1 = input1.size();
  const int N2 = input2.size();

  std::string out;
  out.resize_and_overwrite(N1 + 1, [&](char* const ptr, size_t sz) {
    // Ok, so we are reading from the back here.
    auto readPtr1 = input1.data() + N1;
    auto readPtr2 = input2.data() + N2;

    const rimd::Uint8x16 SIMD_MASK(79);
    const rimd::Uint8x16 ASCII_ZERO('0');
    const rimd::Uint8x16 REVERSE_SHUFFLE_MASK(0x0001020304050607ull, 0x08090A0B0C0D0E0Full);

    unsigned char carry = 0;

    auto writePtr = ptr + N1 + 1;

    while (input2.data() <= readPtr2 - 16) {
      readPtr1 -= 16;
      readPtr2 -= 16;
      writePtr -= 16;
      const auto chunk1 = rimd::Uint8x16::LoadUnaligned(readPtr1);
      const auto chunk2 = rimd::Uint8x16::LoadUnaligned(readPtr2);

      // It's very unfortunate that we are reading backwards, because it affects how we are building our movemask, so
      // we need to sbuffle these to reverse them.
      const auto revChunk1 = _mm_shuffle_epi8(chunk1, REVERSE_SHUFFLE_MASK);
      const auto revChunk2 = _mm_shuffle_epi8(chunk2, REVERSE_SHUFFLE_MASK);

      // This is dirty as fuck, but found with z3. We just need to set the most siginifcant bit, so no need to do
      // all the comparisons, we simply add by 79, which brings '1' over the edge, but not '0'.
      const auto values1 = _mm_add_epi8(revChunk1, SIMD_MASK);
      const auto values2 = _mm_add_epi8(revChunk2, SIMD_MASK);

      // Ok, now we create our movemasks.
      const std::uint32_t mm1 = _mm_movemask_epi8(values1);
      const std::uint32_t mm2 = _mm_movemask_epi8(values2);

      std::uint32_t res;
      _addcarryx_u32(carry, mm1, mm2, &res);
      carry = res >> 16;

      // ok, now need to rewrite stuff.
      const auto outLower = _pdep_u64(res, 0x0101010101010101);
      const auto outUpper = _pdep_u64(res >> 8, 0x0101010101010101);
      const rimd::Uint8x16 result(outUpper, outLower);

      // ok now we just gotta OR it with 0
      const rimd::Uint8x16 charsOut = _mm_or_si128(result, ASCII_ZERO);
      const rimd::Uint8x16 revChars = _mm_shuffle_epi8(charsOut, REVERSE_SHUFFLE_MASK);

      revChars.writeUnaligned(writePtr);
      // Now we just have to write thems
    }

    // std::print("WTF: {}\n", std::string_view{writePtr, ptr+N1+1});

    while (input2.data() < readPtr2) {
      const bool b1 = *--readPtr1 == '1';
      const bool b2 = *--readPtr2 == '1';
      const bool thisOut = (b1 != b2) != carry;

      *--writePtr = thisOut ? '1' : '0';

      carry = (b1 && b2) || ((b1 != b2) && carry);
    }

    // now then we just process the s1 string.
    while (input1.data() < readPtr1) {
      const bool b1 = *--readPtr1 == '1';
      const bool thisOut = b1 != carry;
      *--writePtr = thisOut ? '1' : '0';
      carry = b1 && carry;
    }

    if (carry) {
      // We add a one then reverse, otherwise we don't
      *--writePtr = '1';
      return N1 + 1;
    } else {
      std::ranges::copy(ptr + 1, ptr + N1 + 1, ptr);
      return N1;
    }
  });

  return out;
}

std::string
solveSIMD_AVX2_v1(std::string_view input1, std::string_view input2)
{
  return "";
}

} // namespace p67

int
main()
{
  ankerl::nanobench::Rng rng(10);

  p67::sanityCheck(rng);



  ankerl::nanobench::Bench b;

  p67::runBernoulliTest(rng, "Short 50% strings", 1000, 10, 0.5);
  p67::runBernoulliTest(rng, "Mid 50% strings", 1000, 100, 0.5);
  p67::runBernoulliTest(rng, "Long 50% strings", 1000, 1000, 0.5);
  p67::runBernoulliTest(rng, "Very long 50% strings", 1000, 10000, 0.5);


  return 0;
}