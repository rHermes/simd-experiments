#include <rimd/all.hpp>

#include <cstdint>
#include <immintrin.h>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

/*
 * This file is about trying to understand the shuffle instructions in
 * the SIMD instruction set
 */

#define VERBOSE 0
/* #define ALT 0 */

FORCE_INLINE __m128i
compressInput(const __m128i input, const __m128i mask, std::uint16_t& totalHits)
{
  const std::uint16_t hitMask = static_cast<std::uint16_t>(_mm_movemask_epi8(mask));

  // First we spread the 16 mask values into nibles, 4 bits
  const std::uint64_t hitMask4 = _pdep_u64(hitMask, 0x1111111111111111) * 0x0F;

  // Extract the indices into the left side
  const std::uint64_t indices = _pext_u64(0xFEDCBA9876543210, hitMask4);

  // Now we expand them out again, into bytes
  const std::uint64_t highMask = _pdep_u64(indices >> 32ULL, 0x0F0F0F0F0F0F0F0F);
  const std::uint64_t lowMask = _pdep_u64(indices, 0x0F0F0F0F0F0F0F0F);

  // The problem now is just that we need to know which
  const auto combinedMask = _mm_set_epi64x(highMask, lowMask);
  const auto compressedInput = _mm_shuffle_epi8(input, combinedMask);

  totalHits = static_cast<std::uint16_t>(_mm_popcnt_u64(hitMask));

  return compressedInput;
}

// This is a test to see how I might do leetcode 2696.
int
leetcode_2696_test(std::span<char> buf)
{
  using namespace std::literals;

  const rimd::Uint8x16 PAT1 =
    _mm_setr_epi8('A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B');
  const rimd::Uint8x16 PAT2 =
    _mm_setr_epi8('C', 'D', 'C', 'D', 'C', 'D', 'C', 'D', 'C', 'D', 'C', 'D', 'C', 'D', 'C', 'D');

  // ok, there are a couple of ways we can do this, but I think the best
  // would be to do 4 comparisons. We could also just search 4 times for
  // a, b, c and D and combine those, but it would be equally many searches, for not much gain.
  const auto ALL_ONES = rimd::Uint8x16(0xFF);

  // Ok, time to try to be smart here. What we will be doing, is that we will always operate on 4 bytes at a time.
  // This is not really

  // We can try to read this from the beginning?
  const int N = static_cast<int>(buf.size());

  int outLen = N;

  while (true) {
    auto outPtr = buf.data();
#ifdef VERBOSE
    std::cout << "Phase input:  [" << std::string_view(buf.data(), outLen) << "]\n";
#endif

    int i;
    for (i = 0; i < outLen - 16; i += 16) {
      const auto input = rimd::Uint8x16::LoadUnaligned(buf.data() + i);
      const auto shiftedInput = input.shiftLanesLeft<1>();

#ifdef VERBOSE
      /* std::cout << "Input:  [" << std::string_view(buf.data() + i, 16) << "]\n"; */
#endif

      // Aligned matches
      const auto match1 = _mm_cmpeq_epi16(input, PAT1);
      const auto match2 = _mm_cmpeq_epi16(input, PAT2);

      // Non aligned matches.
      const auto match3 = _mm_cmpeq_epi16(shiftedInput, PAT1);
      const auto match4 = _mm_cmpeq_epi16(shiftedInput, PAT2);

      // OK, now we are going to go back to treating it as a z3 again.
      auto matched1 = _mm_or_si128(match1, match2);
      auto matched2 = _mm_or_si128(match3, match4);

      // We have to shift the non aligned input back.
      auto matchedF = _mm_or_si128(matched1, _mm_bsrli_si128(matched2, 1));
      auto invertedMatchedF = _mm_andnot_si128(matchedF, ALL_ONES);

      std::uint16_t totalHits;
      auto compressedInput = compressInput(input, invertedMatchedF, totalHits);

      /* std::array<char, 16> outputBuf; */
      _mm_storeu_si128(reinterpret_cast<__m128i*>(outPtr), compressedInput);
#ifdef VERBOSE
      /* std::cout << "Output: [" << std::string_view(outPtr, totalHits) << "]\n"; */
#endif
      outPtr += totalHits;
    }

    for (; i < outLen - 1; i++) {
      const auto c1 = buf[i];
      const auto c2 = buf[i + 1];
      if (c1 == 'A' && c2 == 'B') {
        i++;
      } else if (c1 == 'C' && c2 == 'D') {
        i++;
      } else {
        *(outPtr++) = c1;
      }
    }

    if (i < outLen) {
      *(outPtr++) = buf[i];
    }

#ifdef VERBOSE
    std::cout << "Phase output: [" << std::string_view(buf.data(), outPtr) << "]\n\n";
#endif

    int newLen = static_cast<int>(outPtr - buf.data());
    if (newLen == outLen) {
      break;
    }

    outLen = newLen;
  }

#ifdef VERBOSE
  std::cout << "Total output: [" << std::string_view(buf.data(), outLen) << "]\n";
#endif

  // ok, we are going to be
  return outLen;
}

int
simpleSolve(std::span<const char> s)
{
  std::string st = ".";
  int ans = 0;

  const int N = s.size();
  for (int i = 0; i < N; i++) {
    const char c = s[i];
    if (c == 'A' || c == 'C') {
      st.push_back(c);
    } else if (c == 'B' && st.back() == 'A') {
      st.pop_back();
    } else if (c == 'D' && st.back() == 'C') {
      st.pop_back();
    } else [[likely]] {
      ans += st.size();
      st.resize(1);
    }
  }

  return st.size() - 1 + ans;
}

int
main()
{
  using namespace std::literals;
  /* shuffle_chars(); */
  std::vector<std::string> testCases = {
    "ABFCACDB",
    "KLPQIGVASABCBPZAACACACACDBDBDBDBBGDABFAABBGXR",
  };

  for (const auto& tc : testCases) {
    auto expected = simpleSolve({ tc.begin(), tc.end() });

    std::string mine = tc;
    auto comp = leetcode_2696_test({ mine.begin(), mine.end() });
    if (expected != comp) {
      std::cout << "We expected: " << expected << ", but got: " << comp << " for: [" << tc << "]\n";
    }
  }

  /* leetcode_2696_test("0AB3456789ABEFCD"sv); */
  return 0;
}
