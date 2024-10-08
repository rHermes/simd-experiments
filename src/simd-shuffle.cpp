#include <algorithm>
#include <array>
#include <bitset>
#include <charconv>
#include <cstdint>
#include <immintrin.h>
#include <iostream>
#include <numeric>
#include <string>
#include <string_view>

/*
 * This file is about trying to understand the shuffle instructions in
 * the SIMD instruction set
 */

#define VERBOSE 0
/* #define ALT 0 */

template<std::size_t N>
void
appendHexToString(std::string& str, const auto x, bool reverse = false)
{
  std::array<char, 256> buf;
  auto [ptr, ec] = std::to_chars(buf.data(), buf.data() + buf.size(), x, 16);
  str.append(N / 4 - (ptr - buf.data()), '0');
  str.append(buf.data(), ptr);
}

template<std::size_t N>
std::string
reverseInt(const auto x, bool reverse = false, bool show16 = false)
{

  auto s = std::bitset<N>(x).to_string();
  if (reverse) {
    std::ranges::reverse(s);
  }

  if (show16) {
    s += " 0x";
    appendHexToString<N>(s, x, reverse);
  }

  return s;
}

std::string
get128mm(const __m128i x, bool reverse = false, bool show16 = false)
{
	const std::uint64_t first = reverse ? _mm_extract_epi64(x, 0) : _mm_extract_epi64(x, 1);
	const std::uint64_t second = reverse ? _mm_extract_epi64(x, 1) : _mm_extract_epi64(x, 0);

  std::string s;
	s += reverseInt<64>(first);
	s += reverseInt<64>(second);

	if (show16) {
		s += " 0x";
		appendHexToString<64>(s, first, reverse);
		appendHexToString<64>(s, second, reverse);
	}

  return s;
}

// This is a test to see how I might do leetcode 2696.
int
leetcode_2696_test(const std::string_view buf)
{
  using namespace std::literals;

  // ok, there are a couple of ways we can do this, but I think the best
  // would be to do 4 comparisons. We could also just search 4 times for
  // a, b, c and D and combine those, but it would be equally many searches, for not much gain.
  const __m128i PAT1 = _mm_setr_epi8('A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B');
  const __m128i PAT2 = _mm_setr_epi8('C', 'D', 'C', 'D', 'C', 'D', 'C', 'D', 'C', 'D', 'C', 'D', 'C', 'D', 'C', 'D');
  const auto ALL_ONES = _mm_set1_epi8(0xFF);

  int ans = 0;
  for (int i = 0; i < static_cast<int>(buf.size()); i += 16) {
    const auto input = _mm_loadu_si128(reinterpret_cast<const __m128i*>(buf.data() + i));
    const auto shiftedInput = _mm_bslli_si128(input, 1);

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

    const std::uint16_t hitMask = _mm_movemask_epi8(invertedMatchedF);

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

    /* std::cout << "CombinedMask:   [" << get128mm(combinedMask, false, true) << "]\n"; */
    /* std::cout << "CompressedHits: [" << get128mm(compressedHits, false, true) << "]\n"; */

    /* const auto filledIn = _mm_blendv_epi8(_mm_set1_epi8('.'), compressedInput, compressedHits); */
    const auto filledIn =  compressedInput;

    const std::uint16_t totalHits = _mm_popcnt_u64(hitMask);

    std::array<char, 16> outputBuf;
    _mm_storeu_si128(reinterpret_cast<__m128i*>(outputBuf.data()), filledIn);

    ans += totalHits;
    ans += outputBuf[0];

#ifdef VERBOSE
    std::cout << "Input:  [" << buf << "]\n";
    std::cout << "Output: [" << std::string_view(outputBuf.begin(), totalHits) << "]\n";
#endif
  }

  // ok, we are going to be
  return ans;
}

void
shuffle_chars()
{
  using namespace std::literals;

  const auto buf = "0123456789ABCDEF"sv;
  const __m128i input = _mm_loadu_si128(reinterpret_cast<const __m128i*>(buf.data()));

  std::array<char, 16> mask{};
  std::iota(mask.rbegin(), mask.rend(), 0);

  mask[4] |= 1 << 7;
  /* std::cout << std::bitset<8>(mask[4]) << ", wow\n"; */

  const __m128i shuffleMask = _mm_loadu_si128(reinterpret_cast<const __m128i*>(mask.data()));

  const auto shuffled = _mm_shuffle_epi8(input, shuffleMask);
  const auto dots = _mm_set1_epi8('.');

  const auto ans = _mm_blendv_epi8(shuffled, dots, shuffleMask);

  std::array<char, 16> output;
  std::ranges::fill(output, ',');

  _mm_storeu_si128(reinterpret_cast<__m128i*>(output.data()), ans);

  std::cout << "input:  " << buf << "\n";
  std::cout << "output: " << std::string_view(output.begin(), output.end()) << "\n";
}

int
main()
{
  using namespace std::literals;
  /* shuffle_chars(); */
  leetcode_2696_test("0AB3456789ABEFCD"sv);
  return 0;
}
