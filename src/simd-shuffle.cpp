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

template<std::size_t N>
std::string
reverseInt(const auto x, bool reverse = false, bool show16 = false)
{

  auto s = std::bitset<N>(x).to_string();
  if (reverse) {
    std::ranges::reverse(s);
  }

  if (show16) {
    std::array<char, 256> buf;
    auto [ptr, ec] = std::to_chars(buf.data(), buf.data() + buf.size(), x, 16);
    s += " 0x";
    s.append(N / 4 - (ptr - buf.data()), '0');
    s.append(buf.data(), ptr);
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

  // All of these needs to be xored with this 0x80, to remove the upper spot
  const __m128i XORP = _mm_set1_epi8(0x80);
  const std::uint64_t XORR = 0x8080808080808080;

  int ans = 0;
  for (int i = 0; i < static_cast<int>(buf.size()); i += 16) {
    const __m128i input = _mm_loadu_si128(reinterpret_cast<const __m128i*>(buf.data() + i));
    const auto shiftedInput = _mm_bslli_si128(input, 1);

    // Let's build the number we want.

    // Now then, how are we to endure this? Well, we need to compare them as 16 bit ints,
    // as we want it all to match or nothing.

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

    const std::uint16_t hitMask = ~_mm_movemask_epi8(matchedF);

    // Unpack each bit into a byte, we need 2 vectors for this, since the we have 128bits of info
    // Then we multiply by 0xFF, to convert from 01 to FF for each seperate byte.
    const std::uint64_t lowerMask = _pdep_u64(hitMask, 0x0101010101010101) * 0xFF;
    const std::uint64_t upperMask = _pdep_u64(hitMask >> 8, 0x0101010101010101) * 0xFF;

    const std::uint16_t lowerHits = std::popcount(lowerMask) / 8;
    /* const std::uint16_t lowerHits = 8; // std::popcount(lowerMask) / 8; */
    const std::uint16_t upperHits = std::popcount(upperMask) / 8;

    // Now we can create the indicies we want.
    // We could add 0x80 to all, and xor with 0x80 to get just the valid ones.
    const std::uint64_t lowerIndices = 0x8786858483828180;
    const std::uint64_t upperIndices = 0x8F8E8D8C8B8A8988;

    // ok, we want to overwrite the upperIndicies,

    // ok, now we are going to create the two parts of the mask.
    const std::uint64_t lowerPacked = _pext_u64(lowerIndices, lowerMask);
    const std::uint64_t upperPacked = _pext_u64(upperIndices, upperMask);

    const std::uint64_t lowerFixedPacked = lowerPacked | (upperPacked << (8 * lowerHits));
    const std::uint64_t upperFixedPacked = upperPacked >> (64 - (8 * lowerHits));

    // OK, for now let's just do this
    const auto permuteMask = _mm_set_epi64x(upperFixedPacked, lowerFixedPacked);
    const auto finalMask = _mm_xor_si128(permuteMask, XORP);

    const auto afterShuffle = _mm_shuffle_epi8(input, finalMask);

    // I want to use the
    const auto filledIn = _mm_blendv_epi8(afterShuffle, _mm_set1_epi8('.'), finalMask);

    std::array<char, 16> outputBuf;
    _mm_storeu_si128(reinterpret_cast<__m128i*>(outputBuf.data()), filledIn);

    ans += outputBuf[0];

#ifdef VERBOSE
    std::cout << "Hits:  [" << reverseInt<16>(hitMask) << "]\n";
    /* std::cout << "Upper hits: " << upperHits << ", lowerHits: " << lowerHits << "\n"; */

    /* std::cout << "Lower: [" << reverseInt<64>(lowerMask) << "]\n"; */
    /* std::cout << "Upper: [" << reverseInt<64>(upperMask) << "]\n"; */

    /* std::cout << "LI:    [" << reverseInt<64>(lowerIndices) << "]\n"; */
    /* std::cout << "UP:    [" << reverseInt<64>(upperIndices[lowerHits]) << "]\n"; */

    /* std::cout << "\n"; */
    /* std::cout << "LOwer: [" << reverseInt<64>(lowerPacked) << "]\n"; */
    /* std::cout << "LUPer: [" << reverseInt<64>(upperPacked << (8 * lowerHits)) << "]\n"; */

    /* std::cout << "\n"; */

    std::cout << "LoweF: [" << reverseInt<64>(lowerFixedPacked ^ XORR, false, true) << "]\n";
    std::cout << "UPper: [" << reverseInt<64>(upperFixedPacked ^ XORR, false, true) << "]\n";

    std::cout << "\n";

    std::cout << "Input:  [" << buf << "]\n";
    std::cout << "Output: [" << std::string_view(outputBuf.begin(), outputBuf.end()) << "]\n";
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
