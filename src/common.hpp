#pragma once

#include <algorithm>
#include <array>
#include <bitset>
#include <charconv>
#include <cstdint>
#include <string>

#include <immintrin.h>

#if defined(__clang__)
#define FORCE_INLINE [[gnu::always_inline]] [[gnu::gnu_inline]] extern inline

#elif defined(__GNUC__)
#define FORCE_INLINE [[gnu::always_inline]] inline

#elif defined(_MSC_VER)
#pragma warning(error : 4714)
#define FORCE_INLINE __forceinline

#else
#error Unsupported compiler
#endif

template<std::size_t N>
void
appendHexToString(std::string& str, const auto x, bool reverse = false)
{
  static_assert(N % 2 == 0);

  std::array<char, 256> buf;
  auto [ptr, ec] = std::to_chars(buf.data(), buf.data() + buf.size(), x, 16);
  for (auto beg = buf.data(); beg < ptr; beg++) {
    const auto c = *beg;
    if ('9' < c) {
      *beg -= 'a' - 'A';
    }
  }

  auto preSize = str.size();
  str.append(N / 4 - (ptr - buf.data()), '0');
  str.append(buf.data(), ptr);
  // ok, now we need to iterate over 2x of these and reverse it.
  if (reverse) {
    std::size_t l = preSize;
    std::size_t r = str.size() - 2;
    while (l < r) {
      std::swap(str[l], str[r]);
      std::swap(str[l + 1], str[r + 1]);
      l += 2;
      r -= 2;
    }
  }
}

template<std::size_t N>
std::string
toBinaryString(const auto x, bool reverse = false, bool show16 = false)
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

inline std::string
m128toHex(const __m128i x, bool reverse = false)
{
  const std::uint64_t first = reverse ? _mm_extract_epi64(x, 0) : _mm_extract_epi64(x, 1);
  const std::uint64_t second = reverse ? _mm_extract_epi64(x, 1) : _mm_extract_epi64(x, 0);

  std::string s;
  s += "0x";
  appendHexToString<64>(s, first, reverse);
  appendHexToString<64>(s, second, reverse);

  return s;
}

inline std::string
m256toHex(const __m256i x, bool reverse = false)
{
  const std::uint64_t first = reverse ? _mm256_extract_epi64(x, 0) : _mm256_extract_epi64(x, 3);
  const std::uint64_t second = reverse ? _mm256_extract_epi64(x, 1) : _mm256_extract_epi64(x, 2);
  const std::uint64_t third = reverse ? _mm256_extract_epi64(x, 2) : _mm256_extract_epi64(x, 1);
  const std::uint64_t fourth = reverse ? _mm256_extract_epi64(x, 3) : _mm256_extract_epi64(x, 0);

  std::string s;
  s += "0x";
  appendHexToString<64>(s, first, reverse);
  appendHexToString<64>(s, second, reverse);
	s += " ";
  appendHexToString<64>(s, third, reverse);
  appendHexToString<64>(s, fourth, reverse);

  return s;
}

inline std::string
m128toBin(const __m128i x, bool reverse = false, bool show16 = false)
{
  const std::uint64_t first = reverse ? _mm_extract_epi64(x, 0) : _mm_extract_epi64(x, 1);
  const std::uint64_t second = reverse ? _mm_extract_epi64(x, 1) : _mm_extract_epi64(x, 0);

  std::string s;
  s += toBinaryString<64>(first);
  s += toBinaryString<64>(second);

  if (show16) {
    s += " 0x";
    appendHexToString<64>(s, first, reverse);
    appendHexToString<64>(s, second, reverse);
  }

  return s;
}
