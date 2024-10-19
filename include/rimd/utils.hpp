#pragma once
#include <algorithm>
#include <array>
#include <bitset>
#include <charconv>
#include <concepts>
#include <cstdint>
#include <functional>
#include <string>

#include <immintrin.h>
#include <utility>

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

namespace rimd {

template<std::integral T>
auto
extract_128(const __m128i xs)
{
  std::array<T, 128 / sizeof(T)> out;
  _mm_storeu_si128(reinterpret_cast<__m128i*>(out.data()), xs);
  return out;
}

FORCE_INLINE __m256i
flipLanes(const __m256i xs)
{
  return _mm256_permute2x128_si256(xs, xs, 0x01);
}

FORCE_INLINE __m128i
extractUpperLane(const __m256i xs)
{
  return _mm256_castsi256_si128(flipLanes(xs));
}

FORCE_INLINE __m128i
extractLowerLane(const __m256i xs)
{
  return _mm256_castsi256_si128(xs);
}

FORCE_INLINE __m128i
flipBits(const __m128i xs)
{
  return _mm_xor_si128(xs, _mm_set1_epi32(-1));
}

FORCE_INLINE __m256i
flipBits(const __m256i xs)
{
  return _mm256_xor_si256(xs, _mm256_set1_epi32(-1));
}

/// Switch the sign of the elements
template<std::size_t Bits>
FORCE_INLINE __m128i
negate(__m128i xs)
{
  if constexpr (Bits == 8) {
    return _mm_sub_epi8(_mm_setzero_si128(), xs);
  } else if constexpr (Bits == 16) {
    return _mm_sub_epi16(_mm_setzero_si128(), xs);
  } else if constexpr (Bits == 32) {
    return _mm_sub_epi32(_mm_setzero_si128(), xs);
  } else if constexpr (Bits == 64) {
    return _mm_sub_epi64(_mm_setzero_si128(), xs);
  } else {
    static_assert(Bits == 8, "Bits must be 8, 16, 32 or 64");
  }
}

/// Switch the sign of the elements
template<std::size_t Bits>
FORCE_INLINE __m256i
negate(__m256i xs)
{
  if constexpr (Bits == 8) {
    return _mm256_sub_epi8(_mm256_setzero_si256(), xs);
  } else if constexpr (Bits == 16) {
    return _mm256_sub_epi16(_mm256_setzero_si256(), xs);
  } else if constexpr (Bits == 32) {
    return _mm256_sub_epi32(_mm256_setzero_si256(), xs);
  } else if constexpr (Bits == 64) {
    return _mm256_sub_epi64(_mm256_setzero_si256(), xs);
  } else {
    static_assert(Bits == 8, "Bits must be 8, 16, 32 or 64");
  }
}

template<bool Right, int Times, typename T>
FORCE_INLINE T
shiftLanes(T xs)
{
  if constexpr (std::same_as<T, __m128i>) {
    if constexpr (Right) {
      return _mm_bsrli_si128(xs, Times);
    } else {
      return _mm_bslli_si128(xs, Times);
    }
  } else if constexpr (std::same_as<T, __m256i>) {
    if constexpr (Right) {
      return _mm256_bsrli_epi128(xs, Times);
    } else {
      return _mm256_bslli_epi128(xs, Times);
    }
  } else {
    static_assert(false, "Not a valid size type");
    std::unreachable();
  }
}

namespace details {
template<std::size_t Bits,
         typename T,
         auto& F1,
         auto& F2,
         auto& F3,
         auto& F4,
         auto& F5,
         auto& F6,
         auto& F7,
         auto& F8,
         typename... Args>
FORCE_INLINE auto
functionSelector(Args... args)
{
  if constexpr (std::same_as<T, __m128i>) {
    if constexpr (Bits == 8) {
      return F1(args...);
    } else if constexpr (Bits == 16) {
      return F2(args...);
    } else if constexpr (Bits == 32) {
      return F3(args...);
    } else if constexpr (Bits == 64) {
      return F4(args...);
    } else {
      static_assert(Bits == 8, "Bits must be 8, 16, 32 or 64");
      std::unreachable();
    }
  } else if constexpr (std::same_as<T, __m256i>) {
    if constexpr (Bits == 8) {
      return F5(args...);
    } else if constexpr (Bits == 16) {
      return F6(args...);
    } else if constexpr (Bits == 32) {
      return F7(args...);
    } else if constexpr (Bits == 64) {
      return F8(args...);
    } else {
      static_assert(Bits == 8, "Bits must be 8, 16, 32 or 64");
      std::unreachable();
    }
  } else {
    static_assert(false, "type not supported");
    std::unreachable();
  }
}

template<std::size_t Bits, typename T>
FORCE_INLINE T
addElements(T a, T b)
{
  return functionSelector<Bits,
                          T,
                          _mm_add_epi8,
                          _mm_add_epi16,
                          _mm_add_epi32,
                          _mm_add_epi64,
                          _mm256_add_epi8,
                          _mm256_add_epi16,
                          _mm256_add_epi32,
                          _mm256_add_epi64>(a, b);
}

template<std::size_t Bits, typename T>
FORCE_INLINE T
signedMinElements(T a, T b)
{
  return functionSelector<Bits,
                          T,
                          _mm_min_epi8,
                          _mm_min_epi16,
                          _mm_min_epi32,
                          _mm_min_epi64,
                          _mm256_min_epi8,
                          _mm256_min_epi16,
                          _mm256_min_epi32,
                          _mm256_min_epi64>(a, b);
}

template<std::size_t Bits, typename T>
FORCE_INLINE T
unsignedMinElements(T a, T b)
{
  return functionSelector<Bits,
                          T,
                          _mm_min_epu8,
                          _mm_min_epu16,
                          _mm_min_epu32,
                          _mm_min_epu64,
                          _mm256_min_epu8,
                          _mm256_min_epu16,
                          _mm256_min_epu32,
                          _mm256_min_epu64>(a, b);
}

/*

Eisie suggested this one. She also pointed out that we should do this via a LIFT macro for the functions, so that its
more of a sure thing that they get inlined. But I might be able to use force_inline?

The LIFT macro would be used to turn a function pointer into a lambda


template<std::size_t Bits, typename ParallelAdd, typename ShiftByBytes>
FORCE_INLINE __m256i
general_calcRunningSum(__m256i xs, ParallelAdd add, ShiftByBytes shift)
{
  if constexpr (Bits == 8) {
    xs = add(xs, shift(xs, 1));
    xs = add(xs, shift(xs, 2));
    xs = add(xs, shift(xs, 4));
    xs = add(xs, shift(xs, 8));
    return xs;
  } else if constexpr (Bits == 16) {
    xs = add(xs, shift(xs, 2));
    xs = add(xs, shift(xs, 4));
    xs = add(xs, shift(xs, 8));
    return xs;
  } else if constexpr (Bits == 32) {
    xs = add(xs, shift(xs, 4));
    xs = add(xs, shift(xs, 8));
    return xs;
  } else if constexpr (Bits == 64) {
    xs = add(xs, shift(xs, 8));
    return xs;
  } else {
    static_assert(Bits == 8, "Bits must be 8, 16, 32 or 64");
  }
}
*/

template<std::size_t Bits, bool Reverse, typename T, typename F>
FORCE_INLINE T
genericRunningPsa(F&& op, T xs)
{
  if constexpr (Bits == 8) {
    xs = op(xs, shiftLanes<Reverse, 1, T>(xs));
    xs = op(xs, shiftLanes<Reverse, 2, T>(xs));
    xs = op(xs, shiftLanes<Reverse, 4, T>(xs));
    xs = op(xs, shiftLanes<Reverse, 8, T>(xs));
    return xs;
  } else if constexpr (Bits == 16) {
    xs = op(xs, shiftLanes<Reverse, 2, T>(xs));
    xs = op(xs, shiftLanes<Reverse, 4, T>(xs));
    xs = op(xs, shiftLanes<Reverse, 8, T>(xs));
    return xs;
  } else if constexpr (Bits == 32) {
    xs = op(xs, shiftLanes<Reverse, 4, T>(xs));
    xs = op(xs, shiftLanes<Reverse, 8, T>(xs));
    return xs;
  } else if constexpr (Bits == 64) {
    xs = op(xs, shiftLanes<Reverse, 8, T>(xs));
    return xs;
  } else {
    static_assert(Bits == 8, "Bits must be 8, 16, 32 or 64");
    std::unreachable();
  }
}
} // namespace details

template<std::size_t Bits, bool Reverse = false, typename T>
FORCE_INLINE T
calcRunningSum(T xs)
{
  return details::genericRunningPsa<Bits, Reverse>(details::addElements<Bits, T>, xs);
}

template<std::size_t Bits, bool Reverse = false, typename T>
FORCE_INLINE T
calcRunningMinSigned(T xs)
{
  return details::genericRunningPsa<Bits, Reverse>(details::signedMinElements<Bits, T>, xs);
}

template<std::size_t Bits, bool Reverse = false, typename T>
FORCE_INLINE T
calcRunningMinUnsigned(T xs)
{
  return details::genericRunningPsa<Bits, Reverse>(details::unsignedMinElements<Bits, T>, xs);
}

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

} // namespace simd
