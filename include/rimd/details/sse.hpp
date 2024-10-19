#pragma once

#include "misc.hpp"

#include <concepts>
#include <cstdint>

#include <immintrin.h>

namespace rimd {

// This borrows the naming convention from arms Neon.

template<typename T>
concept IsIntegerSSE = requires(T t) {
  // TODO(rHermes): Make it so that these are required to return the correct types.
  { T::Zero() } -> std::same_as<T>;
  { t.data() } -> std::same_as<__m128i>;

  requires std::convertible_to<T, __m128i>;
};

namespace details {
// put some mixins here, zero and data and so on?
}

/**
 * Class representing a vector of 16 x unsigned 8 bit characters.
 */
class Uint8x16
{
private:
  __m128i m_data{ _mm_undefined_si128() };

public:
  // TODO(rHermes): Decide if default should be undefined or zero.
  // TODO(

  FORCE_INLINE Uint8x16() = default;
  FORCE_INLINE Uint8x16(__m128i v) : m_data(v) {}

  FORCE_INLINE Uint8x16(const std::uint8_t s15,
                        const std::uint8_t s14,
                        const std::uint8_t s13,
                        const std::uint8_t s12,
                        const std::uint8_t s11,
                        const std::uint8_t s10,
                        const std::uint8_t s9,
                        const std::uint8_t s8,
                        const std::uint8_t s7,
                        const std::uint8_t s6,
                        const std::uint8_t s5,
                        const std::uint8_t s4,
                        const std::uint8_t s3,
                        const std::uint8_t s2,
                        const std::uint8_t s1,
                        const std::uint8_t s0)
  {
    m_data = _mm_set_epi8(s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0);
  }
  FORCE_INLINE explicit Uint8x16(const std::uint8_t s0) { m_data = _mm_set1_epi8(s0); }

  FORCE_INLINE operator __m128i() const { return m_data; }

  template<IsIntegerSSE T>
  FORCE_INLINE explicit operator T() const
  {
    return T(m_data);
  };

  [[nodiscard]] FORCE_INLINE static Uint8x16 Zero() { return { _mm_setzero_si128() }; };
  [[nodiscard]] FORCE_INLINE static Uint8x16 LoadUnaligned(void const* ptr)
  {
    return _mm_loadu_si128(static_cast<const __m128i*>(ptr));
  }

  [[nodiscard]] FORCE_INLINE __m128i data() const { return m_data; }

  // Lane methods
  template<int Bytes>
  [[nodiscard]] FORCE_INLINE Uint8x16 shiftLanesRight() const
  {
    return _mm_bsrli_si128(m_data, Bytes);
  }

  template<int Bytes>
  [[nodiscard]] FORCE_INLINE Uint8x16 shiftLanesLeft() const
  {
    return _mm_bslli_si128(m_data, Bytes);
  }
};

/**
 * Class representing a vector of 16 x unsigned 8 bit characters.
 */
class Int8x16
{
private:
  __m128i m_data{ _mm_undefined_si128() };

public:
  // TODO(rHermes): Decide if default should be undefined or zero.
  // TODO(

  Int8x16() = default;
  Int8x16(__m128i v) : m_data(v) {}

  [[nodiscard]] __m128i data() const { return m_data; }
  operator __m128i() const { return m_data; }

  [[nodiscard]] static Int8x16 Zero() { return { _mm_setzero_si128() }; }
};

static_assert(IsIntegerSSE<Uint8x16>);
static_assert(IsIntegerSSE<Int8x16>);
} // namespace rimd