#pragma once

#include <concepts>

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

  Uint8x16() = default;
  Uint8x16(__m128i v) : m_data(v) {}

  [[nodiscard]] __m128i data() const { return m_data; }
  operator __m128i() const { return m_data; }

  [[nodiscard]] static Uint8x16 Zero() { return { _mm_setzero_si128() }; }
};

static_assert(IsIntegerSSE<Uint8x16>);
  /**
   * A 128bit vector seen as 16x8bit unsigned
   */
  class Vec16u
{};

} // namespace rimd