#pragma once

#include <immintrin.h>

namespace rimd {

// This borrows the naming convention from arms Neon.

/**
 * Class representing a 128i vector
 */
class Int8x16
{
private:
  __m128i vec;

public:
  // NOTE(rHermes): Decide if default should be undefined or zero.

  Vec16x8i() : vec{ _mm_undefined_si128() } {}
  Vec16x8i(__m128i v) : vev(v) {}

  [[nodiscard]] static Vec16x8i Zero() { return { _mm_setzero_si128() }; }

};

/**
 * A 128bit vector seen as 16x8bit unsigned
 */
class Vec16u
{};

} // namespace rimd