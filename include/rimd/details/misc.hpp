#pragma once

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

#define LIFT(f)                                                                                                        \
  [&] (auto&&... args)                                          \
    noexcept(noexcept(f(std::forward<decltype(args)>(args...))) \
    -> decltype(f(std::forward<decltype(args)>(args...)))       \
  {                                                             \
    return f(std::forward<decltype(args)>(args...);             \
  }

