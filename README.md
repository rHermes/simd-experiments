# simd-experiments

This is a simple repository that I use to learn SIMD. As of now, there are
mostly projects dealing with solving various leetcode tasks with SIMD.

This is written to be sorta cross platform, but my primary development device
is Linux.

# Projects

## `leetcode-921`

This is a solution to [Leetcode 921](https://leetcode.com/problems/minimum-add-to-make-parentheses-valid/).
The task description is:

### Description

A parentheses string is valid if and only if:
- It is the empty string,
- It can be written as `AB` (`A` concatenated with `B`), where `A` and `B` are valid strings, or
- It can be written as `(A)`, where `A` is a valid string.

You are given a parentheses string `s`. In one move, you can insert a parenthesis at any position of the string.

For example, if `s = "()))"`, you can insert an opening parenthesis to be "(()))" or a closing parenthesis to be "())))".

Return the minimum number of moves required to make s valid.

### Solution notes

I've implemented this using SSE4.2 and AVX2. I learned a lot about doing prefix sums in SIMD registers and more.
I'll continue to hammer away at this one, to try to optimize it more, but as of now, the AVX2 version is roughly
26x times faster than the scalar version.


## `leetcode-2696`

This is a partial solution to [Leetcode 2696](https://leetcode.com/problems/minimum-string-length-after-removing-substrings).
The task description is:

### Description

You are given a string `s` consisting only of **uppercase** English letters.

You can apply some operations to this string where, in one operation, you can remove any occurrence of one of the substrings `"AB"` or `"CD"` from `s`.

Return *the __minimum__ possible length of the resulting string that you can obtain.*

Note that the string concatenates after removing the substring and could produce new `"AB"` or `"CD"` substrings.


### Takeaways

I didn't implement the full solution here, but I did end up emulating a sort of "compress" function
using just `SSE4.2`. This would have been trivial with `AVX512`, but this is not supported on most
CPUs these days.

I am very proud of the final version and it gave me tons of new topics to research.


## `leetcode-2938`

Speedup:
    - SSE4: 20x
    - AVX2: 30x

This is a partial solution to [Leetcode 2938](https://leetcode.com/problems/separate-black-and-white-balls/).
The task description is:

### Description

There are `n` balls on a table, each ball has a color black or white.

You are given a **0-indexed** binary string `s` of length `n`, where `1` and `0` represent black and white balls, respectively.

In each step, you can choose two adjacent balls and swap them.

_Return the **minimum** number of steps to group all the black balls to the right and all the white balls to the left._

### Solution notes

I ended up very pleased with this one, it took some real thinking to be able to figure this one out.

The trick is to always consider the current chunk in relation to some anchor I, which is one before the current
chunk starts. When you write `I = oldPos + lag`, we can express `oldPos` as `I - lag`, which turns out to cancel
all references to `I` in the final expression. This means that the only state we need to keep is `answer` and `lag`.

There is a bit more to it, but I it's rather elegant. Especially the fact that I chose to have `I` as one before, so
that we can identify 0 from 1.

For AVX2, I couldn't just use 1-32, because the 8 bit psa on the upper lane could overflow. So instead I did 1-16 on
the upper lane also. and then added `16*numberOfZeros` to the sum after. This is fast because a multiplying by 16 is the
same as left shifting 4 times.

One of the slower parts here is that we need to calculate a triangle number for the number of zeros. I would ideally prefer
to just use a lookup table for this and shuffle, but the problem is that 0 is also an option, so I would need to do some
sort of XOR and it just felt bad.

This was really nice :)

### Takeaways

# Sources and resources

- https://uica.uops.info/
- https://godbolt.org
- https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
