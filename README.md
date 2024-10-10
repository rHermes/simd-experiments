# simd-experiments

This is a simple repository that I use to learn SIMD. As of now, there are
mostly projects dealing with solving various leetcode tasks with SIMD.

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


# Sources and resources

- https://uica.uops.info/
- https://godbolt.org
- https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#
