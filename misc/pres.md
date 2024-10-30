# SIMD

## Why watch this talk?

- IN my examples we achieve an average of a 30x speedup over the scalar code
  - This is including autovectorization by the compilers, on the scalar code.
- SIMD is still nieche and finding a good holoistic intro to it is hard.
  - I haven't found one yet.
- I have made some pretty rad visualizations, which I might consider some of the best on the internet as of now
- I have a cool voice ;)
- It's my last day, and I want to knowledge dump :p


## What is SIMD?

- Single Instruction, Multiple Data
- A way to increase throughput, by processing more elements at once.
- Somewhere between parallel programming and GPU
- You are probabily all using it through either libraries or auto vectorization

## Compiler tech

- Modern compilers can autovectorize known loops.
  - This has limits
  - Usually more than enough and better than what a human would write for the equivalent loops
  - Fails to kick in for more complex examples.
- Modern compilers also don't view intrinsics as assembly instructions
  - SIMD operations have IR representations
  - Compiler will manipulate these just like normal operations
  - Might reorder or change instructions as it wants
  - Might even switch SIMD use totally!
    - From SSE to AVX2 or from AVX2 to AVX512!
  - This makes SIMD so much easier than it used to be.


## What we are talking about

- Specifically x86 stuff, no ARM
  - SSE
  - AVX2
- ARM also has, called NEO
  - More methodical
  - Limited to 128 bits

## It's SIMD all the way

- On modern machines, there is no non SIMD version of SIMD operators.
- A scalar `add` and a SIMD `add` uses the exact same hardware on x64
- (Show picture of zen3 ALUS)
- This also means that you need to support the bitwidth of SIMD on all data paths
    - SSE: 128 bits
    - AVX2: 256 bits
    - AVX512: 512 bits
- This is also the reason why Intel axed AVX512
    - Takes up die space and electricity (which means heat)
    - Intel was never able to figure this out and so decided 256 bit was a good tradeoff between the two
- This also has effects on the implementation of AVX, as we will see later
    - They didn't efficient 256 bit data lanes at the time, so they had to implement it using "double push 128 bit".
    - This manifests that in AVX we talk about "lanes", which are 128bits. Very few operations work across the two lanes in a 256 bit AVX variable.

## About x86 SIMD

- Messy
  - Some conventions, but lots of exceptions
  - Huge amount of very specific functionality. `_mm_hmin_epu16` as an example.
- SSE and AVX are ubiquitus today.
  - Show steam hardware survey
- AVX512 is "dead", for now, on almost all intel CPUs
  - Really sad as the instruction set brough a bunch of amazing operators, not only 512 bit width
  - Intel knows this and so is pushing for something called AVX10, which would just introduce the amazing operators, but not the 512 bit width
    - Not a thing yet, but might be in the future.
    - Also some debate as to if this is a good idea, or if intel should just get their act together and support AVX512

## Programming stuff

- Only two real data types
  - Integer Nbit
  - Float Nbit
- Logically all operators fall into
  - 128bit operations
  - 64 bit signed / unsigned
  - 32 bit signed / unsigned
  - 16 bit signed / unsigned
  - 8 bit signed / unsigned
  - Floating point 32bit or 64bit
    - Intel pushing for 16bit floats, (link proposal here)
- Typical workflow is that you:
  - Load data from memory into SIMD registers
  - Operate on that data with SIMD in a pipeline
  - Write the data from SIMD registers back into memory.
- I have no experience with floating point, so I won't be talking about it at all.
- Some tools that are beyond awesome:
  - LLVM-MCA
    - Part of the LLVM pipeline that assess the uops execution of varius machines
  - Godbolt
    - This goes without saying, but being able to test on different arches and diffent compilers is amazing
  - uCIA
    - I will show this in this talk, but it's a tool to get timings and timing charts!
  - Intel instruction manual
    - Without this, SIMD is useless for me. Uber good index over all instructions, cateogries and can be used offline
    - I'm surprised intel managed to make something so good on the web, given their usual website standards.
  - Stackoverflow
    - A lot of things are hard and you will need help from smarter people.

## The modern monster that is CPUs

- Modern CPUs are superscalar, meaning they execute multiple instructions at once, on a single core.
- Even scarier, they can also do this out of order
- x86 is really just an abstract spec at this point.
  - I think of it more like the JVM bytecode spec, than hardware opcodes like before
- All modern processors break assembly instructions into a stream of smaller operators called "uops"
- These are then executed.
- The fetching, decoding, issuing, executing and retiring of these uops are all done by different parts
- They are also super scalar, meaning that the decoding and issuing can often happen many many cycles before execution
- All of this is propriotary and vendors guard it.
  - Why? Both to keep competition away and to avoid customers relying too much on the individual processor quirks. 
  - This does not stop nerds. The LLVM, GCC and other people spend 100's of hours measuring different execution flows to reverse engineer how the scheduling algorithms work
  - This is then used in the compiler when compiling to maximize theoretical throughput.
  - I know a guy who has an even more precise measurements, but he won't release how he does it, because he is afraid it will get patched.
- We will look at this later.
- Some instructions are even implemented in "microcode", meaning they are not implemented in hardware
  - (FORSHADOWING) this will never ever ever hit us later in this talk.
- What does all this mean?
  - We can no longer rely on intuition to say what will be fast.
    - Amount of assmebly is no good indication
    - Instructions can be fast in theory, slow in practice
    - Some instructions have hidden dependencies.
  - We have to use both benchmarks and executor estimators, to try to understand what is fastests.
- Benchmarks are KEY
  - I will show you that somethings that might look like big improvements, turns out to be slower than "stupidier" approaches.
  - Varies from machine to machine, layout to layout.
  - If this stuff matters to you, you will need to activly make choices based on the hardware you are running.

## All

