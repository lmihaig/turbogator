# TurboGator

## 1. Project Description

> Give the name of the project you are working on. State possible assumptions that you impose beyond the project description (e.g. input size divisible by 4, only consider the 2D version, etc.)

TurboGator

We consider the problem of optimizing the forward inference of the Geometric Algebra Transformer (GATr).

Input: a valid GATr input as specified by the reference implementation, consisting of a tensor of multivectors of shape (..., in_mv_channels, 16), where each 16-dimensional real vector represents one multivector in the projective geometric algebra G(3,0,1). Optional additional inputs are scalar channels of shape (..., in_s_channels), an attention mask of shape (..., num_items, num_items), and a join reference multivector of shape (..., 16) or an equivalent predefined reference choice.

Output: the corresponding forward-pass output of the model, numerically consistent with the reference implementation up to standard floating-point roundoff.
Assumptions: inference-only setting, CPU execution, and possible implementation-dependent constraints on admissible tensor sizes.

## 2. Existing Implementation

> If you use existing code as starting point: What code exists and in what language? We strongly prefer that you implement at least the performance-critical parts from scratch.

Code already exists in the official GitHub repository accompanying the paper. The baseline implementation is written in Python/PyTorch and serves as the starting point for our optimized version.

## 3. C-Straightforward

> Create a straightforward implementation in C (you can use C++ for the infrastructure if needed). Make simplifying assumptions (e.g., suitable divisibility of input size) if needed.

## 4. Validation

> Create a testing infrastructure so you can easily test your code. This involves selecting a suitable set of input sizes (and inputs). Note that this part will not be considered for the project grade.

## 5. Timing

> Create a timing infrastructure to get runtimes for a suitable set of input sizes (and inputs). Note that this infrastructure will not be considered for the project grade. Push sizes to the limit (i.e., until the execution takes minutes or even hours). Make sure to find and use good compiler flags.

## 6. Cost analysis

> Cost analysis: Select a cost measure (usually flop count but may include div, sin, ...; for integer computations analogous) and determine the cost. Briefly say how you did it (e.g., counted in code, had to instrument parts, ...).

## 7. Performance plot

> Create a performance plot. Put it as plot0 (or plot0a, plot0b, ...) into svn in the same directory as this document. As you perform optimizations later, include one line for each relevant optimization.

## 8. Benchmark alternatives

> Find (e.g., on the web) suitable (same algorithm, ideally claims to be fast) benchmark code and include it in your performance plot

## 9. Bottlenecks

> Identify performance bottlenecks in your code by profiling. This will likely involve a simple instrumenting of your code in possible addition to using profiling tools, which are likely not fine grain enough (we have some info on profiling tools on the course website). Focus on bottlenecks when optimizing. The bottlenecks may change during optimization. Understand the cost and performance of the bottlenecks to set expectations right.

## 10. Std. C Optimizations

> Optimize your code using techniques from class. Since teams have four people, divide into subteams. These could focus on optimizing with and without vectorization, different parts of the algorithm, different variants of the algorithms, or any other division of work that makes sense. Also a possible choice: one student focusses on a detailed analysis of your code variants with performance counters to help guiding the optimization. First optimize without SIMD operations:
>
> - For performance critical parts use a suitable code style as learned in class.
> - Use a suitable data representation so you get locality in the access. Spatial: ideally access data in sequence. For data that is accessed only once: do you have at least spatial locality? Temporal: inspect repeated sweeps through large amounts of data, refactor the algorithm to work on smaller amounts of data.
> - Remove unnecessary computations (maybe something can be precomputed and reused).
> - Remove sin/cos etc. computations if they are a bottleneck. Possibilities include using symmetries, precomputations, recurrences, precomputation plus interpolation.
> - Perform basic block optimization: unrolling + scalar replacement, ILP, register locality, etc. Don't call unrolling by itself a performance optimization.
> - Possibly create a simple autotuning infrastructure if there are choices.
> - Make sure to report the best non-vectorized code you obtain, compiled with flags disabling vectorization

## 11. Vectorize AVX

> Optimize using AVX or AVX2 or intrinsics and compare against the best compiler-vectorized scalar code from before. If you have access to AVX-512, also use that. Or feel free to use ARM Neon intrinsics on recent Apple computers:
>
> - Pay attention to alignment.
> - Keep shuffles to a minimum. Inspect available shuffle instructions and choose the most efficient ones.
> - Always be aware of latency and throughput of the intrinsics used.
> - Inspect whether some tricks from class can be used.
> - When done, the subteams could combine the previous scalar optimization with vectorization.
> - Combine SIMD optimizations with non-SIMD optimizations.

## 12. Plot every improvement

> For every significantly successful set of optimizations add a new performance line to the overall performance plot, thus creating line1, line2, etc. Add plots to the svn directory.

## 13. Additional analysis

> Try to explain the runtime and performance behaviour and argue why you could not do better. Try to explain also failed optimization attempts. Generally, any form of analysis and reasoning adds to the value of the project

## 14 Roofline

> Optionally try a roofline analysis to understand the effect of your optimizations. But if you do, explain how you did it, i.e, how you performed the necessary measurements and the cache asumptions. As explained in class: Do not use Intel Advisor's CARM model as it is different from the one in class. You can use the newer MLR model in Intel Advisor but use properly as mentioned in the roofline lecture.
