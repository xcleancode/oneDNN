# Proposal for Primitive Compilation in oneDNN

## 1. Introduction
Slow primitive compilation consistently causes customer issues in oneDNN. To
handle this, we introduced a [global primitive
cache](https://github.com/oneapi-src/oneDNN/tree/rfcs/rfcs/20200207-global-primitive-cache).
This solution is insufficient for the GPU software stack, especially for OpenCL
C implementations which require at least 300 ms to compile. This issue exists
for other kernel generation methods as well. For example, the GPU convolution
generator commonly requires a significant time of 30ms. For many workloads, this
time investment is massive compared to execution time. In addition, we have
encountered workloads which require different primitives based on input,
rendering the global primitive cache relatively ineffective. Finally, slow
primitive generation significantly hampers testing oneDNN functionality. It is
time consuming to test thousands of cases when 300ms is necessary to construct
the primitive. While we have worked around OpenCL C compilation using
[cl_cache](https://github.com/intel/compute-runtime/blob/master/opencl/doc/FAQ.md#feature-cl_cache),
it is completely useless for the first functional test run.

## 2. Methods to Resolve this issue

At its core, there are two possible methods:
1. Improve compilation time
2. Reduce the number of kernels compiled.

In regards to improving compilation time, this is infeasible for our OpenCL C
kernels. As such, one solution is to use a different method of kernel
generation. We have done this for convolution, matrix multiplication and reorder
primitives as the OpenCL C language was hindering optimizations. This resulted
in implementing a custom IR and optimization passes to meet our needs for these
problems. There have been discussions about move element-wise and binary
primitives into this infrastructure as most of the required functionality exists
already due to post-op support.

The process of implementing and maintaining kernels in custom generators is more
expensive than using industry standard languages like OpenCL C. Due to limited
expected performance benefit, we are unlikely to ever port all primitives to
this option. In addition, while we have improved kernel compilation time by an
order of magnitude when compared to OpenCL C, it is unclear if a reasonable
investment in optimization will provide sufficient performance to rely on
code generators alone.

Because of this, the calls to the OpenCL C compiler need to be reduced. To
enable this, we need to create reusable OpenCL kernels. In addition, we need to
enable users to reuse these kernels. The rest of this document will focus on how
this reuse can be enabled. The previous discussion suggests a few desired
requirements on the solution.

1. This is a longterm process as it requires modifying many implementations and,
   as such, needs to be exposed on an as implemented basis.
2. The general methodology applies to all kernel generators.
3. Performance improves for benchdnn testing.

## 3. Method of Kernel Distribution
To begin with, we need to discuss how to distribute these kernels, as it places
significant restrictions on implementations. At its core, there are two options:

### Distribution Option 1: Compile Kernels at Build Time
#### Pros:
Users never need to compile a GPU kernel. Solves first compilation issues.

#### Neutral:
OpenCL C Compiler version is fixed at build time. As such users cannot benefit
or see regressions from installing a more recent compiler.

#### Cons:
Increases library size

Increases compilation time

Unclear how many primitives for which this is feasible.

More challenging to implement GPU kernels. This requires all kernels have a
finite number of configurations with reasonable binary size. This is technically
impossible for most primitives due to our API (in particular due to post-ops and
data layouts), but we can implement this feature for a subset of kernels in
common use. Because of this, we still need to maintain runtime compilation.

#### Implementation:
Because this method increases library size, compilation time, and still requires
runtime compilation, this feature should be enabled via a compilation switch
`ONEDNN_COMPILE_KERNELS_FOR_ARCH=<gpu_arch list>`. The default value should be
`NONE` as most users will need at most 2 GPU architectures compiled, an
integrated GPU and a discrete GPU. As such, for most users a significant
fraction of the increased library size is not helpful.

### Distribution Option 2: Compile Kernels Just in Time
#### Pros:
Smaller Library size

Enables more opportunity for JIT compiling.

#### Cons:
Customer has to compile the OpenCL C kernel at least once

#### Implementation:
This is the current behavior, so no changes are required.

### Recommendation:
The recommendation is option 2, keep the current distribution behavior. To allow
maximum flexibility, we should modify implementations so they can be distributed
under option 1.

## 4. Proposed Kernel Reuse Methodology
There are effectively two methodologies to reuse kernels. First, we can
let users explicitly control kernel reuse, or second reuse can be implicitly
handled by the library, via some kernel registering mechanism.

### Reuse Option 1: Users Rely on cl_cache for OpenCL C Kernels
Allow users to control reusing OpenCL C kernels via the existing cl_cache. This
recommendation has already been made to many customers.

#### Pros:
Allows user control.

Only requires modifying OpenCL C kernels.

#### Cons:
Poor default behavior. The use of cl_cache requires intervention from our users
to setup and manage the generated cache.

No support for non-OpenCL C code generation.

#### Implementation:
None

### Reuse Option 2: Implement Runtime Parameters for primitives
Create primitives where inputs are specified as runtime parameters. Such
functionality already exists for GPU GEMM.

#### Pros:
Allows user control.

Method already exists within oneDNN so there is already precedent.

Feature has been requested by frameworks

Implementable for all kernel generation methods

#### Cons:
Makes API more complicated

Requires user source code changes to benefit

Increases testing load as we need to test combinations of runtime parameters.

Does not improve benchdnn testing performance

Unless care is taken, this will lead to more implementations, increasing
implementation and maintenance burden

Opaque user experience. The user needs to choose between using runtime
parameters to reuse a kernel but with the trade off that they may get worse
performance. Since performance is implementation dependent, users are in a bad
position to when they can make this choice with limited performance degradation.

More work as API support should be implemented for both CPU and GPU.

#### Implementation:
The implementation will be highly dependent on primitive implementations. There
are two main scenarios here. First scenario, we modify our implementations to
generate a finite number of kernels. We then use runtime parameters to
dynamically switch top the best implementation. In this case, since we are
reusing the same finite number of kernels, maintainability is not impacted. The
other scenario requires a separate implementation which increases testing
requirements and maintenance burden.

### Reuse Option 3: Fixed Size Runtime Function Generator Registries
This option creates a fixed size registry for storing reusable kernel
generators. These generators are expected to be performant. For OpenCL C
kernels, these are expected to just be program binaries.

#### Pros:
Requires no user intervention.

No API changes.

Relatively simple implementation

Requirements on Function Generators is similar to Distribution Option 1.

#### Cons:
Requires a finite number of generators, no guarantee generators align with
customer workloads.

Users do not control the registry size

Requires the generator to be GPU agnostic (i.e. should work for 2 or more GPUs
in a system if they have the same architecture) to limit the registry size.

Implementation needs to limit maximum memory usage to a reasonable level.

#### Implementation
This feature will be completely managed by oneDNN developers. As such, there are
no API changes. The only guarantee is that the registry is generated at runtime
and will not use much memory until users creates primitives. It is expected the
implementation will be a hashmap between a kernel configuration and a kernel
generator. Registries will have a compile time defined maximum size to avoid
excessive memory usage. As a registries size is dependent on implementation,
separate registries will exist per implementation. As the requirements on this
implementations are similar to Distribution Option 1,this can be used to enable
build time compilation.

### Reuse Option 4: Global Kernel Generator Cache
This options is a version of the global primitive cache, but for gpu kernel
generators. Kernel generators are expected to be performant. For OpenCL C
kernels, the generator is expected to be a program binary.

#### Pros:
Requires no user intervention.

Implementation for global primitive cache is similar.

Users can control memory usage

Generators can be specialized for each GPU in a system, potentially enabling
faster performance.

#### Cons:
Most complicated implementation (except potentially Option 2) among all the
options.

#### Implementation
The API change will be the same as in the [global primitive
cache](https://github.com/oneapi-src/oneDNN/tree/rfcs/rfcs/20200207-global-primitive-cache),
except the "primitive" name will be replaced with "kernel". Kernels which use
this cache will need to opt in to its uses. Kernels need to be designed for
reuse for the cache to be effective.

### Recommendation
Short Answer: Start with Reuse Option 3 with a transition to a combination of
Reuse Option 2 and Reuse Option 4 on an as need basis.

There are significant benefits to modifying our implementations to generate a
limited and finite number of kernel binaries. Among the benefits are:
* Enables Build Time Compilation
* Enables Runtime Parameters
* Reduces Testing Complexity

While it is impossible to achieve this in general, for sufficiently large
problems, there tend to only be a few important compiled configurations. Because
of these advantages, I recommend we start curating a set of kernels which
address most sufficiently large problems.

To reuse, I recommend starting with Reuse Option 3, the fixed size function
registry. The motivating reason is the kernels generated for Reuse Option 3 also
satisfy the requirements for Distribution Option 1, enabling us to address
performance issues with first compilation. In addition, this option requires no
API changes while allowing users benefit from our improvements.

In cases where Reuse Option 3 is insufficient, we will transition to a
combination of Reuse Options 2 and 4. After the initial work
is performed to (as much as possible) support Reuse Option 3, transitioning to
Reuse Option 4 will not be difficult. At an implementation level, a global
primitive cache is very similar to a function registry. In addition, a
curated set of kernels can be used to easily implement Reuse Option 2 in which
oneDNN users have expressed interest. Option 1 is not recommended as it
only supports OpenCL C kernels.

As a data point, a combination of Reuse Options 2 and 4 is similar to how the
GEMM implementation works in oneMKL today. There are a finite number of curated
GEMM implementations which support runtime parameters, but too many for a
function registry. These kernels are stored in a kernel cache to avoid
recompilation. The set of curated functions work support runtime dimensions, and
so a set of them can be used to support runtime dimensions.
