# Parallel pyFFTW guidelines

## Purpose

The purpose of this page is to provide development guidelines
for the usage of pyFFTW in a multithreaded environment,
i.e. calling pyFFTW functions from several threads
instead of just relying on the internal pyFFTW parallelization.

pyFFTW is a wrapper of FFTW and as such inherits many concepts from FFTW,
which must be well understood to use pyFFTW efficiently.
Therefore, FFTW concepts are presented before discussing pyFFTW usage.

## Main FFTW concepts

### Planning phase

The strength of FFTW is to find the fastest algorithm
for the specific use case of the user, on its specific hardware.
This is accomplished by running an algorithm benchmark before computing any actual DFT.
FFTW names this the Planning phase, result of which is the **Plan**.
Once instantiated, the Plan can be executed,
i.e. the selected algorithm is run (which should be extremely fast).

The set of algorithms is partitioned according to complexity.
User can act on the Planning complexity by selecting the algorithm categories to be benchmarked
using keyword parameters: ESTIMATE < MEASURE < PATIENT < EXHAUSTIVE.
In addition, an approximate Planning timeout can be provided.
The more DFTs you expect the Plan to execute, the more patient you can be at Planning time.

By the way, using one Plan per DFT computation dismisses the benefits of pyFFTW;
Better use simpler tools, like Numpy or Scipy.
This is why there is **no such thing** as:

```python
output = fftw.dft(input)
```

### Memory layout

Because efficiency is tightly coupled with memory layout,
the Planning requires **predefined addresses** for the input and output arrays,
which are unnamed in FFTW's documentation but should be seen as permanent buffers (see later why).
Let's name them... **Buffers**!

The Buffers are used during benchmarking to run the algorithms,
and therefore contain garbage just after the Planing phase.
They are also used at execution time, for intermediate computations,
such that the input Buffer contains garbage after execution.
Thus the name of Buffer ;)

Once the Plan is available, the most efficient way to use it (but not the only one)
is to keep actual data at Buffer addresses as much as possible,
i.e. instead of giving an input array to the DFT operation and getting a returned output array,
the input values should be written into the input Buffer directly,
and the output values should be read from the output Buffer.
This is not very common in Python, and probably not the way you would like it to work.
Hopefully, there is an alternative!

The other (a tad slower) option is to work with standalone arrays:
* Copy the input values in FFTW's input Buffer just before computing the DFT;
* Copied the output values from the output Buffer.

Here is a pseudo-code which illustrates the second option (actual pyFFTW names differ):

```python
# Create the Plan for given shape
plan = create_plan(input.shape)

# Feed the input Buffer from a standalone array
plan.input_buffer[:] = input

# Execute the Plan
plan.execute()

# Copy the output Buffer to a standalone array
output[:] = plan.output_buffer

# WARNING! plan.input_array is garbage now!
```

### Thread safety

Not everything is thread-safe in FFTW, but heavy computations are.
In particular, the Planning **is not** thread-safe:
it writes in global variables, which are later read at Plan execution time.
Plan execution **is** thread-safe, provided that each thread has a dedicated Plan
and therefore dedicated Buffers.

Therefore, the multithreading model is like:

```python
# Create one plan per shape and per thread
# in a single thread (writes global variables)
plans = [create_plan(shape) for t in range(threads)]

# Execute Plans in multiple threads
for t in range(threads)
    # Write in thread-specific Buffers
    plans[t].input_buffer[:] = input_t
    plans[t].execute()
    output_t[:] = plans[t].output_buffer
```

## pyFFTW's API

Several APIs are presented in pyFFTW's documentation,
yet only the so-called Core API is worth considering for heavy work.
It is based on a single multi-purpose class, `FFTW`, which represents a Plan.

### Plan creation

An `FFTW` object is initialized with Buffers,
which must have been allocated with a dedicated SIMD-aware function `empty_aligned(shape, dtype)`.
When transform and inverse transform of the same data must be performed, two Plans must be created.
In this case, it is optimal to use the same Buffers for both Plans.

To avoid polluting production code with low-level boilerplate,
and to ensure forward and backward Plans are coherent,
a light Plan wrapper and Plan makers can be implemented easily:

```python
class DFT:
    def __init__(self, transform, inverse):
        self.transform = transform
        self.inverse = inverse

def create_complex_plan(shape):
    i = pyfftw.empty_aligned(shape, dtype='complex128')
    o = pyfftw.empty_aligned(shape, dtype='complex128')
    return DFT(pyfftw.FFTW(i, o), pyfftw.FFTW(o, i))

def create_real_plan(shape):
    i = pyfftw.empty_aligned(shape, dtype='float64')
    o = pyfftw.empty_aligned(shape, dtype='complex128')
    return DFT(pyfftw.FFTW(i, o), pyfftw.FFTW(o, i)
```


### Plan execution

The Plan is simply executed with call operator: `()`.
Input and output Buffers are accessed as Plan variables
`FFTW.input_array` and `FFTW.output_array`, respectively.
In pyFFTW world, the first example writes:

```python
# Create the Plan for given shape
i = pyfftw.empty_aligned(shape, dtype='complex128')
o = pyfftw.empty_aligned(shape, dtype='complex128')
plan = pyfftw.FFTW(i, o)

# Feed the input Buffer from a standalone array
plan.input_array[:] = input

# Execute the Plan
plan()

# Copy the output Buffer to a standalone array
output[:] = plan.output_array
```

This construct is useful to work with in-place transforms (cumbersome, yet optimal option 1),
but not needed with the standalone array approach (more natural option 2).
The latter approach is common enough that pyFFTW provides a shortcut:
The call operator can take as argument an input array, which will be copied,
and returns the output Buffer, which can be copied:

```python
# Create the Plan for given shape
i = pyfftw.empty_aligned(shape, dtype='complex128')
o = pyfftw.empty_aligned(shape, dtype='complex128')
plan = pyfftw.FFTW(i, o)

# Execute the Plan
output = plan(input)
```

## Wrap-up

Using the proposed `DFT` wrapper, executing forward and backward Plans
inside multiple threads is quite straightforward:

```python
# Create one plan per shape and per thread
plans = [create_complex_plan(shape) for t in range(threads)]

# Execute Plans in multiple threads
for t in range(threads)
    # Write in thread-specific Buffers
    output_t = plans[t].transform(input_t)
    # ... Do something with output_t in frequency domain
    input_t = plans[i].inverse(output_t)
```
