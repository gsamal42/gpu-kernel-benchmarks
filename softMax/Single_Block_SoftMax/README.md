# SoftMax CUDA Kernel

## Overview

Mathematically:

```math
SoftMax(x_i) =
\frac{e^{x_i}}
{\sum_j e^{x_j}}
```

---

# Motivation

The purpose of this implementation is to understand:

* CUDA thread indexing
* Shared memory
* Thread synchronization
* Atomic operations
* GPU reduction patterns
* Kernel optimization techniques

This implementation intentionally starts with a simple and easy-to-understand version before introducing optimizations.

# Execution Flow

```text
Input Vector
      ↓
Compute exp(x)
      ↓
Atomic accumulation
      ↓
Synchronization
      ↓
Normalize outputs
```

---

# Why Shared Memory?

SoftMax requires a common denominator:

```math
\sum_i e^{x_i}
```

Shared memory provides a fast block-local storage that is visible to all threads inside a block.

```text
Thread0
Thread1
Thread2
    ↓
Shared totalSum
```

---

# Why Atomic Add?

Multiple threads update `totalSum` simultaneously.

Without:

```cpp
atomicAdd(&totalSum, expV);
```

race conditions can occur.

---

# Current Limitations

## 1. Single Block Implementation

This implementation assumes:

```cpp
gridDim.x == 1
```

because shared memory is local to a block.

---

## 2. Maximum Threads Per Block

Most NVIDIA GPUs support:

```text
1024 threads/block
```

Therefore:

```cpp
softmax_cuda<<<1,1024>>>();
```

is the maximum direct launch configuration.

---

## 3. Numerical Instability

Current implementation computes:

```cpp
expf(x)
```

which may overflow:

```text
exp(1000) → INF
```

Production implementations instead compute:

```cpp
expf(x - max(x))
```

---

## 4. Atomic Contention

All threads update:

```cpp
totalSum
```

which becomes a bottleneck as thread count increases.

Version 2 improves numerical stability by subtracting the maximum input value before exponentiation.

The kernel computes:

SoftMax(x) =
exp(x - max(x))
--------------------
Σ exp(x - max(x))

This avoids exponential overflow and prevents NaN generation for large inputs.

The implementation still relies on atomics and is limited to single-block execution, but serves as a correctness-oriented intermediate step before introducing parallel reductions.