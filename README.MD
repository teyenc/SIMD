# SIMD Examples

This repository briefly demonstrates the power of SIMD (Single Instruction Multiple Data) operations in C++. SIMD enables parallel processing by performing the same operation on multiple data points simultaneously.

## What is SIMD?

SIMD is a type of parallel processing that allows a single instruction to be performed on multiple data points simultaneously. Think of it like this:

- **Regular (Scalar) Processing**: 
```
1+1, then 2+1, then 3+1, then 4+1 = Four operations
```

- **SIMD Processing**: 
```
[1,2,3,4] + [1,1,1,1] = One operation
```
This is super cool to be honest. 

## Implementation

The C++ implementation shows:
- Processing 100 million floats
- Using SSE instructions to process 16 elements at once
- ~1.6x speedup over scalar operations
- Memory-aligned operations for optimal performance

Example output:
```bash
$ g++ -o main main.cpp
$ ./main
Scalar loop time: 103416 microseconds
SIMD loop time: 64303 microseconds
Speedup: 1.60826x
```

## Building and Running
```bash
g++ -o main main.cpp
./main
```

It's fast (and clever), and we can use it in Databases. 

