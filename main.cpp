#include <immintrin.h>   // For SSE intrinsics
#include <chrono>
#include <iostream>
#include <cmath>

void scalarLoop(int n, float* array) {
    for (int i = 0; i < n; ++i) {
        array[i] += 1.0f;
    }
}

void simdLoopSSE(int n, float* array) {
    int i = 0;
    // Process 16 floats at a time (4 SSE operations)
    for (; i <= n - 16; i += 16) {
        // Load and process first 4 floats
        __m128 va1 = _mm_loadu_ps(&array[i]);
        // Load and process next 4 floats
        __m128 va2 = _mm_loadu_ps(&array[i + 4]);
        // Load and process next 4 floats
        __m128 va3 = _mm_loadu_ps(&array[i + 8]);
        // Load and process last 4 floats
        __m128 va4 = _mm_loadu_ps(&array[i + 12]);
        
        __m128 vone = _mm_set1_ps(1.0f);
        
        va1 = _mm_add_ps(va1, vone);
        va2 = _mm_add_ps(va2, vone);
        va3 = _mm_add_ps(va3, vone);
        va4 = _mm_add_ps(va4, vone);
        
        _mm_storeu_ps(&array[i], va1);
        _mm_storeu_ps(&array[i + 4], va2);
        _mm_storeu_ps(&array[i + 8], va3);
        _mm_storeu_ps(&array[i + 12], va4);
    }
    // Handle remaining elements
    for (; i < n; ++i) {
        array[i] += 1.0f;
    }
}

bool isClose(float a, float b, float relTol = 1e-5f, float absTol = 1e-8f) {
    return std::fabs(a - b) <= std::max(relTol * std::max(std::fabs(a), std::fabs(b)), absTol);
}

bool verifyResults(const float *array, int n) {
    for(int i = 0; i < n; ++i) {
        float expectedValue = static_cast<float>(i + 1);
        if(!isClose(array[i], expectedValue)){
            std::cout << "Verification failed at index: " << i 
                     << "\nExpected value: " << expectedValue
                     << "\nActual value: " << array[i] 
                     << "\nDifference: " << std::fabs(array[i] - expectedValue) << std::endl;
            return false;
        }
    }
    return true; 
}

int main() {
    const int n = 1000 * 1000 * 100;
    // Align memory to 16-byte boundary for better performance
    float* array = static_cast<float*>(_mm_malloc(n * sizeof(float), 16));
    
    // Initialize with some values
    for (int i = 0; i < n; ++i) {
        array[i] = static_cast<float>(i);
    }
    
    // Scalar loop benchmark and execution
    auto startScalar = std::chrono::high_resolution_clock::now();
    scalarLoop(n, array);
    auto endScalar = std::chrono::high_resolution_clock::now();
    auto scalarTime = std::chrono::duration_cast<std::chrono::microseconds>(endScalar - startScalar).count();
    std::cout << "Scalar loop time: " << scalarTime << " microseconds" << std::endl;
    
    // Verify scalar results
    bool scalarResult = verifyResults(array, n);
    
    // Reinitialize array for SIMD
    for (int i = 0; i < n; ++i) {
        array[i] = static_cast<float>(i);
    }
    
    // SIMD loop benchmark and execution
    auto startSSE = std::chrono::high_resolution_clock::now();
    simdLoopSSE(n, array);
    auto endSSE = std::chrono::high_resolution_clock::now();
    auto simdTime = std::chrono::duration_cast<std::chrono::microseconds>(endSSE - startSSE).count();
    std::cout << "SIMD loop time: " << simdTime << " microseconds" << std::endl;
    
    // Calculate and display speedup
    float speedup = static_cast<float>(scalarTime) / simdTime;
    std::cout << "Speedup: " << speedup << "x" << std::endl;
    
    // Verify SIMD results
    bool simdResult = verifyResults(array, n);
    
    if (scalarResult && simdResult) {
        std::cout << "Both loops produced correct results." << std::endl;
    } else {
        std::cout << "Verification failed!" << std::endl;
    }
    
    _mm_free(array);  // Free aligned memory
    return 0;
}