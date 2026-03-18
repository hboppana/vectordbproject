#pragma once
#include "simd_distance.h"
#include "vector.h"
#include <cmath>

// L2 distance dispatches to SIMD implementation in simd_distance.h
// which automatically selects AVX2 or scalar fallback based on compile flags