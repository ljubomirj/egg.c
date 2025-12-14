#!/bin/bash

# Test script for optimized GPU implementation

echo "=== Testing EGGROLL Optimized GPU Implementation ==="
echo

# Set environment variable to use optimized Metal
export EGG_USE_OPTIMIZED_METAL=1

# Build and test the optimized version
echo "Building optimized GPU implementation..."
make build/egg-gpu-macos-metal-optimized.debug

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo
    echo "Running optimized GPU training for 100 steps..."
    echo "Expected: Should be much faster than the original Metal implementation"
    echo

    # Run with a small test
    time ./build/egg-gpu-macos-metal-optimized.debug 2>&1 | head -20

    echo
    echo "=== Performance Comparison ==="
    echo
    echo "To compare with original implementations:"
    echo "1. Original CPU: time ./build/egg-cpu-macos-arm64.debug"
    echo "2. Original Metal: time ./build/egg-gpu-macos-metal.debug"
    echo "3. Optimized Metal: time ./build/egg-gpu-macos-metal-optimized.debug (with EGG_USE_OPTIMIZED_METAL=1)"
    echo
else
    echo "Build failed!"
    exit 1
fi
