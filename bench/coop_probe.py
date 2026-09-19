"""Does a cooperative launch with a hand-rolled grid barrier capture into a
CUDA graph? Our whole decode step is one graph, so if it does not, a
barrier-based fused kernel cannot be used no matter how fast it is.
"""

SOURCE = r"""
// A grid-wide barrier. Safe only under a cooperative launch, which guarantees
// every block is resident; a spin on an unscheduled block would deadlock.
__device__ void grid_barrier(unsigned int* counter, unsigned int* generation,
                             unsigned int blocks)
{
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int gen = atomicAdd(generation, 0u);
        __threadfence();
        if (atomicAdd(counter, 1u) == blocks - 1u) {
            atomicExch(counter, 0u);
            __threadfence();
            atomicAdd(generation, 1u);
        } else {
            while (atomicAdd(generation, 0u) == gen) { __threadfence(); }
        }
    }
    __syncthreads();
}

extern "C" __global__ void two_phase(float* scratch, unsigned int* counter,
                                     unsigned int* generation, int n,
                                     unsigned int blocks)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < n; i += stride) scratch[i] = (float)i;
    grid_barrier(counter, generation, blocks);
    // Phase two reads what every other block wrote.
    for (int i = tid; i < n; i += stride) scratch[i] = scratch[(i + 1) % n] + 1.0f;
}
"""
