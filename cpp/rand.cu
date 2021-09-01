//
// Created by auyar on 17.08.2021.
//
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <string>
#include <math.h>

#include <cuda.h>
#include <curand_kernel.h>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

__global__ void setup_kernel(curandState *state) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence number, no offset */
    curand_init(1234, id, 0, &state[id]);
}

__device__ bool exist(unsigned int* local_results, int size, unsigned int number) {
    for (int i=0; i < size; i++) {
        if (local_results[i] == number){
            return true;
        }
    }

    return false;
}

__global__ void generate_uniform_kernel(curandState *state,
                                        unsigned int *results,
                                        const unsigned int sample_size,
                                        const unsigned int SAMPLES_PER_THREAD,
                                        const unsigned int data_size,
                                        const unsigned int data_size_per_thread) {

    int id = threadIdx.x + blockIdx.x * blockDim.x;

    // this can be a thread that is started extra in the last block,
    //      so it may not generate any random numbers
    // or this can be the last thread to generated random numbers,
    //      it may generate random numbers less than SAMPLES_PER_THREAD in a smaller range
    unsigned int thread_sample_start = id * SAMPLES_PER_THREAD;
    unsigned int samples_this_thread = SAMPLES_PER_THREAD;
    unsigned int data_size_this_thread = data_size_per_thread;

    if (thread_sample_start >= sample_size) {
        // this is an extra thread started in the last block
        return;
    } else if(thread_sample_start >= sample_size - SAMPLES_PER_THREAD) {
        // this is the last thread to generate randoms
        samples_this_thread = sample_size - thread_sample_start;
        data_size_this_thread = data_size - data_size_per_thread * id;
    }

    /* Copy state to local memory for efficiency */
    curandState localState = state[id];

    unsigned int* local_results = new unsigned int[samples_this_thread];
    for (int i=0; i < samples_this_thread; i++) {
        unsigned int rand_num = (unsigned int) ceil(curand_uniform_double(&localState) * data_size_this_thread) - 1;
        // check whether this number already generated
        if (exist(local_results, i, rand_num)) {
            i--;
        } else {
            local_results[i] = rand_num;
        }
    }

    // transfer generated random numbers to the global array
    unsigned int data_start = id * data_size_per_thread;
    for (int i=0, j = thread_sample_start; i < samples_this_thread; i++, j++) {
        results[j] = local_results[i] + data_start;
    }

    /* Copy state back to global memory */
    state[id] = localState;
    delete [] local_results;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("You must specify the data_size and sampling ratio as parameters.\n");
        return 1;
    }
    std::string dataSizeStr = argv[1];
    std::string samplingRatioStr = argv[2];

    unsigned int dataSize = std::stoi(dataSizeStr);
    float samplingRatio = std::stof(samplingRatioStr);

    const unsigned int SAMPLES_PER_THREAD = 64;
    const unsigned int THREADS_PER_BLOCK = 256;

    unsigned int sampleSize = dataSize * samplingRatio;
    const unsigned int totalThreads = ceil((double)sampleSize / (double)SAMPLES_PER_THREAD);

    unsigned int blockCount = ceil((double)totalThreads / (double)THREADS_PER_BLOCK);
    unsigned int data_size_per_thread = (dataSize / sampleSize) * SAMPLES_PER_THREAD;

    printf("dataSize: %i\n", dataSize);
    printf("data_size_per_thread: %i\n", data_size_per_thread);
    printf("sampleSize: %i\n", sampleSize);
    printf("SAMPLES_PER_THREAD: %i\n", SAMPLES_PER_THREAD);
    printf("totalThreads: %i\n", totalThreads);
    printf("blockCount: %i\n", blockCount);

    cudaEvent_t start1, stop1, start2, stop2;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    unsigned int i;
    unsigned int total;
    curandState *devStates;
    unsigned int *devResults, *hostResults;
    int device;

    /* check for double precision support */
    CUDA_CALL(cudaGetDevice(&device));

    /* Allocate space for results on host */
    hostResults = (unsigned int *) calloc(sampleSize, sizeof(int));

    /* Allocate space for results on device */
    CUDA_CALL(cudaMalloc((void **)&devResults, sampleSize * sizeof(unsigned int)));

    /* Set results to 0 */
    CUDA_CALL(cudaMemset(devResults, 0, sampleSize * sizeof(unsigned int)));

    /* Allocate space for prng states on device */
    CUDA_CALL(cudaMalloc((void **)&devStates, blockCount * THREADS_PER_BLOCK * sizeof(curandState)));

    /* Setup prng states */
    cudaEventRecord(start1);
    setup_kernel<<<blockCount, THREADS_PER_BLOCK>>>(devStates);
    cudaEventRecord(stop1);

    /* Generate and use pseudo-random  */
//    generate_kernel<<<64, 64>>>(devPHILOXStates, samplesPerThread, devResults);
//    generate_kernel<<<64, 64>>>(devStates, samplesPerThread, devResults);
    cudaEventRecord(start2);
    generate_uniform_kernel<<<blockCount, THREADS_PER_BLOCK>>>(devStates,
                                                devResults,
                                                sampleSize,
                                                SAMPLES_PER_THREAD,
                                                dataSize,
                                                data_size_per_thread);
    cudaEventRecord(stop2);

    // wait kernels to finish
    // cudaDeviceSynchronize();

    /* Copy device memory to host */
    CUDA_CALL(cudaMemcpy(hostResults, devResults, sampleSize * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    /* Show result */
    total = 0;
    for(i = 0; i < sampleSize; i++) {
        total += hostResults[i];
    }

    std::vector<int> randoms;
    for(i = 0; i < sampleSize; i++) {
        randoms.push_back(hostResults[i]);
    }
    int equalCount = 0;
//    std::sort(randoms.begin(), randoms.end());
    for (int i=0; i<randoms.size() -1; i++) {
        if (randoms[i] == randoms[i+1]) {
//            printf("random numbers are equal at i: %i\n", i);
            equalCount++;
        }
    }

    int zeroCount = 0;
    for (int i=0; i<randoms.size() -1; i++) {
        if (randoms[i] == 0) {
            zeroCount++;
        }
    }

    for(i = 0; i < sampleSize && i < 300; i++) {
        printf("%i: %i\n", i, randoms[i]);
    }
    printf("\n");
//    for(i = randoms.size() - 50; i < randoms.size(); i++) {
//        printf("%i: %i\n", i, randoms[i]);
//    }
    float initDelay = 0;
    cudaEventElapsedTime(&initDelay, start1, stop1);
    float genDelay = 0;
    cudaEventElapsedTime(&genDelay, start2, stop2);

    printf("number of equal random numbers: %i\n", equalCount);
    printf("number of zero random numbers: %i\n", zeroCount);
    printf("curand init delay: %f\n", initDelay);
    printf("curand generate delay: %f\n", genDelay);

    free(hostResults);
    cudaFree(devStates);
    cudaFree(devResults);

    return 0;
}
