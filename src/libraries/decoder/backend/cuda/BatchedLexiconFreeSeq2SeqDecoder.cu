/*	
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.	
 *	
 * Redistribution and use in source and binary forms, with or without	
 * modification, are permitted provided that the following conditions	
 * are met:	
 *  * Redistributions of source code must retain the above copyright	
 *    notice, this list of conditions and the following disclaimer.	
 *  * Redistributions in binary form must reproduce the above copyright	
 *    notice, this list of conditions and the following disclaimer in the	
 *    documentation and/or other materials provided with the distribution.	
 *  * Neither the name of NVIDIA CORPORATION nor the names of its	
 *    contributors may be used to endorse or promote products derived	
 *    from this software without specific prior written permission.	
 *	
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY	
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE	
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR	
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR	
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,	
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,	
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR	
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY	
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT	
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE	
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.	
 */	

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <limits>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <arrayfire.h>
#include <flashlight/common/cuda.h>

#include <Utils.h>
#include <common/Defines.h>

namespace {
#define CHK_CUDA(expression)                                                  \
  {                                                                           \
    cudaError_t status = (expression);                                        \
    if (status != cudaSuccess) {                                              \
      std::cerr << "Error in file: " << __FILE__ << ", on line: " << __LINE__ \
                << ": " << cudaGetErrorString(status) << std::endl;           \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  }

const float kMaxFloat = std::numeric_limits<float>::max();
const int kNumThreads = 1024;

__global__ void calculateScoresKernel(
    int* numHypos,
    int* __restrict numNonEosHypos,
    int* __restrict peakAttnPos,
    int* __restrict tokensDevPtr,
    int* __restrict topKParentIndices,
    int* __restrict topKTokens,
    float* __restrict topKScores,
    const int* const __restrict finalized,
    const int* const __restrict hyposCompactIndices,
    const int* const __restrict usrToLmIdxMap,
    const int* const __restrict lastTokenPositions,
    const int* const __restrict bestPathDev,
    const float* const __restrict amScores,
    const float* const __restrict batchedProb,
    const int timeStep,
    const int CC,
    const int TT,
    const int lmEosTokenIndex,
    const int vocabSize,
    const int eos,
    const int beamSize,
    const int numTokens,
    const int attentionThreshold,
    const float eosScore,
    const float lmWeight,
    const float beamThreshold) {
  extern __shared__ float sm[];

  const int& stagingAreaSize = blockDim.x;
  const int& heapSize = beamSize;
  float* blockWideScores = reinterpret_cast<float*>(&sm[0]);
  int* blockWideParentIndices = reinterpret_cast<int*>(&sm[stagingAreaSize]);
  int* blockWideTokens = reinterpret_cast<int*>(&sm[stagingAreaSize << 1]);

  int heapOffset = stagingAreaSize * 3;
  float* scoreHeap = reinterpret_cast<float*>(&sm[heapOffset]);
  int* parentIndicesHeap =
      reinterpret_cast<int*>(&sm[heapOffset + heapSize + 1]);
  int* tokensHeap =
      reinterpret_cast<int*>(&sm[heapOffset + ((heapSize + 1) << 1)]);

  int& numNewScores = parentIndicesHeap[0]; // First element is always empty
  float& bestScore = scoreHeap[0];
  int& numElemsInHeapSmem =
      *reinterpret_cast<int*>(&sm[(stagingAreaSize + heapSize + 1) * 3]);
  int numElemsInHeap = 0;

  int uc = blockIdx.x;
  if (finalized[uc]) {
    return;
  }

  if (threadIdx.x == 0) {
    numNewScores = 0;
    scoreHeap[1] = -kMaxFloat;
    bestScore = -kMaxFloat;
    if (timeStep == 0) {
      peakAttnPos[uc * beamSize] = -1;
    }
  }
  __syncthreads();

  int offset = 0;
  int numHyposLocal = timeStep == 0 ? 1 : numHypos[uc];

  if (timeStep == 0) {
    offset = uc;
  } else {
    for (int k = 0; k < uc; k++) {
      offset += numNonEosHypos[k];
    }
  }

  int numPossibleHypos = numHyposLocal * numTokens;
  int numItrs = ((numPossibleHypos + blockDim.x - 1) / blockDim.x) * blockDim.x;

  for (int counter = threadIdx.x; counter < numItrs; counter += blockDim.x) {
    bool isContributing = counter < numPossibleHypos;
    float score = -kMaxFloat;
    int parentIdx;
    int tokenIdx;
    if (isContributing) {
      parentIdx = counter / numTokens;
      tokenIdx = counter % numTokens;
      float parentScore =
          timeStep == 0 ? 0 : topKScores[uc * beamSize + parentIdx];
      int parentToken =
          timeStep == 0 ? -1 : topKTokens[uc * beamSize + parentIdx];
      if (parentToken == eos) {
        if (tokenIdx == 0) { // Only one thread should proceed.
          score = parentScore;
          tokenIdx = eos;
        } else {
          isContributing = false;
        }
      } else {
        int i = hyposCompactIndices[uc * beamSize + parentIdx];
        bool isValid =
            abs(bestPathDev[offset + i] - peakAttnPos[uc * beamSize + i]) <=
            attentionThreshold;
        if (isValid) {
          float amScore = amScores[(offset + i) * numTokens + tokenIdx];

          int lmTokenIdx;
          if (tokenIdx == eos) {
            lmTokenIdx = lmEosTokenIndex;
            score = eosScore;
          } else {
            lmTokenIdx = usrToLmIdxMap[tokenIdx];
            score = 0;
          }
          int oldIdx = (offset + i) * vocabSize + lmTokenIdx;
          int offset_ = (oldIdx / CC) * TT * CC;
          int idx_ =
              oldIdx % CC + offset_ + lastTokenPositions[oldIdx / CC] * CC;

          float lmScore = batchedProb[idx_];
          score += parentScore + amScore + lmWeight * lmScore;
        } else {
          isContributing = false;
        }
      }
    }

    __syncthreads();
    if (isContributing && (score > scoreHeap[1])) {
      int compactIdx = atomicAdd(&numNewScores, 1);
      blockWideScores[compactIdx] = score;
      blockWideParentIndices[compactIdx] = parentIdx;
      blockWideTokens[compactIdx] = tokenIdx;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      for (int i = 0; i < numNewScores; i++) {
        float score = blockWideScores[i];
        int parentIndex = blockWideParentIndices[i];
        int tokenIndex = blockWideTokens[i];
        if (score > bestScore) {
          bestScore = score;
        }
        if (numElemsInHeap < heapSize) {
          int k = numElemsInHeap++;
          k = k + 1; // one-based indexing
          scoreHeap[k] = score;
          parentIndicesHeap[k] = parentIndex;
          tokensHeap[k] = tokenIndex;
          while (k > 1 && scoreHeap[k] < scoreHeap[k / 2]) {
            auto tmp_score = scoreHeap[k / 2];
            scoreHeap[k / 2] = scoreHeap[k];
            scoreHeap[k] = tmp_score;

            auto tmpParentIdx = parentIndicesHeap[k / 2];
            parentIndicesHeap[k / 2] = parentIndicesHeap[k];
            parentIndicesHeap[k] = tmpParentIdx;

            auto tmpToken = tokensHeap[k / 2];
            tokensHeap[k / 2] = tokensHeap[k];
            tokensHeap[k] = tmpToken;

            k /= 2;
          }
        } else {
          if (score > scoreHeap[1]) {
            scoreHeap[1] = score;
            parentIndicesHeap[1] = parentIndex;
            tokensHeap[1] = tokenIndex;
            int k = 1;
            while ((2 * k) <= heapSize) {
              int parentIdx = 2 * k;
              if (parentIdx < heapSize) {
                if (scoreHeap[parentIdx] > scoreHeap[parentIdx + 1]) {
                  parentIdx++; // J is the smallest child
                }
              }
              if (scoreHeap[k] < scoreHeap[parentIdx]) {
                break;
              }
              auto tmp_score = scoreHeap[k];
              scoreHeap[k] = scoreHeap[parentIdx];
              scoreHeap[parentIdx] = tmp_score;

              auto tmpParentIdx = parentIndicesHeap[k];
              parentIndicesHeap[k] = parentIndicesHeap[parentIdx];
              parentIndicesHeap[parentIdx] = tmpParentIdx;

              auto tmpToken = tokensHeap[k];
              tokensHeap[k] = tokensHeap[parentIdx];
              tokensHeap[parentIdx] = tmpToken;

              k = parentIdx;
            }
          }
        }
      }
      numNewScores = 0;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    numElemsInHeapSmem = numElemsInHeap;
  }
  __syncthreads();
  if (threadIdx.x < numElemsInHeapSmem) {
    float score = scoreHeap[threadIdx.x + 1];
    if (score >= bestScore - beamThreshold) {
      int idx = atomicAdd(&numNewScores, 1);
      topKScores[uc * beamSize + idx] = score;
      topKParentIndices[uc * beamSize + idx] =
          parentIndicesHeap[threadIdx.x + 1];
      topKTokens[uc * beamSize + idx] = tokensHeap[threadIdx.x + 1];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int numNonEosHypos_counter = 0;
    numHypos[uc] = numNewScores;
    int index = uc * beamSize;
    for (int k = 0; k < numNewScores; k++) {
      int parentIdx = topKParentIndices[uc * beamSize + k];
      int tokenIdx = topKTokens[uc * beamSize + k];
      if (tokenIdx != eos) {
        int i = hyposCompactIndices[uc * beamSize + parentIdx];
        peakAttnPos[index] = bestPathDev[offset + i];
        tokensDevPtr[index] = tokenIdx;
        numNonEosHypos_counter++;
        index++;
      }
    }
    numNonEosHypos[uc] = numNonEosHypos_counter;
  }
}

__global__ void compactTokens(
    int* __restrict tokensDevPtr,
    const int* const __restrict numNonEosHypos,
    const int beamSize,
    const int numUtterances) {
  extern __shared__ int smem[];
  int* smemNumNonEosHypos = &smem[0]; // Size = numUtterances
  int* smemTokens = &smem[numUtterances]; // Size = numUtterances * beamSize

  for (int i = threadIdx.x; i < numUtterances; i += blockDim.x) {
    smemNumNonEosHypos[i] = numNonEosHypos[i];
  }
  __syncthreads();

  for (int counter = threadIdx.x; counter < numUtterances * beamSize;
       counter += blockDim.x) {
    int uc = counter / beamSize;
    int k = counter % beamSize;
    if (k < smemNumNonEosHypos[uc]) {
      int gmem_offset = uc * beamSize;
      int smem_offset = 0;
      for (int i = 0; i < uc; i++) {
        smem_offset += smemNumNonEosHypos[i];
      }
      smemTokens[smem_offset + k] = tokensDevPtr[gmem_offset + k];
    }
  }
  __syncthreads();

  int totalTokens = 0;
  for (int uc = 0; uc < numUtterances; uc++) {
    totalTokens += smemNumNonEosHypos[uc];
  }
  for (int i = threadIdx.x; i < totalTokens; i += blockDim.x) {
    tokensDevPtr[i] = smemTokens[i];
  }
}
} // namespace

namespace w2l {
namespace detail {
void calculateScores(
    const int eos,
    const int timeStep,
    const int numUtterances,
    const DecoderOptions& opt,
    std::vector<int>& finalized,
    std::vector<int>& numHypos,
    const int vocabSize,
    std::vector<int>& hyposCompactIndices,
    af::array& amScores,
    const int lmEosTokenIndex,
    std::vector<int>& lastTokenPositions,
    af::array& batchedLMScores,
    std::vector<float>& topKScores,
    std::vector<int>& topKParentIndices,
    std::vector<int>& topKTokens,
    af::array& bestPath,
    const int attentionThreshold,
    void* workspaceDevPtr,
    int* tokensDevPtr,
    const int* usrToLmIdxMapDevPtr) {
  const int maxNumUtterances = FLAGS_decoder_batch_size;
  const int& beamSize = opt.beamSize;
  const float& beamThreshold = opt.beamThreshold;
  const float& lmWeight = opt.lmWeight;
  const float& eosScore = opt.eosScore;
  const int CC = batchedLMScores.dims(0);
  const int TT = batchedLMScores.dims(1);
  const int numTokens = amScores.dims(0);

  int offset = 0;
  int* numHyposDevPtr = &reinterpret_cast<int*>(workspaceDevPtr)[offset];
  offset += maxNumUtterances;

  int* numNonEosHyposDevPtr = &reinterpret_cast<int*>(workspaceDevPtr)[offset];
  offset += maxNumUtterances;

  int* finalizedDevPtr = &reinterpret_cast<int*>(workspaceDevPtr)[offset];
  offset += maxNumUtterances;

  float* topKScoresDevPtr = &reinterpret_cast<float*>(workspaceDevPtr)[offset];
  offset += maxNumUtterances * beamSize;

  int* peakAttnPosDevPtr = &reinterpret_cast<int*>(workspaceDevPtr)[offset];
  offset += maxNumUtterances * beamSize;

  int* hyposCompactIndicesDevPtr =
      &reinterpret_cast<int*>(workspaceDevPtr)[offset];
  offset += maxNumUtterances * beamSize;

  int* lastTokenPositionsDevPtr =
      &reinterpret_cast<int*>(workspaceDevPtr)[offset];
  offset += maxNumUtterances * beamSize;

  int* topKParentIndicesDevPtr =
      &reinterpret_cast<int*>(workspaceDevPtr)[offset];
  offset += maxNumUtterances * beamSize;

  int* topKTokensDevPtr = &reinterpret_cast<int*>(workspaceDevPtr)[offset];
  offset += maxNumUtterances * beamSize;

  // Getting the device and the stream
  int deviceId = af::getDevice();
  cudaStream_t afCudaStream = afcu::getStream(deviceId);

  // Locking the Input Arrays
  float* amScoresDevPtr = amScores.device<float>();
  float* batchedProbsDevPtr = batchedLMScores.device<float>();
  int* bestPathDev = bestPath.device<int>();

  CHK_CUDA(cudaMemcpyAsync(
      finalizedDevPtr,
      finalized.data(),
      finalized.size() * sizeof(int),
      cudaMemcpyHostToDevice,
      afCudaStream));

  CHK_CUDA(cudaMemcpyAsync(
      hyposCompactIndicesDevPtr,
      hyposCompactIndices.data(),
      hyposCompactIndices.size() * sizeof(int),
      cudaMemcpyHostToDevice,
      afCudaStream));

  CHK_CUDA(cudaMemcpyAsync(
      lastTokenPositionsDevPtr,
      lastTokenPositions.data(),
      lastTokenPositions.size() * sizeof(int),
      cudaMemcpyHostToDevice,
      afCudaStream));

  const int& numBlocks = numUtterances;
  int sharedMemorySizeElems = (kNumThreads + beamSize + 1) * 3 + 1;
  int sharedMemorySizeBytes = sharedMemorySizeElems * sizeof(int);

  calculateScoresKernel<<<
      numBlocks,
      kNumThreads,
      sharedMemorySizeBytes,
      afCudaStream>>>(
      numHyposDevPtr,
      numNonEosHyposDevPtr,
      peakAttnPosDevPtr,
      tokensDevPtr,
      topKParentIndicesDevPtr,
      topKTokensDevPtr,
      topKScoresDevPtr,
      finalizedDevPtr,
      hyposCompactIndicesDevPtr,
      usrToLmIdxMapDevPtr,
      lastTokenPositionsDevPtr,
      bestPathDev,
      amScoresDevPtr,
      batchedProbsDevPtr,
      timeStep,
      CC,
      TT,
      lmEosTokenIndex,
      vocabSize,
      eos,
      beamSize,
      numTokens,
      attentionThreshold,
      eosScore,
      lmWeight,
      beamThreshold);

  sharedMemorySizeElems = numUtterances + numUtterances * beamSize;
  sharedMemorySizeBytes = sharedMemorySizeElems * sizeof(int);
  compactTokens<<<1, kNumThreads, sharedMemorySizeBytes, afCudaStream>>>(
      tokensDevPtr, numNonEosHyposDevPtr, beamSize, numUtterances);

  CHK_CUDA(cudaMemcpyAsync(
      topKScores.data(),
      topKScoresDevPtr,
      numUtterances * beamSize * sizeof(float),
      cudaMemcpyDeviceToHost,
      afCudaStream));

  CHK_CUDA(cudaMemcpyAsync(
      topKParentIndices.data(),
      topKParentIndicesDevPtr,
      numUtterances * beamSize * sizeof(int),
      cudaMemcpyDeviceToHost,
      afCudaStream));

  CHK_CUDA(cudaMemcpyAsync(
      topKTokens.data(),
      topKTokensDevPtr,
      numUtterances * beamSize * sizeof(int),
      cudaMemcpyDeviceToHost,
      afCudaStream));

  CHK_CUDA(cudaMemcpyAsync(
      numHypos.data(),
      numHyposDevPtr,
      numHypos.size() * sizeof(int),
      cudaMemcpyDeviceToHost,
      afCudaStream));

  // Unlocking the Arrays
  amScores.unlock();
  batchedLMScores.unlock();
  bestPath.unlock();
}
} // namespace detail
} // namespace w2l
