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

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <float.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <numeric>

#include "BatchedLexiconFreeSeq2SeqDecoder.h"

namespace {
/**
 * Concatenates a vector of `fl::Variable` and returns the results. The
 * concatenation is performed along the `concatDim`. If the underlying arrays
 * do not have the same value in their `offsetDim` axis, the larges value will
 * be chosen for that dimension and the rest will be padded by `padValue`.
 *
 * @param inputs Input. A vector of flashlight Variables.
 * @param concatDim The dimension along which the Variables are concatenated.
 * @param offsetDim If `Variables` differ in this dimension, they will be
 * padded.
 * @param padValue The value that is used for padding smaller Variables.
 * @param skip If true, the corresponding `Variable` in the `inputs` will not be
 * included in the concatenation.
 *
 * @return The concatenated `Variable`.
 */
template <typename T>
fl::Variable padAndConcat(
    const std::vector<fl::Variable>& inputs,
    const int concatDim,
    const int offsetDim,
    const std::vector<int>& skip,
    const T padValue) {
  if (inputs.empty()) {
    throw std::invalid_argument("Cannot concatenate zero variables");
  }
  if (inputs.size() == 1) {
    return inputs[0];
  }
  int uc = 0;
  while (skip[uc] && uc < inputs.size()) {
    uc++;
  }
  if (uc == inputs.size()) {
    throw std::invalid_argument("Cannot skip all the input Variables.");
  }

  auto dims = inputs[uc].dims();
  int concatSize = dims[concatDim];
  int maxSize = dims[offsetDim];
  for (int i = uc + 1; i < inputs.size(); i++) {
    if (skip[i]) {
      continue;
    }
    concatSize += inputs[i].dims(concatDim);
    maxSize = std::max(maxSize, (int)inputs[i].dims(offsetDim));
  }

  dims[concatDim] = concatSize;
  dims[offsetDim] = maxSize;
  auto result = af::constant(padValue, dims, inputs[uc].type());
  std::array<af::index, 4> slice{af::span, af::span, af::span, af::span};
  uc = -1;
  int start = 0;
  for (const auto& input : inputs) {
    uc++;
    if (skip[uc]) {
      continue;
    }
    slice[concatDim] = af::seq(start, start + input.dims(concatDim) - 1);
    slice[offsetDim] = af::seq(0, input.dims(offsetDim) - 1);
    result(slice[0], slice[1], slice[2], slice[3]) = input.array();
    start += input.dims(concatDim);
  }

  return fl::Variable(result, /* calcGrad */ false);
}

/**
 * Concatenates a vector of `fl::Variables` and returns the results. The
 * concatenation is performed along the `dim` axis. Throws an error if other
 * dimensions are not identical.
 *
 * @param inputs Input. A vector of flashlight Variables.
 * @param dim The dimension along which the Variables are concatenated.
 * @param skip If true, the corresponding `Variable` in the `inputs` will not be
 * included in the concatenation.
 *
 * @return The concatenated `Variable`.
 */
fl::Variable concat(
    const std::vector<fl::Variable>& inputs,
    int dim,
    const std::vector<int>& skip) {
  if (inputs.empty()) {
    throw std::invalid_argument("Cannot concatenate zero variables");
  }
  if (dim < 0 || dim > 3) {
    throw std::invalid_argument("Invalid dimension to concatenate along");
  }
  if (inputs.size() == 1) {
    return inputs[0];
  }
  int uc = 0;
  while (skip[uc] && uc < inputs.size()) {
    uc++;
  }
  if (uc == inputs.size()) {
    throw std::invalid_argument("Cannot skip all the input Variables.");
  }

  auto dims = inputs[uc].dims();
  int concatSize = dims[dim];
  for (int i = uc + 1; i < inputs.size(); i++) {
    if (skip[i]) {
      continue;
    }
    concatSize += inputs[i].dims(dim);
    for (int d = 0; d < 4; d++) {
      if (dim != d && inputs[i].dims(d) != dims[d]) {
        throw std::invalid_argument(
            "Mismatch in dimension not being concatenated");
      }
    }
  }
  dims[dim] = concatSize;
  af::array result(dims, inputs[uc].type());
  std::array<af::index, 4> slice{af::span, af::span, af::span, af::span};
  int start = 0;
  uc = -1;
  for (const auto& input : inputs) {
    uc++;
    if (skip[uc]) {
      continue;
    }
    slice[dim] = af::seq(start, start + input.dims(dim) - 1);
    result(slice[0], slice[1], slice[2], slice[3]) = input.array();
    start += input.dims(dim);
  }

  return fl::Variable(result, /* calcGrad */ false);
}
} // namespace

namespace w2l {
void BatchedLexiconFreeSeq2SeqDecoder::decodeStep(
    std::vector<const float*>& emissions,
    std::vector<int>& T_,
    int N) {
  // Initialization
  numUtterances_ = emissions.size();
  std::fill(finalized_.begin(), finalized_.end(), false);
  std::fill(numHypos_.begin(), numHypos_.end(), 1);
  std::fill(numNonEosHypos_.begin(), numNonEosHypos_.end(), 1);

  auto lastTokenPosition = initialLmState_->length_ - 1;
  lastTokenPositions_.resize(numUtterances_);
  std::fill(
      lastTokenPositions_.begin(),
      lastTokenPositions_.end(),
      lastTokenPosition);

  batchedLMScoresHost_.resize(numUtterances_ * lmVocabSize_);
  for (int uc = 0; uc < numUtterances_; uc++) {
    input_[uc] = fl::input(af::array(N, T_[uc], emissions[uc]));
    lmStates_[uc][0] = initialLmState_;
    std::memcpy(
        &batchedLMScoresHost_[uc * lmVocabSize_],
        initialLmScore_.data(),
        initialLmScore_.size() * sizeof(float));
  }
  auto batchedLMScores = af::array(
      initialLmScore_.size(),
      1,
      numUtterances_,
      batchedLMScoresHost_.data());

  fl::Variable embeddingVectors = initialEmbeddings;

  std::vector<fl::Variable> rnnOutputs(numUtterances_);
  fl::Variable batchedInputRnnStates;
  for (int timeStep = 0; timeStep < maxOutputLength_; timeStep++) {
    std::fill(
        candidatesBestScore_.begin(),
        candidatesBestScore_.end(),
        kNegativeInfinity);
    std::vector<fl::Variable> summaries(numUtterances_);
    std::vector<fl::Variable> alphas(numUtterances_);

    for (int n = 0; n < numAttnRounds_; n++) {
      // Decode RNN
      fl::Variable batchedRnnOutputs, batchedOutputRnnStates;
      std::tie(batchedRnnOutputs, batchedOutputRnnStates) =
          s2sCriterion_->decodeRNN(n)->forward(
              embeddingVectors, batchedInputRnnStates);

      int totalNonEosHypos = 0;
      for (int uc = 0; uc < numUtterances_; uc++) {
        if (finalized_[uc])
          continue;
        for (int i = 0; i < numNonEosHypos_[uc]; i++) {
          rnnHiddenStates_[uc][i * numAttnRounds_ + n] =
              batchedOutputRnnStates.col(totalNonEosHypos + i);
        }
        rnnOutputs[uc] = batchedRnnOutputs.cols(
            totalNonEosHypos, totalNonEosHypos + numNonEosHypos_[uc] - 1);
        totalNonEosHypos += numNonEosHypos_[uc];
      }

      // Attention
      for (int uc = 0; uc < numUtterances_; uc++) {
        if (finalized_[uc])
          continue;
        std::tie(alphas[uc], summaries[uc]) =
            s2sCriterion_->attention(n)->forward(
                rnnOutputs[uc], input_[uc], fl::Variable(), fl::Variable());
        rnnOutputs[uc] = rnnOutputs[uc] + summaries[uc];
      }
    }

    // Concatenating the results
    auto batchedAlphas = padAndConcat(
        /* inputs */ alphas,
        /* concatDim */ 0,
        /* offsetDim */ 1,
        /* skip */ finalized_,
        /* padValue */ -std::numeric_limits<float>::max());

    af::array bestPath, maxValues;
    af::max(maxValues, bestPath, batchedAlphas.array(), 1);
    auto batchedAttnOutputs = concat(rnnOutputs, 1, finalized_);

    // Linear
    auto batchedLinearOutputs =
        s2sCriterion_->linearOut()->forward(batchedAttnOutputs);

    batchedLinearOutputs =
        logSoftmax(batchedLinearOutputs / smoothingTemperature_, 0);
    
    w2l::detail::calculateScores(
        eos_,
        timeStep,
        numUtterances_,
        opt_,
        finalized_,
        numHypos_,
        lmVocabSize_,
        hyposCompactIndices_,
        batchedLinearOutputs.array(),
        lmEosTokenIndex_,
        lastTokenPositions_,
        batchedLMScores,
        topKScores_,
        topKParentIndices_,
        topKTokens_,
        bestPath,
        attentionThreshold_,
        workspaceDevPtr_,
        batchedTokensDevPtr_,
        userToLmIndexMapDevPtr_);

    int longestHistory = -1;
    int totalNumNonEosHypos = 0;
    for (int uc = 0; uc < numUtterances_; uc++) {
      if (finalized_[uc])
        continue;
      int utteranceOffset = uc * opt_.beamSize;
      int index = utteranceOffset;
      for (int i = 0; i < numHypos_[uc]; i++) {
        if (topKTokens_[utteranceOffset + i] != eos_) {
          auto parentIdx = topKParentIndices_[utteranceOffset + i];
          auto parentCompactIdx =
              hyposCompactIndices_[utteranceOffset + parentIdx];
          parentCompactIndices_[index++] = parentCompactIdx;

          BatchedLexiconFreeSeq2SeqLmState* lmInputState =
              lmStates_[uc][parentCompactIdx].get();

          int length;
          std::vector<int>::iterator begin;
          if (lmInputState->length_ == lmMaxHistorySize_) {
            length = lmMaxHistorySize_;
            begin = lmInputState->tokens_.begin() + 1;
          } else {
            length = lmInputState->length_ + 1;
            begin = lmInputState->tokens_.begin();
          }

          auto lmOutputState =
              std::make_shared<BatchedLexiconFreeSeq2SeqLmState>(length);
          std::copy(
              begin,
              lmInputState->tokens_.end(),
              lmOutputState->tokens_.begin());
          lmOutputState->tokens_[length - 1] =
              userToLmIndexMap_[topKTokens_[utteranceOffset + i]];
          tmpLmStates_[uc][i] = lmOutputState;

          if (lmOutputState->length_ > longestHistory) {
            longestHistory = lmOutputState->length_;
          }
          totalNumNonEosHypos++;
        }
      }
    }

    int offset = 0;
    lastTokenPositions_.resize(0);
    for (int uc = 0; uc < numUtterances_; uc++) {
      if (finalized_[uc])
        continue;
      for (int i = 0; i < numHypos_[uc]; i++) {
        if (topKTokens_[uc * opt_.beamSize + i] != eos_) {
          std::memcpy(
              batchedLmTokens_.data() + offset,
              tmpLmStates_[uc][i]->tokens_.data(),
              tmpLmStates_[uc][i]->length_ * sizeof(float));

          auto startPadIndex =
              batchedLmTokens_.data() + offset + tmpLmStates_[uc][i]->length_;
          auto endPadIndex = startPadIndex + longestHistory;
          std::fill(startPadIndex, endPadIndex, lmPadTokenIndex_);

          lastTokenPositions_.push_back(tmpLmStates_[uc][i]->length_ - 1);
          offset += longestHistory;
        }
      }
    }

    if (totalNumNonEosHypos > 0) {
      auto batchedLmTokensVariable = fl::input(af::array(
          longestHistory, totalNumNonEosHypos, batchedLmTokens_.data()));
      batchedLMScores = convLm_->forward({batchedLmTokensVariable})[0].array();

      auto tmp = fl::input(batchedTokens_.rows(0, totalNumNonEosHypos - 1));
      embeddingVectors = s2sCriterion_->embedding()->forward(tmp);
    }

    int timeOffset = (timeStep + 1) * opt_.beamSize;
    for (int uc = 0; uc < numUtterances_; uc++) {
      if (finalized_[uc])
        continue;

      int counter = 0;
      int utteranceOffset = uc * opt_.beamSize;
      for (int i = 0; i < numHypos_[uc]; i++) {
        finalCandidates_[uc][timeOffset + i].token_ =
            topKTokens_[utteranceOffset + i];
        finalCandidates_[uc][timeOffset + i].score_ =
            topKScores_[utteranceOffset + i];
        finalCandidates_[uc][timeOffset + i].parentIndices_ =
            topKParentIndices_[utteranceOffset + i];

        if (topKTokens_[utteranceOffset + i] != eos_) {
          lmStates_[uc][counter] = tmpLmStates_[uc][i];
          hyposCompactIndices_[utteranceOffset + i] = counter;
          counter++;
        }
      }
      numNonEosHypos_[uc] = counter;
    }

    for (int uc = 0; uc < numUtterances_; uc++) {
      if (finalized_[uc])
        continue;
      if (numNonEosHypos_[uc] == 0) {
        if (numHypos_[uc] == 0) {
          finalTimeStep_[uc] = timeStep;
          finalNumHypos_[uc] = numHyposOld_[uc];
        } else {
          finalTimeStep_[uc] = timeStep + 1;
          finalNumHypos_[uc] = numHypos_[uc];
        }
        finalized_[uc] = true;
      }
      numHyposOld_[uc] = numHypos_[uc];
    }

    bool quit = true;
    for (int uc = 0; uc < numUtterances_ && quit; uc++) {
      if (finalized_[uc])
        continue;
      quit = false;
    }
    if (quit) {
      break;
    }

    for (int n = 0; n < numAttnRounds_; n++) {
      nonEosRnnStateVectors_.resize(0);
      for (int uc = 0; uc < numUtterances_; uc++) {
        if (finalized_[uc])
          continue;
        for (int i = 0; i < numNonEosHypos_[uc]; i++) {
          nonEosRnnStateVectors_.push_back(
              rnnHiddenStates_[uc]
                              [parentCompactIndices_[uc * opt_.beamSize + i] *
                                   numAttnRounds_ +
                               n]);
        }
      }
      batchedInputRnnStates = concatenate(nonEosRnnStateVectors_, 1);
    }
  } // for (timeStep)

  for (int uc = 0; uc < numUtterances_; uc++) {
    std::sort(
        finalCandidates_[uc].begin() + finalTimeStep_[uc] * opt_.beamSize,
        finalCandidates_[uc].begin() + finalTimeStep_[uc] * opt_.beamSize +
            finalNumHypos_[uc],
        compareNodeScore);
  }
}

std::vector<std::vector<DecodeResult>>
BatchedLexiconFreeSeq2SeqDecoder::getAllFinalHypothesis() const {
  std::vector<std::vector<DecodeResult>> output(numUtterances_);
  for (int uc = 0; uc < numUtterances_; uc++) {
    output[uc] = std::vector<DecodeResult>(finalNumHypos_[uc]);
    for (int r = 0; r < finalNumHypos_[uc]; r++) {
      DecodeResult result(maxOutputLength_ + 1);
      int parentIdx = r;
      int idx = 0;
      for (int timeStep = finalTimeStep_[uc]; timeStep >= 0;
           timeStep--, idx++) {
        result.tokens[maxOutputLength_ - idx] =
            finalCandidates_[uc][timeStep * opt_.beamSize + parentIdx].token_;
        parentIdx = finalCandidates_[uc][timeStep * opt_.beamSize + parentIdx]
                        .parentIndices_;
      }
      output[uc][r] = result;
    }
  }
  return output;
}

DecodeResult BatchedLexiconFreeSeq2SeqDecoder::getBestHypothesis(
    int /* unused */) const {}

void BatchedLexiconFreeSeq2SeqDecoder::prune(int /* unused */) {
  return;
}

int BatchedLexiconFreeSeq2SeqDecoder::nDecodedFramesInBuffer() const {
  /* unused function */
  return -1;
}

} // namespace w2l
