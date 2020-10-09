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

#pragma once

#include <iostream>
#include <memory>
#include <unordered_map>

#include "criterion/Seq2SeqCriterion.h"
#include "libraries/decoder/Decoder.h"
#define NUM_TIMERS 6

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
    std::vector<int>& hyposCompactIndices_,
    af::array& amScores,
    const int lmEosTokenIndex,
    std::vector<int>& lastTokenPositions,
    af::array& batchedLMScores,
    std::vector<float>& topKScores,
    std::vector<int>& topKParentIndices,
    std::vector<int>& topKTokens,
    af::array& bestPath,
    const int attentionThreshold,
    void* workspace,
    int* tokensDevPtr,
    const int* userToLmIndexMapDevPtr);
}

struct BatchedLexiconFreeSeq2SeqDecoderState {
  int parentIndices_ = -1;
  int token_;
  float score_;
  BatchedLexiconFreeSeq2SeqDecoderState()
      : parentIndices_(-1), token_(-1), score_(0) {}
};

struct BatchedLexiconFreeSeq2SeqLmState {
  std::vector<int> tokens_;
  int length_;

  BatchedLexiconFreeSeq2SeqLmState() : length_(0) {}
  explicit BatchedLexiconFreeSeq2SeqLmState(int size)
      : tokens_(std::vector<int>(size)), length_(size) {}
};

class BatchedLexiconFreeSeq2SeqDecoder : public Decoder {
 private:
  const int attentionThreshold_ = FLAGS_attentionthreshold;
  const int eos_;
  const int maxNumUtterances_ = FLAGS_decoder_batch_size;
  const int maxOutputLength_;

  const float smoothingTemperature_ = FLAGS_smoothingtemperature;

  int lmEosTokenIndex_;
  int lmMaxHistorySize_;
  int lmPadTokenIndex_;
  int lmVocabSize_;
  int numAttnRounds_;
  int numUtterances_;

  int* userToLmIndexMapDevPtr_;
  int* batchedTokensDevPtr_;
  
  void* workspaceDevPtr_;

  const Seq2SeqCriterion* s2sCriterion_;

  std::shared_ptr<BatchedLexiconFreeSeq2SeqLmState> initialLmState_;
  std::shared_ptr<fl::Module> convLm_;

  std::vector<int> batchedLmTokens_;
  std::vector<int> finalized_;
  std::vector<int> finalNumHypos_;
  std::vector<int> finalParentIndices_;
  std::vector<int> finalTimeStep_;
  std::vector<int> finalTokens_;
  std::vector<int> hyposCompactIndices_;
  std::vector<int> lastTokenPositions_;
  std::vector<int> topKParentIndices_;
  std::vector<int> topKTokens_;
  std::vector<int> userToLmIndexMap_;
  std::vector<int> numHypos_;
  std::vector<int> numHyposOld_;
  std::vector<int> numNonEosHypos_;
  std::vector<int> parentCompactIndices_;

  std::vector<float> batchedLMScoresHost_;
  std::vector<float> candidatesBestScore_;
  std::vector<float> finalScores_;
  std::vector<float> initialLmScore_;
  std::vector<float> topKScores_;

  std::vector<fl::Variable> input_;
  std::vector<fl::Variable> nonEosRnnStateVectors_;

  std::vector<std::vector<fl::Variable>> rnnHiddenStates_;
  std::vector<std::vector<std::shared_ptr<BatchedLexiconFreeSeq2SeqLmState>>>
      lmStates_;
  std::vector<std::vector<std::shared_ptr<BatchedLexiconFreeSeq2SeqLmState>>>
      tmpLmStates_;
  std::vector<std::vector<BatchedLexiconFreeSeq2SeqDecoderState>>
      finalCandidates_;

  af::array batchedTokens_;
  af::array userToLmIndexMapArray_;
  af::array workspace_;

  fl::Variable initialEmbeddings;

  static bool compareNodeScore(
      const BatchedLexiconFreeSeq2SeqDecoderState& node1,
      const BatchedLexiconFreeSeq2SeqDecoderState& node2) {
    return node1.score_ > node2.score_;
  }

 public:
  BatchedLexiconFreeSeq2SeqDecoder(
      const std::string& tokenVocabPath,
      const Dictionary& usrTknDict,
      const std::shared_ptr<fl::Module> convLm,
      const DecoderOptions& opt,
      const int eos,
      const int maxOutputLength,
      const std::shared_ptr<SequenceCriterion> criterion,
      const int historySize = 49)
      : Decoder(opt),
        convLm_(convLm),
        eos_(eos),
        maxOutputLength_(maxOutputLength) {
    // Computing initial LM state and its scores
    // A. Initialization
    auto languageModelVocab = Dictionary(tokenVocabPath);
    languageModelVocab.setDefaultIndex(languageModelVocab.getIndex(kUnkToken));

    // B. Get what we need from Language Model Vocab
    lmPadTokenIndex_ = languageModelVocab.getIndex(kLmPadToken);
    lmVocabSize_ = languageModelVocab.indexSize();
    lmMaxHistorySize_ = historySize;
    lmEosTokenIndex_ = languageModelVocab.getIndex(kLmEosToken);

    // C. Initial Language Model State
    initialLmState_ = std::make_shared<BatchedLexiconFreeSeq2SeqLmState>(1);
    initialLmState_->tokens_[0] = lmEosTokenIndex_;

    // D. Initial Language Model Score
    auto sampleSize = initialLmState_->tokens_.size();
    int batchSize = 1;
    auto input = fl::input(
        af::array(sampleSize, batchSize, initialLmState_->tokens_.data()));
    auto output = convLm_->forward({input})[0];
    lastTokenPositions_ = {initialLmState_->length_ - 1};
    initialLmScore_ = afToVector<float>(output);

    // E. Creating User to LM Map
    userToLmIndexMap_.resize(usrTknDict.indexSize());
    for (int i = 0; i < usrTknDict.indexSize(); i++) {
      auto token = usrTknDict.getEntry(i);
      int lmIdx = languageModelVocab.getIndex(token.c_str());
      userToLmIndexMap_[i] = lmIdx;
    }
    userToLmIndexMapArray_ =
        af::array(userToLmIndexMap_.size(), userToLmIndexMap_.data());
    userToLmIndexMapDevPtr_ = userToLmIndexMapArray_.device<int>();

    // Create Criterion
    s2sCriterion_ = static_cast<Seq2SeqCriterion*>(criterion.get());
    numAttnRounds_ = s2sCriterion_->getNumberAttnRounds();
    if (s2sCriterion_->getInputFeeding()) {
      std::cout << "Input Feeding is disabled" << std::endl;
      exit(0);
    }
    fl::Variable embeddingOut = s2sCriterion_->startEmbedding();
    embeddingOut = moddims(embeddingOut, {embeddingOut.dims(0), -1});
    std::vector<fl::Variable> tmp;
    for (int uc = 0; uc < maxNumUtterances_; uc++) {
      tmp.push_back(embeddingOut);
    }
    initialEmbeddings = fl::concatenate(tmp, 1);

    // Host-side Memory Allocations
    topKTokens_ = std::vector<int>(maxNumUtterances_ * opt.beamSize, 0);
    topKScores_ = std::vector<float>(maxNumUtterances_ * opt.beamSize, 0);
    topKParentIndices_ = std::vector<int>(maxNumUtterances_ * opt.beamSize, 0);
    candidatesBestScore_ = std::vector<float>(maxNumUtterances_);

    finalized_ = std::vector<int>(maxNumUtterances_, false);
    numHypos_ = std::vector<int>(maxNumUtterances_);
    finalNumHypos_ = std::vector<int>(maxNumUtterances_);
    parentCompactIndices_ = std::vector<int>(maxNumUtterances_ * opt.beamSize);
    numHyposOld_ = std::vector<int>(maxNumUtterances_);

    numNonEosHypos_ = std::vector<int>(maxNumUtterances_, 1);
    nonEosRnnStateVectors_ =
        std::vector<fl::Variable>(maxNumUtterances_ * opt.beamSize);
    input_ = std::vector<fl::Variable>(maxNumUtterances_);

    batchedLMScoresHost_ = std::vector<float>(maxNumUtterances_ * lmVocabSize_);
    batchedLmTokens_ =
        std::vector<int>(opt.beamSize * maxNumUtterances_ * maxOutputLength_);

    lmStates_ = std::vector<
        std::vector<std::shared_ptr<BatchedLexiconFreeSeq2SeqLmState>>>(
        maxNumUtterances_,
        std::vector<std::shared_ptr<BatchedLexiconFreeSeq2SeqLmState>>(
            opt.beamSize));
    tmpLmStates_ = std::vector<
        std::vector<std::shared_ptr<BatchedLexiconFreeSeq2SeqLmState>>>(
        maxNumUtterances_,
        std::vector<std::shared_ptr<BatchedLexiconFreeSeq2SeqLmState>>(
            opt.beamSize));

    rnnHiddenStates_ = std::vector<std::vector<fl::Variable>>(
        maxNumUtterances_,
        std::vector<fl::Variable>(
            opt.beamSize * s2sCriterion_->getNumberAttnRounds()));

    finalCandidates_ =
        std::vector<std::vector<BatchedLexiconFreeSeq2SeqDecoderState>>(
            maxNumUtterances_,
            std::vector<BatchedLexiconFreeSeq2SeqDecoderState>(
                maxOutputLength_ * opt.beamSize));
    finalTimeStep_ = std::vector<int>(maxNumUtterances_);

    hyposCompactIndices_ = std::vector<int>(maxNumUtterances_ * opt.beamSize);

    // GPU-side Memory Allocations
    int workspaceSize = maxNumUtterances_ + // Num Hypos
        maxNumUtterances_ + // Num Non EOS Hypos
        maxNumUtterances_ + // Finalized
        maxNumUtterances_ * opt_.beamSize + // Top K scores
        maxNumUtterances_ * opt_.beamSize + // Top K parent indices
        maxNumUtterances_ * opt_.beamSize + // Top K tokens
        maxNumUtterances_ * opt_.beamSize + // Peak ATTN Positions
        maxNumUtterances_ * opt_.beamSize + // Beam Members Compact IDX
        maxNumUtterances_ * opt_.beamSize; // Last Token Position
    workspace_ = af::array(workspaceSize, s32);
    workspaceDevPtr_ = workspace_.device<void>();

    batchedTokens_ = af::array(maxNumUtterances_ * opt.beamSize, s32);
    batchedTokensDevPtr_ = batchedTokens_.device<int>();
  }

  ~BatchedLexiconFreeSeq2SeqDecoder() {
    workspace_.unlock();
    userToLmIndexMapArray_.unlock();
    batchedTokens_.unlock();
  }

  void decodeStep(
      std::vector<const float*>& emissions,
      std::vector<int>& T,
      int N) override;

  void prune(int lookBack = 0) override;

  int nDecodedFramesInBuffer() const override;

  DecodeResult getBestHypothesis(int lookBack = 0) const override;

  std::vector<std::vector<DecodeResult>> getAllFinalHypothesis() const override;
};

} // namespace w2l
