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

#include <float.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>


#include <arrayfire.h>
#include <Utils.h>

namespace w2l {
namespace detail {
void calculateScores(
    const int eos,
    const int timeStep,
    const int num_utterances,
    const DecoderOptions& opt,
    std::vector<int>& finalized,
    std::vector<int>& numHypos,
    const int vocab_size,
    std::vector<int>& beam_members_non_eos_compact_idx,
    af::array& amScores,
    const int kLmEosTokenIndex,
    std::vector<int>& lastTokenPositions,
    af::array& batchedLMScores,
    std::vector<float>& top_k_scores,
    std::vector<int>& topKParentIndices,
    std::vector<int>& topKTokens,
    af::array& bestPath,
    const int attentionThreshold_,
    void* workspace,
    int* tokens_dev,
    const int* user_to_lm_index_map_dev_) {
  std::cout << "BatchedLexiconFreeSeq2SeqDecoder is not supported in CPU yet."
            << std::endl;
  std::exit(0);
}
} // namespace detail
} // namespace w2l
