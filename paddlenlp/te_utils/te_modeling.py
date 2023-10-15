# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle.nn as nn

from .te_helper import TransformerEngineHelper


class GPTDecoderLayerWithNVTEBackend(nn.Layer):
    """
    The transformer decoder layer using Transformer Backend.
    """

    def __init__(self, config):

        super(GPTDecoderLayerWithNVTEBackend, self).__init__()

        self.config = config

        TransformerLayer = TransformerEngineHelper.get_transformer_layer()
        self.transformer = TransformerLayer(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            hidden_dropout=config.hidden_dropout_prob,
            attention_dropout=config.attention_probs_dropout_prob,
            self_attn_mask_type="causal",
            layer_type="encoder",
            activation=config.hidden_activation,
            set_parallel_mode=config.tensor_parallel_degree > 1,
            backend=config.transformer_engine_backend,
        )

    def forward(
        self, hidden_states, attention_mask=None, use_cache=False, past_key_value=None, output_attentions=False
    ):
        return self.transformer(
            hidden_states,
            attention_mask,
            recompute_core_attention=(self.config.use_recompute and self.config.recompute_granularity == "core_attn"),
        )
