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

import os
import re
from contextlib import contextmanager

import paddle

from paddlenlp.transformers.model_utils import load_sharded_checkpoint

try:
    import transformer_engine.paddle as te

    _IS_TRANSFORMER_ENGINE_INSTALLED = True

except ModuleNotFoundError:
    _IS_TRANSFORMER_ENGINE_INSTALLED = False


def get_filename_for_this_rank(filenames, rank):
    # split filename into 3 parts
    for filename in filenames:
        filename_split = filename.split(".")
        assert len(filename_split) == 3, "filename must be like model_state.tp01.pdparams"
        file_rank = int(filename_split[1][2:])
        # file_rank = int(filename_split[1])

        if file_rank == rank:
            return filename


def check_params_valid(state_dict, state_dict_original):
    for name, param in state_dict.items():
        if name not in state_dict_original:
            raise ValueError("The input checkpoint is not a valid checkpoint.")
        if param.shape != state_dict_original[name].shape:
            raise ValueError("The input checkpoint is not a valid checkpoint.")


class TransformerEngineHelper:
    @staticmethod
    def is_installed():
        return _IS_TRANSFORMER_ENGINE_INSTALLED

    @staticmethod
    def get_transformer_layer():
        assert (
            TransformerEngineHelper.is_installed()
        ), "TransformerEngine is not installed. Please install it first or disable it."
        return te.TransformerLayer

    @staticmethod
    def get_te_recompute_func():
        assert (
            TransformerEngineHelper.is_installed()
        ), "TransformerEngine is not installed. Please install it first or disable it."
        return te.recompute

    @staticmethod
    @contextmanager
    def fp8_autocast(enabled=False):
        if TransformerEngineHelper.is_installed():
            with te.fp8_autocast(enabled=enabled):
                yield
        else:  # null context
            yield

    @staticmethod
    def te_init_weights(layer, config):
        if not TransformerEngineHelper.is_installed():
            return
        if isinstance(
            layer,
            (
                te.LayerNormLinear,
                te.Linear,
            ),
        ):
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=config.initializer_range,
                        shape=layer.weight.shape,
                    )
                )

        if isinstance(layer, te.LayerNormMLP):
            if isinstance(layer.fc1_weight, paddle.Tensor):
                layer.fc1_weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=config.initializer_range,
                        shape=layer.fc1_weight.shape,
                    )
                )
            if isinstance(layer.fc2_weight, paddle.Tensor):
                layer.fc2_weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=config.initializer_range,
                        shape=layer.fc2_weight.shape,
                    )
                )

    @staticmethod
    def reset_te_init_weights(model, ckpt_path, config):
        if not TransformerEngineHelper.is_installed() or ckpt_path is None:
            return

        # get files from ckpt_path
        ckpt_files = os.listdir(ckpt_path)
        ckpt_files = [x for x in ckpt_files if x.endswith(".pdparams")]
        if config.tensor_parallel_degree > 1:
            assert (
                len(ckpt_files) == config.tensor_parallel_degree
            ), "number of ckpt files must be equal to tensor_parallel_degree"
            filename = get_filename_for_this_rank(ckpt_files, config.tensor_parallel_rank)
        elif len(ckpt_files) > 1:
            # sharded ckpt. For example: model_state-00001-of-00002.pdparams
            # check the filename, must has 0000x-of-0000x pattern in filename
            for filename in ckpt_files:
                result = re.search(r"\d+-of-\d+", filename)
                assert result is not None, "filename must has 0000x-of-0000x pattern"
            missing_keys, unexpected_keys = load_sharded_checkpoint(model, ckpt_path)
            assert len(missing_keys) == 0, "missing keys must be empty"
            assert len(unexpected_keys) == 0, "unexpected keys must be empty"
            return
        else:
            assert len(ckpt_files) == 1, "number of ckpt files must be equal to 1"
            filename = ckpt_files[0]

        state_dict = paddle.load(os.path.join(ckpt_path, filename))
        original_state_dict = model.state_dict()
        check_params_valid(state_dict, original_state_dict)
        for name, param in state_dict.items():
            # print(f"TE Layer: {name} | Size: {param.shape} | dtype: {param.dtype}")
            state_dict[name] = param
        model.set_state_dict(state_dict)
