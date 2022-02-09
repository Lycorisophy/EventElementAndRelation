# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
"""Convert ELECTRA checkpoint."""


import argparse

import torch

from language_model.transformers.configuration_electra import ElectraConfig
from language_model.transformers.modeling_electra import ElectraForMaskedLM, ElectraForPreTraining, load_tf_weights_in_electra
from language_model.transformers.utils import logging


logging.set_verbosity_info()


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, config_file, pytorch_dump_path, discriminator_or_generator):
    # Initialise PyTorch model
    config = ElectraConfig.from_json_file(config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))

    if discriminator_or_generator == "discriminator":
        model = ElectraForPreTraining(config)
    elif discriminator_or_generator == "generator":
        model = ElectraForMaskedLM(config)
    else:
        raise ValueError("The discriminator_or_generator argument should be either 'discriminator' or 'generator'")

    # Load weights from tf checkpoint
    load_tf_weights_in_electra(
        model, config, tf_checkpoint_path, discriminator_or_generator=discriminator_or_generator
    )

    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--tf_checkpoint_path",
        default='pretrained_model/electra_180g_large/',
        type=str,
        required=True,
        help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--config_file",
        default='pretrained_model/electra_180g_large/large_discriminator_config.json',
        type=str,
        required=True,
        help="The config json file corresponding to the pre-trained model. \n"
        "This specifies the model architecture.",
    )
    parser.add_argument(
        "--pytorch_dump_path",
        default='pretrained_model/pytorch_electra_180g_large/model.bin',
        type=str,
        required=True,
        help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--discriminator_or_generator",
        default='discriminator',
        type=str,
        required=True,
        help="Whether to export the generator or the discriminator. Should be a string, either 'discriminator' or "
        "'generator'.",
    )
    # args = parser.parse_args()
    # convert_tf_checkpoint_to_pytorch(
    #     args.tf_checkpoint_path, args.config_file, args.pytorch_dump_path, args.discriminator_or_generator
    # )
    convert_tf_checkpoint_to_pytorch(
        'D:/句子关系抽取/pretrained_model/electra_180g_large/electra_180g_large.ckpt',
        'D:/句子关系抽取/pretrained_model/electra_180g_large/large_discriminator_config.json',
        'D:/句子关系抽取/pretrained_model/pytorch_electra_180g_large/embedding.bin',
        'discriminator'
    )