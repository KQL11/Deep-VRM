from functools import partial
from typing import Any, Dict, List, Literal, Optional

import torch
from transformers.integrations import is_deepspeed_zero3_enabled

from swift.llm import (Model, ModelGroup, ModelMeta, MultiModelKeys, Template, TemplateMeta, get_model_tokenizer,
                       get_model_tokenizer_with_flash_attn, get_packed_seq_params, get_template, register_model,
                       register_model_arch, register_template, to_float_dtype)
from swift.llm.model.model.qwen import patch_qwen_vl_utils
from swift.llm.model.patcher import patch_get_input_embeddings
from swift.llm.model.utils import use_submodel_func
from swift.llm.template.template_inputs import StdTemplateInputs
from swift.llm.template.utils import Context, findall
from swift.llm.template.vision_utils import load_audio
from swift.utils import get_env_args, get_logger, is_deepspeed_enabled

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

register_model_arch(
    MultiModelKeys(
        'my_qwen2_5_vl',
        # `freeze_llm`, `freeze_vit`, `freeze_aligner` behavior is determined by the values below.
        # For example: full parameter training, if `freeze_vit=True`, it will freeze parameters of
        # model layers prefixed with `thinker.audio_tower` and `thinker.visual`.
        # LoRA training, if `freeze_vit=False`, it will additionally add LoRA to Linear layers
        # prefixed with `thinker.audio_tower` and `thinker.visual`.
        language_model=['model.language_model', 'lm_head'],
        vision_tower=['model.visual_low_level', 'model.visual'],
        aligner=['model.visual_low_level.merger', 'model.visual.merger'],
        # Generator parts will never be trained or remain frozen.
        # generator=['talker', 'token2wav'],
    ))


def get_model_tokenizer_qwen2_5_vl(model_dir, *args, **kwargs):
    # from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, Qwen2_5OmniConfig
    # Avoid lazy __init__ export issues by importing directly from modules
    from DeepVRM import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, Qwen2_5_VLConfig
    # from .modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    # from .processing_qwen2_5_vl import Qwen2_5_VLProcessor
    # from .configuration_qwen2_5_vl import Qwen2_5_VLConfig
    from qwen_vl_utils import vision_process
    print('Run my_qwen2_5_vl...')
    kwargs['automodel_class'] = kwargs['automodel_class'] or Qwen2_5_VLForConditionalGeneration
    # Customize how to get tokenizer and config in `get_model_tokenizer_with_flash_attn`
    processor = Qwen2_5_VLProcessor.from_pretrained(model_dir, trust_remote_code=True)
    kwargs['tokenizer'] = processor.tokenizer
    kwargs['model_config'] = Qwen2_5_VLConfig.from_pretrained(model_dir, trust_remote_code=True)
    enable_audio_output = get_env_args('ENABLE_AUDIO_OUTPUT', bool, None)
    if enable_audio_output is not None:
        kwargs['model_config'].enable_audio_output = enable_audio_output
    # Control constants in qwen_omni_utils library via environment variables,
    # e.g., `MAX_PIXELS`, etc.
    patch_qwen_vl_utils(vision_process)
    # Recommended: Use this function to get model and tokenizer.
    # Avoid using AutoModelForCausalLM directly (may cause incompatibility).
    model, _ = get_model_tokenizer_with_flash_attn(model_dir, *args, **kwargs)
    if model:
        # For multimodal model consistency, we replace the model's forward/generate functions
        # with those of its language_model.
        # Handle additional parts separately.
        # use_submodel_func(model, 'model')
        # Some custom settings for model/config (usually not needed; configure based on
        # specific model if errors occur during training/inference)
        model.config.keys_to_ignore_at_inference += ['hidden_states', 'attention_mask']
        # model.config.talker_config.pad_token_id = None
        # Avoid inplace operations on leaf_variable during training
        # (replacing parts of input_embeds with images_embeds)
        patch_get_input_embeddings(model.model.visual, 'patch_embed')
    # Must return model and processor (multimodal) / tokenizer (text-only)
    return model, processor


register_model(
    ModelMeta(
        'my_qwen2_5_vl',
        [
            ModelGroup([
                Model('Qwen/Qwen2.5-VL-3B', 'Qwen/Qwen2.5-VL-3B'),
                Model('Qwen/Qwen2.5-VL-7B', 'Qwen/Qwen2.5-VL-7B'),
            ]),
        ],
        'my_qwen2_5_vl',
        # Function to get model and processor.
        get_model_tokenizer_qwen2_5_vl,
        is_multimodal=True,  # Whether it's a multimodal model
        model_arch='my_qwen2_5_vl',  # Usually set only for multimodal models
        # Used for automatic model_type matching
        architectures=['Qwen2_5_VLForConditionalGeneration'],
        # Used to prompt users about dependency versions (can be removed)
        requires=['transformers>=4.50', 'soundfile', 'qwen_vl_utils', 'decord'],
        # Used to prompt users (can be removed)
        tags=['vision', 'video'],
        # # Additional files to save during full parameter training/merge-lora
        # additional_saved_files=['spk_dict.pt'],
    ))
