import os
from typing import TYPE_CHECKING, Optional

import safetensors.torch
import torch

# from swift.llm import deep_getattr, get_multimodal_target_regex
from swift.llm import deep_getattr
from swift.plugin import Tuner, extra_tuners
from swift.tuners import LoraConfig, Swift
from swift.utils import get_logger
from swift.utils import activate_parameters, find_all_linears, find_embedding, find_norm, freeze_parameters, get_logger
from torch import nn

logger = get_logger()
if TYPE_CHECKING:
    from swift.llm import TrainArguments


def is_vit_param(model_arch, parameter_name: str) -> bool:
    for module_prefix in model_arch.vision_tower + model_arch.aligner:
        if "visual_low_level" in module_prefix:    
            if f'.{module_prefix}.' in parameter_name:
                return True
    return False


# NOTE: Set the trainable paramaters for multimodal models (by LoRA)
def get_multimodal_target_regex(
    model,
    *,
    freeze_llm: bool = False,
    freeze_vit: bool = True,
    freeze_aligner: bool = True,
    include_embedding: bool = False,
    exclude_router: bool = False,
) -> str:
    model_arch = model.model_meta.model_arch
    modules = []
    if not freeze_llm:
        modules += model_arch.language_model
    if not freeze_vit:
        modules += model_arch.vision_tower
    if not freeze_aligner:
        modules += model_arch.aligner
    assert len(modules) > 0, f'modules: {modules}'

    extra_layers = []
    if include_embedding:
        extra_layers.append(nn.Embedding)
    res = []
    # NOTE:
    target_modules_dict = {}
    
    for module in modules:
        rejected_modules = []
        if not freeze_vit or not freeze_llm:
            for aligner in model_arch.aligner:
                if aligner.startswith(f'{module}.'):
                    rejected_modules.append(aligner)

        sub_module = deep_getattr(model, module)
        if isinstance(sub_module, nn.Linear) and module.endswith('lm_head'):
            target_modules = []
        else:
            target_modules = find_all_linears(sub_module, model_arch, extra_layers)
        if exclude_router and model.model_info.is_moe_model:
            target_modules = [tm for tm in target_modules if tm not in {'gate'}]
        if not target_modules:
            continue
        target_modules = [tm for tm in target_modules if tm]
        target_pattern = rf'.*\.({"|".join(target_modules)})' if target_modules else ''
        rejected_pattern = rf'(?!({"|".join(rejected_modules)}))' if rejected_modules else ''
        
        # NOTE: 
        if 'model.visual_low_level' in module or 'model.language_model' in module:
            res.append(rf'{rejected_pattern}{module}{target_pattern}')
        
        # NOTE:
        target_modules_dict[module] = rf'{rejected_pattern}{module}{target_pattern}'

    return rf'^({"|".join(res)})$', target_modules_dict


class CustomTuner(Tuner):
    """Full-parameter training of ViT while LoRA training LLM"""

    @staticmethod
    def from_pretrained(model: torch.nn.Module, model_id: str, **kwargs) -> torch.nn.Module:
        model = Swift.from_pretrained(model, model_id, **kwargs)
        # state_dict = safetensors.torch.load_file(os.path.join(model_id, 'vit_low_level.safetensors'))
        # model.load_state_dict(state_dict, strict=False)
        return model

    @staticmethod
    def save_pretrained(
        model: torch.nn.Module,
        save_directory: str,
        state_dict: Optional[dict] = None,
        safe_serialization: bool = True,
        **kwargs,
    ) -> None:
        if state_dict is None:
            state_dict = {}
            for n, p in model.named_parameters():
                if p.requires_grad:
                    state_dict[n] = p.detach().cpu()
        model.save_pretrained(save_directory, state_dict=state_dict, safe_serialization=safe_serialization, **kwargs)
        # # vit
        # model_arch = model.model_meta.model_arch
        # state_dict = {k: v for k, v in state_dict.items() if is_vit_param(model_arch, k)}
        # safetensors.torch.save_file(
        #     state_dict, os.path.join(save_directory, 'vit_low_level.safetensors'), metadata={'format': 'pt'})

    @staticmethod
    def prepare_model(args: 'TrainArguments', model: torch.nn.Module) -> torch.nn.Module:
        model_arch = model.model_meta.model_arch
        target_regex, module_dict = get_multimodal_target_regex(model, freeze_vit=False, freeze_llm=False, freeze_aligner=False)
        
        # lora_setting = {}
        # for module_name in module_dict.keys():
        #     if 'language_model' in module_name:
        #         lora_setting['llm'] = rf'^({"|".join(module_dict[module_name])})$'
        #     elif 'visual_low_level' in module_name:
        #         if 'vit' not in lora_setting.keys():
        #             lora_setting['vit'] = module_dict[module_name]
        #         else:
        #             lora_setting['vit'] += rf'|^{module_dict[module_name]})$'
        
        logger.info(f'target_regex: {target_regex}')
        
        all_layers_list = []
        import re
        regex = re.compile(target_regex)
        for name, param in model.named_modules():
            all_layers_list.append(name)

        target_modules = [name for name in all_layers_list if regex.match(name)]
        target_layer = 16       # This layer should be after the insertion layer; for example, start training from layer 25 when inserting at layer 24.
        lora_modules = []
        for item in target_modules:
            if 'model.language_model' in item:
                layer_num = int(item.split('model.language_model.layers.')[1].split('.')[0])
                if layer_num >= target_layer:
                    lora_modules.append(item)
            else:
                lora_modules.append(item)
                
        for item in lora_modules:
            logger.info(item)
                
        lora_config = LoraConfig(
            task_type='CAUSAL_LM', r=args.lora_rank, lora_alpha=args.lora_alpha, target_modules=lora_modules,
            )

        model = Swift.prepare_model(model, lora_config)

        # Just train ViT who extracts low-level features
        model.requires_grad_(False)
        # for module_prefix in model_arch.vision_tower + model_arch.aligner:
        #     if 'visual_low_level' in module_prefix:
        #         deep_getattr(model, module_prefix).requires_grad_(True)
        for name, param in model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
        
        # Print the trainable parameters
        total_params = 0
        trainable_params = 0
        for name, param in model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                logger.info(f'Trainable parameter: {name}, shape: {param.shape}')
        
        logger.info(f'Total parameters: {total_params}, Trainable parameters: {trainable_params}, '
                    f'Percentage: {100 * trainable_params / total_params:.4f}%')
        
        return model

extra_tuners['custom'] = CustomTuner
