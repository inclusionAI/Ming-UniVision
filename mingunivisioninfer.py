import os
import time
import torch
from transformers import AutoProcessor

from modeling_bailingmm import MingUniVisionForConditionalGeneration
from IPython import embed
import torchvision
from PIL import Image
import re

import torch.nn as nn
from collections import defaultdict
from bailingmm_utils import process_ratio
import torchvision.transforms as T
import warnings
import argparse
from transformers import (
    AutoProcessor,
    AutoTokenizer,
)

warnings.filterwarnings("ignore")


def tensor_to_pil(image_tensor):
    """将tensor转换为PIL图像"""
    mean = torch.Tensor([0.5,0.5,0.5]).view(1,-1,1,1).cuda()
    std = torch.Tensor([0.5,0.5,0.5]).view(1,-1,1,1).cuda()
    image_tensor = (image_tensor*std + mean)[0]
    image_tensor = T.ToPILImage()(image_tensor)
    return image_tensor


class MingUniVisionInfer:
    def __init__(self,
        model_name_or_path
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.model, self.tokenizer, self.processor = self.load_model_processor()
        self.model.tokenizer = self.tokenizer
        self.model.model.tokenizer = self.tokenizer

    def load_model_processor(self):
        tokenizer = AutoTokenizer.from_pretrained(".", trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(".", trust_remote_code=True)
        
        model = MingUniVisionForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        model = model.to("cuda", dtype=torch.bfloat16)

        return model, tokenizer, processor

    def generate(self, messages, max_new_tokens=512, output_image_prefix="output", for_edit=False):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, use_system=True
        )

        image_inputs, _, _ = self.processor.process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            return_tensors="pt",
            image_patch_size=self.model.vision.patch_size,
            for_edit=for_edit,
        ).to(self.model.device)

        for k in inputs.keys():
            if k == "pixel_values":
                inputs[k] = inputs[k].to(dtype=torch.bfloat16)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                output_image_prefix=output_image_prefix,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text
    
    def reset_inner_state(self):
        self.model.reset_inner_state()