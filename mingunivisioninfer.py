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

    def generate(self, messages, max_new_tokens=512, image_gen=False, **image_gen_param):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, use_system=True
        )

        image_inputs, _, _ = self.processor.process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            return_tensors="pt",
            image_patch_size=self.model.vision.patch_size,
        ).to(self.model.device)

        for k in inputs.keys():
            if k == "pixel_values":
                inputs[k] = inputs[k].to(dtype=torch.bfloat16)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                image_gen=image_gen,
                **image_gen_param,
            )

        if image_gen:
            return generated_ids

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="inclusionAI/Ming-UniVision-16B-A3B")
    parser.add_argument('--max_new_tokens', type=int, default=512)
    args = parser.parse_args()

    model_name_or_path = args.model_name_or_path
    model = MingUniVisionInfer(model_name_or_path)

    image_gen_prompt = "A beautiful girl."
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "text", "text": image_gen_prompt},
            ],
        }
    ]

    srt_time = time.time()
    img_tensor = model.generate(messages, max_new_tokens=args.max_new_tokens, image_gen=True, image_gen_prompt=image_gen_prompt, image_gen_height=512, image_gen_width=512)
    print(f"Generate time: {(time.time() - srt_time):.2f}s")
    pil_img = tensor_to_pil(img_tensor)
    pil_img.save("a_beautiful_girl.jpg")

    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "image", "image": "a_beautiful_girl.jpg"},
                {"type": "text", "text": "Please describe the picture."},
            ],
        }
    ]

    srt_time = time.time()
    output_text = model.generate(messages, max_new_tokens=args.max_new_tokens)
    print(f"Generate time: {(time.time() - srt_time):.2f}s")
    print(output_text)

    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "text", "text": "请详细介绍鹦鹉的生活习性。"}
            ],
        }
    ]

    srt_time = time.time()
    output_text = model.generate(messages, max_new_tokens=args.max_new_tokens)
    print(f"Generate time: {(time.time() - srt_time):.2f}s")
    print(output_text)

    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "text", "text": "中国的首都是哪里？"},
            ],
        },
        {
            "role": "ASSISTANT",
            "content": [
                {"type": "text", "text": "北京"},
            ],
        },
        {
            "role": "HUMAN",
            "content": [
                {"type": "text", "text": "它的占地面积是多少？有多少常住人口？"},
            ],
        },
    ]

    srt_time = time.time()
    output_text = model.generate(messages, max_new_tokens=args.max_new_tokens)
    print(f"Generate time: {(time.time() - srt_time):.2f}s")
    print(output_text)
