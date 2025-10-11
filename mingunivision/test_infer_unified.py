

import os
import torch
from mingunivisioninfer import MingUniVisionInfer
model = MingUniVisionInfer("inclusionAI/Ming-UniVision-16B-A3B")
torch.manual_seed(11)

# single round generation
image_gen_prompt = "Please generate the corresponding image based on the description. A cute girl."
messages = [{
  "role": "HUMAN",
  "content": [{"type": "text", "text": image_gen_prompt},],
}]
output_text = model.generate(messages, max_new_tokens=512, output_image_prefix="a_cute_girl")
model.reset_inner_state()

# single ground understanding
messages = [{
  "role": "HUMAN",
  "content": [
    {"type": "image", "image": "a_cute_girl.png"},
    {"type": "text", "text": "Please describe the picture in detail."},
  ],
}]
output_text = model.generate(messages, max_new_tokens=512)
print(output_text)
model.reset_inner_state()

# multi-round editing
messages = [{
  "role": "HUMAN",
  "content": [
    {"type": "image", "image": "a_cute_girl.png"},
    {"type": "text", "text": "Given the edit instruction: Change the color of her cloth to red, please identify the editing region"},
  ],
}]
output_text = model.generate(messages, max_new_tokens=512, for_edit=True, output_image_prefix="edit_round_0")

messages = [{
  "role": "HUMAN",
  "content": [
    {"type": "text", "text": "Change the color of her cloth to red."},
  ],
}]
output_text = model.generate(messages, max_new_tokens=512, for_edit=True, output_image_prefix="edit_round_1")

messages = [{
  "role": "HUMAN",
  "content": [
    {"type": "text", "text": "Refine the image for better clarity."},
  ],
}]
output_text = model.generate(messages, max_new_tokens=512, for_edit=True, output_image_prefix="edit_round_2")

model.reset_inner_state()

# single round text-based conversation
messages = [{
  "role": "HUMAN",
  "content": [
    {"type": "text", "text": "请详细介绍鹦鹉的习性。"},
  ],
}]

output_text = model.generate(messages, max_new_tokens=512)
print(output_text)
model.reset_inner_state()
