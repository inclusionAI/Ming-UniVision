

from mingunivisioninfer import MingUniVisionInfer, tensor_to_pil
model = MingUniVisionInfer("inclusionAI/Ming-UniVision-16B-A3B")

image_gen_prompt = "A beautiful girl."
messages = [{
  "role": "HUMAN",
  "content": [{"type": "text", "text": image_gen_prompt},],
}]

img_tensor = model.generate(messages, max_new_tokens=512, image_gen=True, image_gen_prompt=image_gen_prompt, image_gen_height=512, image_gen_width=512)
pil_img = tensor_to_pil(img_tensor)
pil_img.save("a_beautiful_girl.jpg")

messages = [{
  "role": "HUMAN",
  "content": [
    {"type": "image", "image": "a_beautiful_girl.jpg"},
    {"type": "text", "text": "Please describe the picture."},
  ],
}]
output_text = model.generate(messages, max_new_tokens=512)
print(output_text)

messages = [{
  "role": "HUMAN",
  "content": [
    {"type": "text", "text": "请详细介绍鹦鹉的习性。"},
  ],
}]

output_text = model.generate(messages, max_new_tokens=512)
print(output_text)