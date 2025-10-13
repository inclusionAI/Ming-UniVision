import json
import torch
from PIL import Image
import torchvision.transforms as T
from omegaconf import MISSING, OmegaConf, DictConfig, ListConfig
from mingtok.modeling_mingtok import MingTok
from mingtok.utils import CenterCropProcessor

if __name__ == "__main__":

    mingtok_model = MingTok.from_pretrained("inclusionAI/MingTok-Vision")
    mingtok_model = mingtok_model.cuda()

    img_path = "mingtok/asset/mingtok.png"
    save_path = "mingtok/asset/mingtok_recon.png"

    image = Image.open(img_path).convert("RGB")

    processor = CenterCropProcessor(image_size=512, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    image = processor(image).cuda().unsqueeze(0)

    out = mingtok_model.forward_enc_dec(image)

    output_mean = torch.Tensor([0.5,0.5,0.5]).view(1,-1,1,1).cuda()
    output_std = torch.Tensor([0.5,0.5,0.5]).view(1,-1,1,1).cuda()
    output_image = (out*output_std + output_mean)[0]
    output_image = T.ToPILImage()(output_image)
    output_image.save(save_path)