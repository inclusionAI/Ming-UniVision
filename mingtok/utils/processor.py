
from omegaconf import OmegaConf
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode


class CenterCropProcessor():
    def __init__(
        self, image_size=512, mean=None, std=None
    ):
        if mean is None:
            mean = (0.5, 0.5, 0.5)
        if std is None:
            std = (0.5, 0.5, 0.5)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    size=image_size,
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 512)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)


        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
        )