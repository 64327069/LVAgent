import torch
import torch.nn.functional as F

from transformers import SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel

from .base_encoder import BaseVisionTower, ProcessorWrapper


class SiglipVisionTower(BaseVisionTower):
    def __init__(self, vision_tower_name, args, delay_load=False):
        super(SiglipVisionTower, self).__init__(vision_tower_name, args, delay_load)
        
        model_path = "google/siglip-so400m-patch14-384"
        base_model_name, res, interp = model_path, 384, 576
        self.vision_tower_name = base_model_name
        self._image_size = res if res is not None else 512
        self._interp_size = interp
        if not self.delay_load:
            self.load_model()
        elif self.unfreeze_mm_vision_tower:
            self.load_model()
        else:
            self._hidden_size = 1152

    def load_model(self, device_map=None):
        self.vision_model = "siglip"
        # clip_model, processor = create_model_from_pretrained(self.vision_tower_name)
        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name)

        # self.vision_tower = clip_model.visual.trunk
        self.vision_tower.output_tokens = True

        self._hidden_size = self.vision_tower.config.hidden_size
        self._image_size = self.vision_tower.config.image_size
        self._patch_size = self.vision_tower.config.patch_size
        self.image_processor = SiglipImageProcessor.from_pretrained(
            self.vision_tower_name
        )

        self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
        self.is_loaded = True

    def interpolate(self, image_features):
        if self._interp_size is None:
            return image_features

        b, num_tokens, dim = image_features.shape

        if num_tokens != self.num_patches:
            target_h = target_w = int(self._interp_size**0.5)
            h = w = int(num_tokens**0.5)

            image_features = image_features.view(b, h, w, dim)
            image_features = image_features.permute(0, 3, 1, 2).contiguous()

            image_features = F.interpolate(
                image_features.to(torch.float32),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).to(image_features.dtype)

            # Permute the dimensions back to (b, target_h, target_w, dim)
            image_features = image_features.permute(0, 2, 3, 1).contiguous()

            # Flatten the spatial dimensions (target_h, target_w) into a single dimension
            image_features = image_features.flatten(1, 2)

        return image_features

    def _forward(self, images, interpolate_token=576):
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            image_features = self.vision_tower.forward(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            ).hidden_states[-1]
            interp_features = self.interpolate(image_features)
            return interp_features
