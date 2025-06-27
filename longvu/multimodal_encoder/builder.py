# pyre-unsafe
import copy

from .dino_encoder import DinoVisionTower
from .siglip_encoder import SiglipVisionTower


def build_vision_tower_aux_list(vision_tower_cfg, **kwargs):
    vision_tower_aux_name_list = getattr(
        vision_tower_cfg,
        "mm_vision_tower_aux_list",
        getattr(vision_tower_cfg, "vision_tower_aux_list", None),
    )
    vision_tower_aux_token_len_list = getattr(
        vision_tower_cfg,
        "mm_vision_tower_aux_token_len_list",
        getattr(vision_tower_cfg, "vision_tower_aux_token_len_list", None),
    )
    vision_tower_aux_list = []
    for vision_tower_aux_name, vision_tower_aux_token_len in zip(
        vision_tower_aux_name_list, vision_tower_aux_token_len_list
    ):
        config = copy.deepcopy(vision_tower_cfg)
        vision_tower_aux_name += "-interp{}".format(vision_tower_aux_token_len)
        if "siglip" in vision_tower_aux_name.lower():
            vision_tower_aux_list.append(
                SiglipVisionTower(vision_tower_aux_name, args=config, **kwargs)
            )

        # SSL-based Vision Towers
        elif "dinov2" in vision_tower_aux_name.lower():
            vision_tower_aux_list.append(
                DinoVisionTower(vision_tower_aux_name, args=config, **kwargs)
            )
        else:
            raise ValueError(f"Unknown vision tower: {vision_tower_aux_name}")
    return vision_tower_aux_list
