#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import math
import random
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from longvu.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)

from .multimodal_encoder.builder import build_vision_tower_aux_list
from .multimodal_projector.builder import build_vision_projector
from .vision_sampler import VisionTokenSampler

IS_XLA_AVAILABLE = False


class CambrianMetaModel:

    def __init__(self, config):
        super(CambrianMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower_aux_list"):

            projector_type = getattr(config, "mm_projector_type", "linear")
            if projector_type == "sva":

                vision_hidden_size = config.vision_hidden_size
                num_query_group = config.num_query_group
                query_num_list = config.query_num_list
                connector_only = config.connector_only
                connector_depth = config.connector_depth
                self.vision_tower_aux_list = build_vision_tower_aux_list(
                    config, delay_load=True
                )
                self.mm_projector = nn.Sequential(
                    nn.Linear(vision_hidden_size * num_query_group, config.hidden_size),
                    nn.GELU(),
                    nn.Linear(config.hidden_size, config.hidden_size),
                )

                image_token_len = config.image_token_len
                vision_tower_aux_token_len_list = (
                    self.config.mm_vision_tower_aux_token_len_list
                )
                cross_att_token_len_list = [
                    int(vision_tower_aux_token_len**0.5) // int(image_token_len**0.5)
                    for vision_tower_aux_token_len in vision_tower_aux_token_len_list
                ]

                for aux_i, vision_tower_aux in enumerate(self.vision_tower_aux_list):
                    setattr(
                        self,
                        "mm_projector_aux_{}".format(aux_i),
                        nn.Sequential(
                            nn.Linear(vision_tower_aux.hidden_size, vision_hidden_size),
                            nn.GELU(),
                            nn.Linear(vision_hidden_size, vision_hidden_size),
                            nn.LayerNorm(vision_hidden_size),
                        ),
                    )

                for query_group_i in range(num_query_group):
                    cross_att_token_len_list = [
                        int(vision_tower_aux_token_len**0.5)
                        // int(query_num_list[query_group_i] ** 0.5)
                        for vision_tower_aux_token_len in vision_tower_aux_token_len_list
                    ]
                    setattr(
                        self,
                        "vision_sampler_{}".format(query_group_i),
                        VisionTokenSampler(
                            vision_hidden_size,
                            vision_hidden_size,
                            [vision_hidden_size] * len(self.vision_tower_aux_list),
                            cross_att_token_len_list,
                            vision_hidden_size,
                            connector_depth,
                        ),
                    )

                if not connector_only:
                    num_of_vision_sampler_layers = (
                        config.num_of_vision_sampler_layers
                    ) = config.num_of_vision_sampler_layers
                    config.start_of_vision_sampler_layers = (
                        config.start_of_vision_sampler_layers
                    )
                    config.stride_of_vision_sampler_layers = (
                        config.stride_of_vision_sampler_layers
                    )
                    cross_att_token_len_list = [
                        int(vision_tower_aux_token_len**0.5)
                        // int(image_token_len**0.5)
                        for vision_tower_aux_token_len in vision_tower_aux_token_len_list
                    ]
                    self.vision_sampler_layers = nn.ModuleList(
                        [
                            VisionTokenSampler(
                                config.hidden_size,
                                vision_hidden_size,
                                [vision_hidden_size] * len(self.vision_tower_aux_list),
                                cross_att_token_len_list,
                                vision_hidden_size,
                                1,
                            )
                            for layer_idx in range(0, num_of_vision_sampler_layers)
                        ]
                    )

                self.vision_query = nn.Parameter(
                    torch.randn((num_query_group, vision_hidden_size), dtype=self.dtype)
                )

                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

                self.frame_pos = torch.stack(
                    [
                        1
                        / torch.pow(
                            torch.tensor(10000),
                            torch.tensor(2 * (hid_j // 2) / config.hidden_size),
                        )
                        for hid_j in range(config.hidden_size)
                    ]
                )

            else:
                self.vision_tower_aux_list = build_vision_tower_aux_list(
                    config, delay_load=True
                )
                config.mm_hidden_size = sum(
                    [
                        vision_tower_aux.hidden_size
                        for vision_tower_aux in self.vision_tower_aux_list
                    ]
                )
                self.mm_projector = build_vision_projector(config)
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_frame_pos(self, time_range):
        frame_pos = self.frame_pos.reshape(1, -1) * time_range.reshape(-1, 1).to(
            self.frame_pos.device
        )
        frame_pos[:, 0::2] = torch.sin(frame_pos[:, 0::2])
        frame_pos[:, 1::2] = torch.cos(frame_pos[:, 0::2])
        frame_pos = frame_pos.unsqueeze(1)
        return frame_pos

    # def get_vision_tower(self):
    #     vision_tower = getattr(self, 'vision_tower', None)
    #     if type(vision_tower) is list:
    #         vision_tower = vision_tower[0]
    #     return vision_tower

    def get_vision_tower_aux_list(self):
        vision_tower_aux_list = getattr(self, "vision_tower_aux_list", None)
        return vision_tower_aux_list

    def initialize_vision_modules(self, model_args, fsdp=None):
        # vision_tower = model_args.vision_tower
        num_query_group = model_args.num_query_group
        query_num_list = model_args.query_num_list
        vision_hidden_size = model_args.vision_hidden_size
        vision_tower_aux_list = model_args.vision_tower_aux_list
        vision_tower_aux_token_len_list = model_args.vision_tower_aux_token_len_list
        image_token_len = model_args.image_token_len
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        connector_only = model_args.connector_only
        connector_depth = model_args.connector_depth

        # self.config.mm_vision_tower = vision_tower
        self.config.image_token_len = image_token_len
        self.config.num_query_group = num_query_group
        self.config.query_num_list = query_num_list
        assert num_query_group == len(query_num_list)
        self.config.connector_depth = connector_depth
        self.config.mm_vision_tower_aux_list = vision_tower_aux_list
        self.config.mm_vision_tower_aux_token_len_list = vision_tower_aux_token_len_list
        self.config.connector_only = connector_only
        self.config.highres_connect = model_args.highres_connect
        self.config.highres = model_args.highres
        self.config.frame_pos = model_args.frame_pos
        self.config.lowres_token = model_args.lowres_token
        self.config.connect_layer = model_args.connect_layer
        self.config.dino_threshold = getattr(model_args, "dino_threshold", 0.83)
        self.config.drop_threshold = getattr(model_args, "drop_threshold", 0.6)
        self.config.is_image_newline = getattr(model_args, "is_image_newline", True)

        if self.get_vision_tower_aux_list() is None:
            vision_tower_aux_list = build_vision_tower_aux_list(model_args)
            if model_args.unfreeze_mm_vision_tower:
                self.vision_tower_aux_list = nn.ModuleList(vision_tower_aux_list)
            else:
                self.vision_tower_aux_list = vision_tower_aux_list
        else:
            vision_tower_aux_list = self.vision_tower_aux_list
            for vision_tower_aux in vision_tower_aux_list:
                vision_tower_aux.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(
            model_args, "mm_projector_type", "linear"
        )
        self.config.vision_hidden_size = vision_hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, "mm_projector", None) is None:

            if self.config.mm_projector_type == "sva":
                self.mm_projector = nn.Sequential(
                    nn.Linear(
                        vision_hidden_size * num_query_group, self.config.hidden_size
                    ),
                    nn.GELU(),
                    nn.Linear(self.config.hidden_size, self.config.hidden_size),
                )
                for aux_i, vision_tower_aux in enumerate(vision_tower_aux_list):
                    setattr(
                        self,
                        "mm_projector_aux_{}".format(aux_i),
                        nn.Sequential(
                            nn.Linear(vision_tower_aux.hidden_size, vision_hidden_size),
                            nn.GELU(),
                            nn.Linear(vision_hidden_size, vision_hidden_size),
                            nn.LayerNorm(vision_hidden_size),
                        ),
                    )

                # vision sampler for each group of query as the connector before the LLM
                for query_group_i in range(num_query_group):
                    cross_att_token_len_list = [
                        int(vision_tower_aux_token_len**0.5)
                        // int(query_num_list[query_group_i] ** 0.5)
                        for vision_tower_aux_token_len in vision_tower_aux_token_len_list
                    ]
                    setattr(
                        self,
                        "vision_sampler_{}".format(query_group_i),
                        VisionTokenSampler(
                            vision_hidden_size,
                            vision_hidden_size,
                            [vision_hidden_size] * len(vision_tower_aux_list),
                            cross_att_token_len_list,
                            vision_hidden_size,
                            connector_depth,
                        ),
                    )

                # sampler layers within LLM
                if not connector_only:
                    num_of_vision_sampler_layers = (
                        self.config.num_of_vision_sampler_layers
                    ) = model_args.num_of_vision_sampler_layers
                    self.config.start_of_vision_sampler_layers = (
                        model_args.start_of_vision_sampler_layers
                    )
                    self.config.stride_of_vision_sampler_layers = (
                        model_args.stride_of_vision_sampler_layers
                    )
                    cross_att_token_len_list = [
                        int(vision_tower_aux_token_len**0.5)
                        // int(image_token_len**0.5)
                        for vision_tower_aux_token_len in vision_tower_aux_token_len_list
                    ]
                    self.vision_sampler_layers = nn.ModuleList(
                        [
                            VisionTokenSampler(
                                self.config.hidden_size,
                                vision_hidden_size,
                                [vision_hidden_size] * len(vision_tower_aux_list),
                                cross_att_token_len_list,
                                vision_hidden_size,
                                1,
                            )
                            for layer_idx in range(0, num_of_vision_sampler_layers)
                        ]
                    )
                vision_embed_std = 1 / torch.sqrt(
                    torch.tensor(vision_hidden_size, dtype=self.dtype)
                )
                self.vision_query = nn.Parameter(
                    torch.randn((num_query_group, vision_hidden_size), dtype=self.dtype)
                    * vision_embed_std
                )

                embed_std = 1 / torch.sqrt(
                    torch.tensor(self.config.hidden_size, dtype=self.dtype)
                )
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )

            else:
                self.config.mm_hidden_size = sum(
                    [
                        vision_tower_aux.hidden_size
                        for vision_tower_aux in vision_tower_aux_list
                    ]
                )
                self.mm_projector = build_vision_projector(self.config)
                embed_std = 1 / torch.sqrt(
                    torch.tensor(self.config.hidden_size, dtype=self.dtype)
                )
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(
                pretrain_mm_mlp_adapter, map_location="cpu"
            )

            def get_w(weights, keyword):
                return {
                    k.split(keyword + ".")[1]: v
                    for k, v in weights.items()
                    if keyword + "." in k
                }

            self.mm_projector.load_state_dict(
                get_w(mm_projector_weights, "mm_projector"), strict=True
            )

            if self.config.mm_projector_type == "sva":
                for aux_i in range(len(vision_tower_aux_list)):
                    getattr(self, "mm_projector_aux_{}".format(aux_i)).load_state_dict(
                        get_w(
                            mm_projector_weights, "mm_projector_aux_{}".format(aux_i)
                        ),
                        strict=True,
                    )

                for query_group_i in range(num_query_group):
                    getattr(
                        self, "vision_sampler_{}".format(query_group_i)
                    ).load_state_dict(
                        get_w(
                            mm_projector_weights,
                            "vision_sampler_{}".format(query_group_i),
                        ),
                        strict=True,
                    )

                if not connector_only:
                    self.vision_sampler_layers.load_state_dict(
                        get_w(mm_projector_weights, "vision_sampler_layers"),
                        strict=True,
                    )
                self.vision_query.data = mm_projector_weights["model.vision_query"]
            self.image_newline.data = mm_projector_weights["model.image_newline"]


def unmask_attention_mask(mask, original_size):
    original_w, original_h = original_size
    cur_h, cur_w = mask.shape[1:3]

    original_aspect_ratio = original_w / original_h
    current_aspect_ratio = cur_w / cur_h

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = cur_w / original_w
        new_height = int(original_h * scale_factor)
        padding = (cur_h - new_height) // 2
        if padding > 0:
            mask[:, :padding, :] = 0
            mask[:, -padding:, :] = 0
        return mask
    else:
        scale_factor = cur_h / original_h
        new_width = int(original_w * scale_factor)
        padding = (cur_w - new_width) // 2
        if padding > 0:
            mask[:, :, :padding] = 0
            mask[:, :, -padding:] = 0
        return mask


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:3]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
        # if 0 in unpadded_tensor.shape:
        #     print(f"scale_factor: {scale_factor}, new_height: {new_height}, padding: {padding}, original_width: {original_width}, original_height: {original_height}")
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]
        # if 0 in unpadded_tensor.shape:
        #     print(f"scale_factor: {scale_factor}, new_width: {new_width}, padding: {padding}, original_width: {original_width}, original_height: {original_height}")

    return unpadded_tensor


class CambrianMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    # def get_vision_tower(self):
    #     return self.get_model().get_vision_tower()

    def get_vision_tower_aux_list(self):
        return self.get_model().get_vision_tower_aux_list()

    def rearrange_vision_tower_features_train(
        self,
        vision_tower_aux_feature_list,
        vision_tower_aux_attention_masks_list,
        query_side_len,
    ):
        vision_tower_aux_feature_rearranged_list = []
        vision_tower_aux_attention_masks_rearranged_list = []
        bs = vision_tower_aux_feature_list[0].shape[0]
        for vision_tower_aux_feature, vision_tower_aux_attention_masks in zip(
            vision_tower_aux_feature_list, vision_tower_aux_attention_masks_list
        ):
            aux_height = aux_width = int(vision_tower_aux_feature.shape[1] ** 0.5)
            assert (aux_height // query_side_len) * query_side_len == aux_height

            reduce_factor = aux_height // query_side_len
            vision_tower_aux_feature_rearranged = vision_tower_aux_feature.view(
                bs, query_side_len, reduce_factor, query_side_len, reduce_factor, -1
            )
            vision_tower_aux_feature_rearranged = (
                vision_tower_aux_feature_rearranged.permute(0, 1, 3, 2, 4, 5)
                .contiguous()
                .flatten(0, 2)
                .flatten(1, 2)
            )

            vision_tower_aux_attention_masks_rearranged = (
                vision_tower_aux_attention_masks.view(
                    bs * query_side_len * query_side_len, reduce_factor * reduce_factor
                )
            )

            vision_tower_aux_feature_rearranged_list.append(
                vision_tower_aux_feature_rearranged
            )
            vision_tower_aux_attention_masks_rearranged_list.append(
                vision_tower_aux_attention_masks_rearranged
            )
        return (
            vision_tower_aux_feature_rearranged_list,
            vision_tower_aux_attention_masks_rearranged_list,
        )

    def rearrange_vision_tower_features_inference(
        self, vision_tower_aux_feature_list, query_side_len, image_sizes, unpad=False
    ):
        vision_tower_aux_feature_rearranged_list = []
        vision_tower_aux_attention_masks_rearranged_list = []
        bs = vision_tower_aux_feature_list[0].shape[0]
        for vision_tower_aux_feature in vision_tower_aux_feature_list:
            aux_height = aux_width = int(vision_tower_aux_feature.shape[1] ** 0.5)
            assert (aux_height // query_side_len) * query_side_len == aux_height

            reduce_factor = aux_height // query_side_len

            vision_tower_aux_feature_rearranged = []
            vision_tower_aux_attention_masks_rearranged = []
            for batch_i in range(bs):
                image_size = image_sizes[batch_i]
                cur_vision_tower_aux_feature = vision_tower_aux_feature[batch_i]

                cur_vision_tower_aux_attention_masks_rearranged = torch.ones(
                    (1, aux_height, aux_width),
                    dtype=torch.bool,
                    device=cur_vision_tower_aux_feature.device,
                )
                cur_vision_tower_aux_feature_rearranged = (
                    cur_vision_tower_aux_feature.view(
                        1,
                        query_side_len,
                        reduce_factor,
                        query_side_len,
                        reduce_factor,
                        -1,
                    )
                )
                cur_vision_tower_aux_feature_rearranged = (
                    cur_vision_tower_aux_feature_rearranged.permute(
                        0, 1, 3, 2, 4, 5
                    ).contiguous()
                )
                if unpad:
                    cur_vision_tower_aux_feature_rearranged = unpad_image(
                        cur_vision_tower_aux_feature_rearranged, image_size
                    )
                cur_vision_tower_aux_feature_rearranged = (
                    cur_vision_tower_aux_feature_rearranged.flatten(0, 2).flatten(1, 2)
                )  # query_side_len*query_side_len X reduce_factor*reduce_factor X C

                cur_vision_tower_aux_attention_masks_rearranged = unmask_attention_mask(
                    cur_vision_tower_aux_attention_masks_rearranged, image_size
                )
                cur_vision_tower_aux_attention_masks_rearranged = (
                    cur_vision_tower_aux_attention_masks_rearranged.view(
                        1, query_side_len, reduce_factor, query_side_len, reduce_factor
                    )
                    .permute(0, 1, 3, 2, 4)
                    .contiguous()
                )
                if unpad:
                    cur_vision_tower_aux_attention_masks_rearranged = unpad_image(
                        cur_vision_tower_aux_attention_masks_rearranged, image_size
                    )
                cur_vision_tower_aux_attention_masks_rearranged = (
                    cur_vision_tower_aux_attention_masks_rearranged.flatten(
                        0, 2
                    ).flatten(1, 2)
                )

                cur_vision_tower_aux_attention_masks_rearranged[
                    cur_vision_tower_aux_attention_masks_rearranged.sum(-1) == 0
                ] = True

                vision_tower_aux_feature_rearranged.append(
                    cur_vision_tower_aux_feature_rearranged
                )
                vision_tower_aux_attention_masks_rearranged.append(
                    cur_vision_tower_aux_attention_masks_rearranged
                )

            vision_tower_aux_feature_rearranged = torch.cat(
                vision_tower_aux_feature_rearranged, 0
            )
            vision_tower_aux_attention_masks_rearranged = torch.cat(
                vision_tower_aux_attention_masks_rearranged, 0
            )

            vision_tower_aux_feature_rearranged_list.append(
                vision_tower_aux_feature_rearranged
            )
            vision_tower_aux_attention_masks_rearranged_list.append(
                vision_tower_aux_attention_masks_rearranged
            )

        return (
            vision_tower_aux_feature_rearranged_list,
            vision_tower_aux_attention_masks_rearranged_list,
        )

    def encode_images(self, image_aux_list, encode_type=None):
        vision_tower_aux_list = self.get_model().get_vision_tower_aux_list()
        image_aux_features_list = []
        chunk_size = 64
        if encode_type == "dino":
            image_aux = image_aux_list[-1]
            vision_tower_aux = vision_tower_aux_list[-1]
            if image_aux.shape[0] > chunk_size:
                image_aux_features_chunks = []
                for start_idx in range(0, image_aux.shape[0], chunk_size):
                    end_idx = min(start_idx + chunk_size, image_aux.shape[0])
                    chunk = image_aux[start_idx:end_idx]
                    image_aux_features_chunk = vision_tower_aux(chunk)
                    image_aux_features_chunks.append(image_aux_features_chunk)
                image_aux_features = torch.cat(image_aux_features_chunks, dim=0)
            else:
                image_aux_features = vision_tower_aux(image_aux)
            return image_aux_features
        elif encode_type == "siglip":
            image_aux = image_aux_list[0]
            vision_tower_aux = vision_tower_aux_list[0]
            if image_aux.shape[0] > chunk_size:
                image_aux_features_chunks = []
                for start_idx in range(0, image_aux.shape[0], chunk_size):
                    end_idx = min(start_idx + chunk_size, image_aux.shape[0])
                    chunk = image_aux[start_idx:end_idx]
                    image_aux_features_chunk = vision_tower_aux(chunk)
                    image_aux_features_chunks.append(image_aux_features_chunk)
                image_aux_features = torch.cat(image_aux_features_chunks, dim=0)
            else:
                image_aux_features = vision_tower_aux(image_aux)
            return image_aux_features
        else:
            for image_aux, vision_tower_aux in zip(
                image_aux_list, vision_tower_aux_list
            ):
                if image_aux.shape[0] > chunk_size:
                    image_aux_features_chunks = []
                    for start_idx in range(0, image_aux.shape[0], chunk_size):
                        end_idx = min(start_idx + chunk_size, image_aux.shape[0])
                        chunk = image_aux[start_idx:end_idx]
                        image_aux_features_chunk = vision_tower_aux(chunk)
                        image_aux_features_chunks.append(image_aux_features_chunk)
                    image_aux_features = torch.cat(image_aux_features_chunks, dim=0)
                else:
                    image_aux_features = vision_tower_aux(image_aux)
                image_aux_features_list.append(image_aux_features)
            return image_aux_features_list

    def select_frame(
        self,
        feature_list,
        split_sizes,
        input_ids,
        new_image_aux_list,
        image_sizes,
        window_size=16,
        threshold=0.83,
    ):
        dino_features_batch = torch.split(feature_list, split_sizes, dim=0)
        new_image_aux_batch_0 = torch.split(new_image_aux_list[0], split_sizes, dim=0)
        new_image_aux_batch_1 = torch.split(new_image_aux_list[1], split_sizes, dim=0)
        new_split_sizes = []
        selected_frames_all_0 = []
        selected_frames_all_1 = []
        selected_frames_feature_all = []
        selected_frame_indices_all = []
        for i_batch, frame_features in enumerate(dino_features_batch):
            try:
                if "llama" in self.get_model().config.model_type:
                    text_len = torch.where(input_ids[i_batch] == 128002)[-1][0]
                else:
                    text_len = torch.where(input_ids[i_batch] == 151643)[-1][0]
            except:
                text_len = len(input_ids[i_batch])
            original_width, original_height = image_sizes[i_batch]
            if getattr(self.get_model().config, "highres", False):
                token_per_frame = self.get_model().config.lowres_token ** 2
            else:
                token_per_frame = self.get_model().config.image_token_len
            # current_height, current_width = token_per_side, token_per_side
            # original_aspect_ratio = original_width / original_height
            # current_aspect_ratio = current_width / current_height
            # if original_aspect_ratio > current_aspect_ratio:
            #     scale_factor = current_width / original_width
            #     new_height = int(original_height * scale_factor)
            #     padding = math.ceil((current_height - new_height) / 2.0)
            #     token_per_frame = (
            #         current_height - padding * 2
            #     ) * token_per_side + token_per_side
            # else:
            #     scale_factor = current_height / original_height
            #     new_width = int(original_width * scale_factor)
            #     padding = math.ceil((current_width - new_width) / 2.0)
            #     token_per_frame = (current_width - padding * 2) * token_per_side + (
            #         current_width - padding * 2
            #     )
            # token_per_frame = (
            #     token_per_side**2 if token_per_frame < 1 else token_per_frame
            # )
            max_num_frames = max(
                1,
                (
                    self.get_model().config.tokenizer_model_max_length
                    - text_len
                    - getattr(self.get_model().config, "inference_max_length", 16)
                )
                // token_per_frame,
            )
            if len(frame_features) < max_num_frames:
                selected_frames_all_0.append(new_image_aux_batch_0[i_batch])
                selected_frames_all_1.append(new_image_aux_batch_1[i_batch])
                selected_frames_feature_all.append(frame_features)
                new_split_sizes.append(len(frame_features))
                selected_frame_indices_all.append(torch.arange(len(frame_features)))
                continue

            num_segments = len(frame_features) // window_size
            if num_segments == 0:
                query_feature = frame_features.flatten(1, 2)
                query_feature = query_feature / torch.norm(
                    (query_feature), dim=1, keepdim=True
                )
                similarities = torch.mean(query_feature @ query_feature.T, dim=1)
                similarities[len(frame_features) // 2] = 0
                indices = torch.where(similarities < threshold)[0]
                selected_frame_indices_all.append(indices)
                selected_frames_all_0.append(new_image_aux_batch_0[i_batch][indices])
                selected_frames_all_1.append(new_image_aux_batch_1[i_batch][indices])
                selected_frames_feature_all.append(frame_features[indices])
                new_split_sizes.append(len(indices))
                continue
            segments_frames_0 = []
            segments_frames_1 = []
            segments_features = []
            for start_idx in range(0, len(frame_features), window_size):
                end_idx = min(start_idx + window_size, len(frame_features))
                segments_frames_0.append(
                    new_image_aux_batch_0[i_batch][start_idx:end_idx]
                )
                segments_frames_1.append(
                    new_image_aux_batch_1[i_batch][start_idx:end_idx]
                )
                segments_features.append(frame_features[start_idx:end_idx])
            selected_frames_0 = []
            selected_frames_1 = []
            selected_features = []
            selected_frame_indices = []
            for i, segment in enumerate(segments_features):
                query_feature = segment.flatten(1, 2)
                query_feature = query_feature / torch.norm(
                    (query_feature), dim=1, keepdim=True
                )
                similarities = torch.mean(query_feature @ query_feature.T, dim=1)
                similarities[len(segment) // 2] = 0
                indices = torch.where(similarities < threshold)[0]
                selected_frames_0.append(segments_frames_0[i][indices])
                selected_frames_1.append(segments_frames_1[i][indices])
                selected_features.append(segment[indices])
                selected_frame_indices.extend(indices + i * window_size)
            selected_frames_0 = torch.cat(selected_frames_0, dim=0)
            selected_frames_1 = torch.cat(selected_frames_1, dim=0)
            selected_features = torch.cat(selected_features, dim=0)
            selected_frame_indices = torch.tensor(selected_frame_indices)
            # ablation
            max_num_frames = 400  # in case of OOM
            if len(selected_frames_0) > max_num_frames:
                interval = len(selected_frames_0) / float(max_num_frames)
                indices = [int(interval * i) for i in range(max_num_frames)]
                new_split_sizes.append(len(indices))
                selected_frames_all_0.append(selected_frames_0[indices])
                selected_frames_all_1.append(selected_frames_1[indices])
                selected_frames_feature_all.append(selected_features[indices])
                selected_frame_indices = selected_frame_indices[indices]
            else:
                new_split_sizes.append(len(selected_frames_0))
                selected_frames_all_0.append(selected_frames_0)
                selected_frames_all_1.append(selected_frames_1)
                selected_frames_feature_all.append(selected_features)
            selected_frame_indices_all.append(selected_frame_indices)
        selected_frames_all_0 = torch.cat(selected_frames_all_0, dim=0)
        selected_frames_all_1 = torch.cat(selected_frames_all_1, dim=0)
        selected_frames_feature_all = torch.cat(selected_frames_feature_all, dim=0)
        return (
            selected_frames_feature_all,
            new_split_sizes,
            [selected_frames_all_0, selected_frames_all_1],
            selected_frame_indices_all,
        )

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        image_aux_attention_masks_list=None,
        image_sizes=None,
    ):
        # vision_tower = self.get_vision_tower()
        vision_tower_aux_list = self.get_model().get_vision_tower_aux_list()
        if vision_tower_aux_list is None or images is None or input_ids.shape[1] == 1:
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
                None,
                None,
                None,
                None,
            )

        image_aux_list = images

        split_sizes = None

        if type(image_aux_list[0]) is list or image_aux_list[0].ndim == 5:
            split_sizes_ori = [
                1 if image.ndim == 3 else image.shape[0] for image in image_aux_list[0]
            ]
            new_image_aux_list = []
            for image_aux in image_aux_list:
                if type(image_aux) is list:
                    image_aux = [
                        x.unsqueeze(0) if x.ndim == 3 else x for x in image_aux
                    ]
                concat_image_aux = torch.cat([image for image in image_aux], dim=0)
                new_image_aux_list.append(concat_image_aux)
            image_aux_features_dino = self.encode_images(
                new_image_aux_list, encode_type="dino"
            )

            (
                image_aux_features_dino,
                split_sizes,
                new_image_aux_list,
                selected_frame_indices_all,
            ) = self.select_frame(
                image_aux_features_dino,
                split_sizes_ori,
                input_ids,
                new_image_aux_list,
                image_sizes,
                threshold=getattr(self.get_model().config, "dino_threshold", 0.83),
            )

            image_aux_features_siglip = self.encode_images(
                new_image_aux_list, encode_type="siglip"
            )
            image_aux_features_list = [
                image_aux_features_siglip,
                image_aux_features_dino,
            ]

            bs = image_aux_features_list[0].shape[0]
            dtype = new_image_aux_list[0].dtype

            frame_sizes = []
            for i in range(len(image_sizes)):
                for j in range(split_sizes[i]):
                    frame_sizes.append(image_sizes[i])
            image_sizes = frame_sizes
        else:
            image_aux_features_list = self.encode_images(image_aux_list)
            bs = image_aux_list[0].shape[0]
            dtype = image_aux_list[0].dtype

        image_token_len = self.get_model().config.image_token_len
        query_num_list = self.get_model().config.query_num_list

        final_height = final_width = int(image_token_len**0.5)

        final_image_features_list = []
        final_image_features_down_list = []

        # only needed for sva
        vision_tower_aux_feature_list_final = None
        vision_tower_aux_attention_masks_list_final = None
        global_context_feature_final = None

        if self.get_model().config.mm_projector_type == "sva":
            vision_tower_aux_feature_list = []
            vision_tower_aux_attention_masks_list = []
            # get vision tokens from each vision tower
            for aux_i in range(len(vision_tower_aux_list)):
                image_aux_features = image_aux_features_list[aux_i]

                image_aux_features = getattr(
                    self.get_model(), "mm_projector_aux_{}".format(aux_i)
                )(image_aux_features).to(dtype)
                if aux_i == 0:
                    global_context_feature = image_aux_features.mean(1).view(
                        bs, 1, 1, -1
                    )

                vision_tower_aux_feature_list.append(image_aux_features)
            input_mix_res = True
            input_high_res = True
            # perform vision sampling for each query group
            for query_group_i, query_num in enumerate(query_num_list):
                query_features_i = (
                    self.get_model()
                    .vision_query[query_group_i, :]
                    .view(1, 1, 1, -1)
                    .expand(bs, query_num, -1, -1)
                )
                global_context_feature_i = global_context_feature.expand(
                    -1, query_num, 1, -1
                ).flatten(0, 1)
                query_side_len = int(query_num**0.5)
                if IS_XLA_AVAILABLE:
                    (
                        vision_tower_aux_feature_list_i,
                        vision_tower_aux_attention_masks_list_i,
                    ) = self.rearrange_vision_tower_features_train(
                        vision_tower_aux_feature_list,
                        image_aux_attention_masks_list,
                        query_side_len,
                    )
                else:
                    (
                        vision_tower_aux_feature_list_i,
                        vision_tower_aux_attention_masks_list_i,
                    ) = self.rearrange_vision_tower_features_inference(
                        vision_tower_aux_feature_list, query_side_len, image_sizes
                    )

                query_features_i = getattr(
                    self.get_model(), "vision_sampler_{}".format(query_group_i)
                )(
                    query_features_i.flatten(0, 1),
                    global_context_feature_i,
                    *vision_tower_aux_feature_list_i,
                    *vision_tower_aux_attention_masks_list_i,
                )
                query_features_i = query_features_i.view(bs, query_num, -1)

                if split_sizes is not None:
                    try:
                        if "llama" in self.get_model().config.model_type:
                            text_len = torch.where(input_ids[0] == 128002)[-1][0]
                        else:
                            text_len = torch.where(input_ids[0] == 151643)[-1][0]
                    except:
                        text_len = len(input_ids[0])
                    max_visual_len = (
                        self.get_model().config.tokenizer_model_max_length
                        - text_len
                        - getattr(self.get_model().config, "inference_max_length", 16)
                    )
                    max_num_frames = max(
                        1,
                        math.floor(max_visual_len // (final_height * final_width)),
                    )
                    max_num_frames_low = max(
                        1,
                        math.floor(
                            max_visual_len
                            // (self.get_model().config.lowres_token ** 2)
                        ),
                    )
                    if split_sizes[0] < max_num_frames:
                        input_mix_res = False
                    elif split_sizes[0] > max_num_frames_low:
                        input_mix_res = False
                        input_high_res = False

                # input_mix_res = False  # ablation

                if (getattr(self.config, "highres", False)) and input_mix_res:
                    _query_features_i = (
                        query_features_i.permute(0, 2, 1)
                        .contiguous()
                        .view(bs, -1, query_side_len, query_side_len)
                    )
                    _query_features_i = F.interpolate(
                        _query_features_i.float(),
                        size=(
                            self.get_model().config.lowres_token,
                            self.get_model().config.lowres_token,
                        ),
                        mode="bilinear",
                        align_corners=False,
                    ).to(dtype=query_features_i.dtype)
                    _query_features_i = (
                        _query_features_i.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
                    )
                    final_image_features_down_list.append(_query_features_i)

                # interpolate to the final target size
                if query_side_len != final_height:
                    query_features_i = (
                        query_features_i.permute(0, 2, 1)
                        .contiguous()
                        .view(bs, -1, query_side_len, query_side_len)
                    )
                    if input_high_res:
                        query_features_i = F.interpolate(
                            query_features_i.float(),
                            size=(final_height, final_width),
                            mode="bilinear",
                            align_corners=False,
                        ).to(dtype=query_features_i.dtype)
                    else:
                        query_features_i = F.interpolate(
                            query_features_i.float(),
                            size=(8, 8),
                            mode="bilinear",
                            align_corners=False,
                        ).to(dtype=query_features_i.dtype)
                    query_features_i = (
                        query_features_i.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
                    )
                final_image_features_list.append(query_features_i)

            if IS_XLA_AVAILABLE:
                (
                    vision_tower_aux_feature_list_final,
                    vision_tower_aux_attention_masks_list_final,
                ) = self.rearrange_vision_tower_features_train(
                    vision_tower_aux_feature_list,
                    image_aux_attention_masks_list,
                    final_height,
                )
                global_context_feature_final = global_context_feature.expand(
                    -1, final_height * final_width, 1, -1
                ).flatten(0, 1)
        else:
            final_image_features_list = image_aux_features_list

        image_features = torch.cat(final_image_features_list, -1)
        image_features = self.get_model().mm_projector(image_features).to(dtype)

        if (getattr(self.config, "highres", False)) and input_mix_res:
            image_features_down = torch.cat(final_image_features_down_list, -1)
            image_features_down = (
                self.get_model().mm_projector(image_features_down).to(dtype)
            )

        if IS_XLA_AVAILABLE:
            image_features = image_features.view(
                image_features.shape[0], final_height, final_width, -1
            )
            image_features = torch.cat(
                (
                    image_features,
                    self.model.image_newline[None, None, None, :].expand(
                        image_features.shape[0], final_height, 1, -1
                    ),
                ),
                dim=2,
            )
            image_features = image_features.flatten(1, 2)
            final_size = [(final_height, final_width)] * bs

        else:
            image_features = image_features.view(bs, final_height, final_width, -1)
            if (getattr(self.config, "highres", False)) and input_mix_res:
                image_features_down = image_features_down.view(
                    bs,
                    self.get_model().config.lowres_token,
                    self.get_model().config.lowres_token,
                    -1,
                )
            image_features_unpadded = []
            image_features_downsample = []
            final_size = []
            if self.get_model().config.mm_projector_type == "sva":
                (
                    vision_tower_aux_feature_list_final,
                    vision_tower_aux_attention_masks_list_final,
                ) = self.rearrange_vision_tower_features_inference(
                    vision_tower_aux_feature_list, final_height, image_sizes, unpad=True
                )
                global_context_feature_final = []
            for batch_i in range(bs):
                cur_image_feature = image_features[batch_i]
                image_size = image_sizes[batch_i]

                cur_image_feature = unpad_image(
                    cur_image_feature.unsqueeze(0), image_size
                )

                cur_h, cur_w = cur_image_feature.shape[1:3]
                try:  # fix bug for some invalid image
                    cur_image_feature = cur_image_feature.view(1, cur_h, cur_w, -1)
                    final_size.append((cur_h, cur_w))
                except:
                    # print(f"invalid after unpad {image_features[batch_i].shape}, {image_sizes[batch_i]}", flush=True)
                    cur_image_feature = image_features[batch_i].unsqueeze(0)
                    image_size = image_sizes[batch_i]
                    cur_h, cur_w = cur_image_feature.shape[1:3]
                    cur_image_feature = cur_image_feature.view(1, cur_h, cur_w, -1)
                    final_size.append((cur_h, cur_w))

                if (getattr(self.config, "highres", False)) and input_mix_res:
                    cur_image_feature_down = unpad_image(
                        image_features_down[batch_i].unsqueeze(0),
                        (
                            int(
                                image_size[0]
                                / (
                                    image_token_len**0.5
                                    / self.get_model().config.lowres_token
                                )
                            ),
                            int(
                                image_size[1]
                                / (
                                    image_token_len**0.5
                                    / self.get_model().config.lowres_token
                                )
                            ),
                        ),
                    )
                    _cur_h, _cur_w = cur_image_feature_down.shape[1:3]

                    try:  # fix bug for some invalid image
                        cur_image_feature_down = cur_image_feature_down.view(
                            1, _cur_h, _cur_w, -1
                        )
                    except:
                        print("invalid after unpad", flush=True)
                        cur_image_feature_down = image_features_down[batch_i].unsqueeze(
                            0
                        )
                        _cur_h, _cur_w = cur_image_feature_down.shape[1:3]
                        cur_image_feature_down = cur_image_feature_down.view(
                            1, _cur_h, _cur_w, -1
                        )

                    cur_image_feature_down = torch.cat(
                        (
                            cur_image_feature_down,
                            self.model.image_newline.view(1, 1, 1, -1)
                            .expand(1, _cur_h, 1, -1)
                            .to(cur_image_feature_down.device),
                        ),
                        dim=2,
                    ).flatten(1, 2)

                    if split_sizes is None and getattr(self.config, "frame_pos", False):
                        frame_pos = (
                            self.get_model()
                            .get_frame_pos(torch.arange(1))
                            .to(cur_image_feature_down.device)
                            .to(cur_image_feature_down.dtype)
                        )
                        cur_image_feature_down += frame_pos

                    image_features_downsample.append(cur_image_feature_down.squeeze(0))

                cur_image_feature = torch.cat(
                    (
                        cur_image_feature,
                        self.model.image_newline.view(1, 1, 1, -1)
                        .expand(1, cur_h, 1, -1)
                        .to(cur_image_feature.device),
                    ),
                    dim=2,
                )

                if split_sizes is None and getattr(self.config, "frame_pos", False):
                    frame_pos = (
                        self.get_model()
                        .get_frame_pos(torch.arange(1))
                        .to(cur_image_feature.device)
                        .to(cur_image_feature.dtype)
                    )
                    cur_image_feature += frame_pos

                cur_image_feature = cur_image_feature.flatten(1, 2)
                image_features_unpadded.append(cur_image_feature.squeeze(0))

                if self.get_model().config.mm_projector_type == "sva":
                    cur_global_context_feature = global_context_feature[batch_i].expand(
                        cur_h * cur_w, 1, -1
                    )
                    global_context_feature_final.append(cur_global_context_feature)
            if self.get_model().config.mm_projector_type == "sva":
                global_context_feature_final = torch.cat(
                    global_context_feature_final, 0
                )

            if (getattr(self.config, "highres", False)) and input_mix_res:
                image_features = image_features_downsample
            else:
                image_features = image_features_unpadded

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
            self.config, "mm_use_im_start_end", False
        ):
            raise NotImplementedError

        split_image_features_unpadded = None
        frame_split_sizes = None

        if split_sizes is not None:
            split_image_features = []
            split_image_features_unpadded = (
                []
                if (getattr(self.config, "highres", False)) and input_mix_res
                else None
            )
            start_idx = 0
            for split_batch_idx, split_size in enumerate(split_sizes):
                if isinstance(image_features[start_idx : start_idx + split_size], list):
                    if getattr(self.config, "frame_pos", False):
                        frame_feature = torch.cat(
                            image_features[start_idx : start_idx + split_size], dim=0
                        ).reshape(split_size, -1, image_features[0].shape[-1])
                        frame_pos = (
                            self.get_model()
                            .get_frame_pos(selected_frame_indices_all[split_batch_idx])
                            .to(frame_feature.device)
                            .to(frame_feature.dtype)
                        )
                        frame_feature += frame_pos
                        split_image_features.append(
                            frame_feature.reshape(-1, image_features[0].shape[-1])
                        )
                    else:
                        split_image_features.append(
                            torch.cat(
                                image_features[start_idx : start_idx + split_size],
                                dim=0,
                            )
                        )
                    if (getattr(self.config, "highres", False)) and input_mix_res:
                        if getattr(self.config, "frame_pos", False):
                            frame_feature = torch.cat(
                                image_features_unpadded[
                                    start_idx : start_idx + split_size
                                ],
                                dim=0,
                            ).reshape(split_size, -1, image_features[0].shape[-1])
                            frame_pos = (
                                self.get_model()
                                .get_frame_pos(
                                    selected_frame_indices_all[split_batch_idx]
                                )
                                .to(frame_feature.device)
                                .to(frame_feature.dtype)
                            )
                            frame_feature += frame_pos
                            split_image_features_unpadded.append(
                                frame_feature.reshape(-1, image_features[0].shape[-1])
                            )
                        else:
                            split_image_features_unpadded.append(
                                torch.cat(
                                    image_features_unpadded[
                                        start_idx : start_idx + split_size
                                    ],
                                    dim=0,
                                )
                            )
                else:
                    if getattr(self.config, "frame_pos", False):
                        frame_feature = image_features[
                            start_idx : start_idx + split_size
                        ].reshape(split_size, -1, image_features[0].shape[-1])
                        frame_pos = (
                            self.get_model()
                            .get_frame_pos(selected_frame_indices_all[split_batch_idx])
                            .to(frame_feature.device)
                            .to(frame_feature.dtype)
                        )
                        frame_feature += frame_pos
                        split_image_features.append(
                            frame_feature.reshape(-1, image_features[0].shape[-1])
                        )
                    else:
                        split_image_features.append(
                            image_features[start_idx : start_idx + split_size]
                        )
                    if (getattr(self.config, "highres", False)) and input_mix_res:
                        if getattr(self.config, "frame_pos", False):
                            frame_feature = image_features_unpadded[
                                start_idx : start_idx + split_size
                            ]
                            frame_pos = (
                                self.get_model()
                                .get_frame_pos(
                                    selected_frame_indices_all[split_batch_idx]
                                )
                                .to(frame_feature.device)
                                .to(frame_feature.dtype)
                            )
                            frame_feature += frame_pos
                            split_image_features_unpadded.append(
                                frame_feature.reshape(-1, image_features[0].shape[-1])
                            )
                        else:
                            split_image_features_unpadded.append(
                                image_features_unpadded[
                                    start_idx : start_idx + split_size
                                ]
                            )
                start_idx += split_size
            image_features = split_image_features
            frame_split_sizes = split_sizes

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids

        attention_mask = attention_mask | (input_ids == IMAGE_TOKEN_INDEX)

        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        new_input_embeds = []
        new_labels = []
        image_token_indices_batch = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_image_features[0:0]], dim=0
                )
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = (
                [-1]
                + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
                + [cur_input_ids.shape[0]]
            )
            image_token_indices_batch.append(
                torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()[0]
            )
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(
                    cur_input_ids[
                        image_token_indices[i] + 1 : image_token_indices[i + 1]
                    ]
                )
                cur_labels_noim.append(
                    cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]]
                )
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(
                torch.cat(cur_input_ids_noim)
            )
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            text_len = sum([x.shape[0] for x in cur_input_embeds_no_im])
            visual_len = len(image_features[cur_image_idx])
            max_visual_len = (
                self.get_model().config.tokenizer_model_max_length
                - getattr(self.get_model().config, "inference_max_length", 16)
                - text_len
            )
            mix_token = False

            # ablation mix
            if (
                input_mix_res
                and (
                    self.get_model().config.image_token_len
                    > getattr(self.get_model().config, "lowres_token", 8) ** 2
                )
                and frame_split_sizes is not None
                and getattr(self.config, "highres", False)
            ):
                if max_visual_len > visual_len:
                    visual_emb = image_features[cur_image_idx]
                    text_emb = cur_input_embeds_no_im[-1]
                    highres_num = math.floor(
                        (max_visual_len - visual_len)
                        / (
                            split_image_features_unpadded[cur_image_idx].shape[0]
                            // frame_split_sizes[cur_image_idx]
                            - visual_emb.shape[0] // frame_split_sizes[cur_image_idx]
                        )
                    )
                    if highres_num >= 1:
                        mix_token = True
                        sim = torch.matmul(visual_emb, text_emb.transpose(0, 1)).mean(
                            dim=-1
                        )
                        sim_frame = sim.reshape(
                            frame_split_sizes[cur_image_idx], -1
                        ).mean(dim=-1)
                        highres_num = min(highres_num, sim_frame.shape[0])
                        top_values, top_indices = torch.topk(sim_frame, highres_num)
                        if len(top_indices) > 0:
                            sorted_indices = torch.sort(top_indices)[1]
                            top_indices = top_indices[sorted_indices]
                            visual_emb_frame = image_features[cur_image_idx].reshape(
                                frame_split_sizes[cur_image_idx],
                                -1,
                                image_features[cur_image_idx].shape[-1],
                            )
                            visual_emb_frame_highres = split_image_features_unpadded[
                                cur_image_idx
                            ].reshape(
                                frame_split_sizes[cur_image_idx],
                                -1,
                                split_image_features_unpadded[cur_image_idx].shape[-1],
                            )
                            current_point = 0
                            mix_visual_emb_frame = []
                            for frame_i in range(len(visual_emb_frame)):
                                if current_point > len(top_indices) - 1:
                                    mix_visual_emb_frame.append(
                                        visual_emb_frame[frame_i]
                                    )
                                    continue
                                if frame_i == top_indices[current_point]:
                                    mix_visual_emb_frame.append(
                                        visual_emb_frame_highres[frame_i]
                                    )
                                    current_point += 1
                                else:
                                    mix_visual_emb_frame.append(
                                        visual_emb_frame[frame_i]
                                    )
                            image_features[cur_image_idx] = torch.cat(
                                mix_visual_emb_frame, dim=0
                            )
            # ablation drop

            if (
                max_visual_len < visual_len
                and frame_split_sizes is not None
                and not mix_token
            ):
                visual_emb_frame = image_features[cur_image_idx].reshape(
                    frame_split_sizes[cur_image_idx],
                    -1,
                    image_features[cur_image_idx].shape[-1],
                )

                new_visual_emb_frames = []
                for start_idx in range(0, len(visual_emb_frame), 8):
                    end_idx = min(start_idx + 8, len(visual_emb_frame))
                    chunk_feature = visual_emb_frame[start_idx:end_idx]  # 8, HW, C
                    if len(chunk_feature) == 1:
                        new_visual_emb_frames.append(chunk_feature[0])
                        continue
                    sim = F.cosine_similarity(
                        chunk_feature[0]
                        .unsqueeze(0)
                        .repeat_interleave(len(chunk_feature[1:]), dim=0),
                        chunk_feature[1:],
                        dim=-1,
                    )
                    new_visual_emb_frame = torch.cat(
                        [
                            chunk_feature[0],
                            chunk_feature[1:].flatten(0, 1)[
                                sim.flatten(0, 1)
                                < getattr(
                                    self.get_model().config, "drop_threshold", 0.7
                                )
                            ],
                        ],
                        dim=0,
                    )
                    new_visual_emb_frames.append(new_visual_emb_frame)

                reduced_visual_len = sum([x.shape[0] for x in new_visual_emb_frames])

                if reduced_visual_len > max_visual_len:
                    force_remove = math.ceil(
                        (reduced_visual_len - max_visual_len)
                        / len(new_visual_emb_frames)
                    )
                    for chunk_i in range(len(new_visual_emb_frames)):
                        new_visual_emb_frames[chunk_i] = new_visual_emb_frames[chunk_i][
                            :-force_remove
                        ]
                    new_visual_emb_frames = torch.cat(new_visual_emb_frames, dim=0)
                else:
                    new_visual_emb_frames = torch.cat(new_visual_emb_frames, dim=0)

                image_features[cur_image_idx] = new_visual_emb_frames[:max_visual_len]

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(
            self.config, "tokenizer_model_max_length", None
        )
        if tokenizer_model_max_length is not None:
            new_input_embeds = [
                x[:tokenizer_model_max_length] for x in new_input_embeds
            ]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        position_ids = torch.zeros(
            (batch_size, max_len),
            dtype=position_ids.dtype,
            device=position_ids.device,
        )

        for i, (cur_new_embed, cur_new_labels) in enumerate(
            zip(new_input_embeds, new_labels)
        ):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0,
                        cur_len,
                        dtype=position_ids.dtype,
                        device=position_ids.device,
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0,
                        cur_len,
                        dtype=position_ids.dtype,
                        device=position_ids.device,
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
            vision_tower_aux_feature_list_final,
            vision_tower_aux_attention_masks_list_final,
            final_size,
            global_context_feature_final,
        )

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(
                    model_args.pretrain_mm_mlp_adapter, map_location="cpu"
                )
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[
                        -num_new_tokens:
                    ]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
