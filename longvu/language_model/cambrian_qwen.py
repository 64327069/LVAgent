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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation.utils import GenerateOutput

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.utils import logging

from ..cambrian_arch import CambrianMetaForCausalLM, CambrianMetaModel

IS_XLA_AVAILABLE = False

from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2Model

logger = logging.get_logger(__name__)


class CambrianConfig(Qwen2Config):
    model_type = "cambrian_qwen"

    debug = "debug"


class CambrianQwenModel(CambrianMetaModel, Qwen2Model):
    config_class = CambrianConfig

    def __init__(self, config: Qwen2Config):
        super(CambrianQwenModel, self).__init__(config)

    def forward(
        self,
        # pyre-fixme[9]: input_ids has type `LongTensor`; used as `None`.
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        vision_tower_aux_feature_list: Optional[List[torch.FloatTensor]] = None,
        vision_tower_aux_attention_masks_list: Optional[List[torch.Tensor]] = None,
        final_vision_feature_size: Optional[List[tuple]] = None,
        global_context_feature: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            # pyre-fixme[16]: `CambrianQwenModel` has no attribute `config`.
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        # pyre-fixme[16]: `CambrianQwenModel` has no attribute `gradient_checkpointing`.
        # pyre-fixme[16]: `CambrianQwenModel` has no attribute `training`.
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        use_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            use_legacy_cache = True
            # pyre-fixme[6]: For 1st argument expected
            #  `Optional[Tuple[Tuple[FloatTensor]]]` but got
            #  `Optional[List[FloatTensor]]`.
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        if inputs_embeds is None:
            # pyre-fixme[16]: `CambrianQwenModel` has no attribute `embed_tokens`.
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = (
                # pyre-fixme[16]: Item `List` of `Union[List[torch._C.FloatTensor],
                #  DynamicCache]` has no attribute `get_seq_length`.
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # pyre-fixme[16]: `CambrianQwenModel` has no attribute `_update_causal_mask`.
        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        # pyre-fixme[16]: `CambrianQwenModel` has no attribute `layers`.
        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                # pyre-fixme[16]: `CambrianQwenModel` has no attribute
                #  `_gradient_checkpointing_func`.
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # pyre-fixme[16]: `CambrianQwenModel` has no attribute `norm`.
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if use_legacy_cache
                else next_decoder_cache
            )

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class CambrianQwenForCausalLM(Qwen2ForCausalLM, CambrianMetaForCausalLM):
    config_class = CambrianConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "cambrian_qwen"
        config.rope_scaling = None

        self.model = CambrianQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        # pyre-fixme[9]: input_ids has type `LongTensor`; used as `None`.
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_aux_attention_masks_list: Optional[List[torch.Tensor]] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        input_image_features = None
        highres_image_features = None
        frame_split_sizes = None

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                vision_tower_aux_feature_list,
                vision_tower_aux_attention_masks_list,
                final_vision_feature_size,
                global_context_feature,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_aux_attention_masks_list,
                image_sizes,
            )

        if dpo_forward:
            # pyre-fixme[29]: `CambrianQwenModel` is not a function.
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            if hasattr(self, "vision_tower_aux_feature_list"):
                # pyre-fixme[29]: `CambrianQwenModel` is not a function.
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    vision_tower_aux_feature_list=(
                        # pyre-fixme[61]: `vision_tower_aux_feature_list` is
                        #  undefined, or not always defined.
                        vision_tower_aux_feature_list
                        if inputs_embeds is None
                        # pyre-fixme[16]: `CambrianQwenForCausalLM` has no attribute
                        #  `vision_tower_aux_feature_list`.
                        else self.vision_tower_aux_feature_list
                    ),
                    vision_tower_aux_attention_masks_list=(
                        # pyre-fixme[61]: `vision_tower_aux_attention_masks_list` is
                        #  undefined, or not always defined.
                        vision_tower_aux_attention_masks_list
                        if inputs_embeds is None
                        # pyre-fixme[16]: `CambrianQwenForCausalLM` has no attribute
                        #  `vision_tower_aux_attention_masks_list`.
                        else self.vision_tower_aux_attention_masks_list
                    ),
                    final_vision_feature_size=(
                        # pyre-fixme[61]: `final_vision_feature_size` is undefined,
                        #  or not always defined.
                        final_vision_feature_size
                        if inputs_embeds is None
                        # pyre-fixme[16]: `CambrianQwenForCausalLM` has no attribute
                        #  `final_vision_feature_size`.
                        else self.final_vision_feature_size
                    ),
                    global_context_feature=(
                        # pyre-fixme[61]: `global_context_feature` is undefined, or
                        #  not always defined.
                        global_context_feature
                        if inputs_embeds is None
                        # pyre-fixme[16]: `CambrianQwenForCausalLM` has no attribute
                        #  `global_context_feature`.
                        else self.global_context_feature
                    ),
                )
            else:
                # pyre-fixme[29]: `CambrianQwenModel` is not a function.
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    # final_vision_feature_size=final_vision_feature_size,
                )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            logits = logits.float()

            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                # pyre-fixme[16]: `CambrianQwenForCausalLM` has no attribute `config`.
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                vision_tower_aux_feature_list,
                vision_tower_aux_attention_masks_list,
                final_vision_feature_size,
                global_context_feature,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
            )
            # pyre-fixme[16]: `CambrianQwenForCausalLM` has no attribute
            #  `vision_tower_aux_feature_list`.
            self.vision_tower_aux_feature_list = vision_tower_aux_feature_list
            # pyre-fixme[16]: `CambrianQwenForCausalLM` has no attribute
            #  `vision_tower_aux_attention_masks_list`.
            self.vision_tower_aux_attention_masks_list = (
                vision_tower_aux_attention_masks_list
            )
            # pyre-fixme[16]: `CambrianQwenForCausalLM` has no attribute
            #  `final_vision_feature_size`.
            self.final_vision_feature_size = final_vision_feature_size
            # pyre-fixme[16]: `CambrianQwenForCausalLM` has no attribute
            #  `global_context_feature`.
            self.global_context_feature = global_context_feature
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        # pyre-fixme[16]: `Qwen2ForCausalLM` has no attribute `generate`.
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("cambrian_qwen", CambrianConfig)
AutoModelForCausalLM.register(CambrianConfig, CambrianQwenForCausalLM)
