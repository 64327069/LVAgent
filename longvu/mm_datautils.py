# pyre-strict
import copy
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import tokenizers

import torch

import transformers

from longvu import conversation as conversation_lib

from longvu.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)

# pyre-fixme[21]: Could not find module `decord`.
from decord import cpu, VideoReader  # @manual=fbsource//third-party/pypi/decord:decord

from packaging import version
from PIL import Image
from torch import distributed as dist
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    StateDictType,
)
from torch.utils.data import Dataset

# pyre-fixme
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse(
    "0.14"
)
from transformers import StoppingCriteria

from longvu.mm_utils import KeywordsStoppingCriteria


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def maybe_zero_3(param, ignore_status: bool = False, name=None):
    # NO deepspeed

    # from deepspeed import zero
    # from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    # if hasattr(param, "ds_id"):
    #     if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
    #         if not ignore_status:
    #             print(name, 'no ignore status')
    #     with zero.GatheredParameters([param]):
    #         param = param.data.detach().cpu().clone()
    # else:
    #     param = param.detach().cpu().clone()
    return param.detach().cpu().clone()


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {
        k: t
        for k, t in named_params
        if any(key_match in k for key_match in keys_to_match)
    }
    to_return = {
        k: maybe_zero_3(v, ignore_status=True, name=k).cpu()
        for k, v in to_return.items()
    }
    return to_return


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(
    trainer: transformers.Trainer, output_dir: str
) -> None:
    """Collects the state dict and dump to disk."""
    global_rank = dist.get_rank()
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    # pyre-fixme[16]: `Trainer` has no attribute `args`.
    if len(trainer.args.fsdp) == 0:
        # pyre-fixme[16]: `Trainer` has no attribute `model`.
        cpu_state_dict = trainer.model.state_dict()
    else:
        with FSDP.state_dict_type(
            trainer.model, StateDictType.FULL_STATE_DICT, save_policy
        ):
            cpu_state_dict = trainer.model.state_dict()

    for key in cpu_state_dict.keys():
        cpu_state_dict[key] = cpu_state_dict[key].to(torch.bfloat16)

    if global_rank == 0:
        trainer.model.config.save_pretrained(output_dir)
        current_folder = output_dir.split("/")[-1]
        parent_folder = os.path.dirname(output_dir)
        save_path = os.path.join(output_dir, "pytorch_model.bin")
        if getattr(trainer.args, "tune_mm_mlp_adapter", False) and not getattr(
            trainer.args, "tune_text_decoder", False
        ):
            # Only save Adapter
            keys_to_match = ["mm_projector"]
            if getattr(trainer.args, "use_im_start_end", False):
                keys_to_match.extend(["embed_tokens", "embed_in"])

            freeze_layer_remove = []
            for key in cpu_state_dict.keys():
                remove = True
                for key_match in keys_to_match:
                    if key_match in key:
                        remove = False
                        break
                if remove:
                    freeze_layer_remove.append(key)
            for key in freeze_layer_remove:
                del cpu_state_dict[key]

            if current_folder.startswith("checkpoint-"):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                save_path = os.path.join(mm_projector_folder, f"{current_folder}.bin")
            else:
                save_path = os.path.join(output_dir, f"mm_projector.bin")
        torch.save(cpu_state_dict, save_path)


def smart_tokenizer_and_embedding_resize(
    # pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
    #  `typing.Dict[<key type>, <value type>]` to avoid runtime subscripting errors.
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
) -> None:
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    # pyre-fixme[16]: `PreTrainedModel` has no attribute `resize_token_embeddings`.
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        # pyre-fixme[16]: `PreTrainedModel` has no attribute `get_input_embeddings`.
        input_embeddings = model.get_input_embeddings().weight.data
        # pyre-fixme[16]: `PreTrainedModel` has no attribute `get_output_embeddings`.
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(
    strings: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    # pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
    #  `typing.Dict[<key type>, <value type>]` to avoid runtime subscripting errors.
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


# pyre-fixme[2]: Parameter must be annotated.
def _mask_targets(target, tokenized_lens, speakers) -> None:
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def _add_speaker_and_signal(header, source, get_conversation: bool = True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = (
            BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        )
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def longvu_process_images(images, image_processor, model_cfg):
    if isinstance(image_processor, list):
        processor_aux_list = image_processor
        new_images_aux_list = []
        for image in images:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            image_aux_list = []
            for processor_aux in processor_aux_list:
                image_aux = image
                if hasattr(processor_aux, "image_mean"):
                    try:
                        target_resolution = processor_aux.crop_size["height"]
                    except:
                        target_resolution = processor_aux.size["height"]
                    image_aux = expand2square(
                        image_aux, tuple(int(x * 255) for x in processor_aux.image_mean)
                    ).resize((target_resolution, target_resolution))
                image_aux = processor_aux.preprocess(image_aux, return_tensors="pt")[
                    "pixel_values"
                ][0]
                image_aux_list.append(image_aux)
            new_images_aux_list.append(image_aux_list)
        new_images_aux_list = [
            list(batch_image_aux) for batch_image_aux in zip(*new_images_aux_list)
        ]
        new_images_aux_list = [
            torch.stack(image_aux).half().cuda() for image_aux in new_images_aux_list
        ]
        return new_images_aux_list
    else:
        image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
        new_images = []
        if image_aspect_ratio == "pad":
            for image in images:
                image = expand2square(
                    image, tuple(int(x * 255) for x in image_processor.image_mean)
                )
                image = image_processor.preprocess(image, return_tensors="pt")[
                    "pixel_values"
                ][0]
                new_images.append(image)
        else:
            return image_processor(images, return_tensors="pt")["pixel_values"]
        if all(x.shape == new_images[0].shape for x in new_images):
            new_images = torch.stack(new_images, dim=0)
        return new_images


# pyre-fixme[2]: Parameter must be annotated.
# pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
#  `typing.Dict[<key type>, <value type>]` to avoid runtime subscripting errors.
def preprocess_multimodal(sources: Sequence[str], data_args) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        # pyre-fixme[7]: Expected `Dict[typing.Any, typing.Any]` but got
        #  `Sequence[str]`.
        return sources

    for source in sources:
        for sentence in source:
            if (
                # pyre-fixme[6]: For 1st argument expected `Union[slice, SupportsIndex]`
                #  but got `str`.
                DEFAULT_IMAGE_TOKEN in sentence["value"]
                # pyre-fixme[6]: For 1st argument expected `Union[slice, SupportsIndex]`
                #  but got `str`.
                or "<video>" in sentence["value"]
            ):
                # pyre-fixme[16]: `str` has no attribute `__setitem__`.
                sentence["value"] = (
                    # pyre-fixme[6]: For 1st argument expected `Union[slice,
                    #  SupportsIndex]` but got `str`.
                    sentence["value"]
                    .replace(DEFAULT_IMAGE_TOKEN, "")
                    .replace("<video>", "")
                    .strip()
                )
                # pyre-fixme[6]: For 1st argument expected `Union[slice,
                #  SupportsIndex]` but got `str`.
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                # pyre-fixme[6]: For 1st argument expected `Union[slice,
                #  SupportsIndex]` but got `str`.
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    # pyre-fixme[6]: For 1st argument expected `Union[slice,
                    #  SupportsIndex]` but got `str`.
                    sentence["value"] = sentence["value"].replace(
                        DEFAULT_IMAGE_TOKEN,
                        "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>",
                    )
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = (
                    DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                )
            # pyre-fixme[6]: For 1st argument expected `Union[slice, SupportsIndex]`
            #  but got `str`.
            sentence["value"] = sentence["value"].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )

    # pyre-fixme[7]: Expected `Dict[typing.Any, typing.Any]` but got `Sequence[str]`.
    return sources


def preprocess_llama_2(
    # pyre-fixme[2]: Parameter must be annotated.
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    # pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
    #  `typing.Dict[<key type>, <value type>]` to avoid runtime subscripting errors.
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [
                longvu_tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(longvu_tokenizer_image_token(rou, tokenizer))
                instruction_len = len(longvu_tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    # pyre-fixme[2]: Parameter must be annotated.
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    # pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
    #  `typing.Dict[<key type>, <value type>]` to avoid runtime subscripting errors.
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [
                longvu_tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(longvu_tokenizer_image_token(rou, tokenizer))
                instruction_len = len(longvu_tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            # pyre-fixme
            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


# pyre-fixme[3]: Return type must be annotated.
def longvu_tokenizer_image_token(
    # pyre-fixme[2]: Parameter must be annotated.
    prompt,
    # pyre-fixme[2]: Parameter must be annotated.
    tokenizer,
    # pyre-fixme[2]: Parameter must be annotated.
    image_token_index=IMAGE_TOKEN_INDEX,
    # pyre-fixme[2]: Parameter must be annotated.
    return_tensors=None,
):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


# pyre-fixme[3]: Return type must be annotated.
def tokenizer_image_token_llama3(
    # pyre-fixme[2]: Parameter must be annotated.
    prompt,
    # pyre-fixme[2]: Parameter must be annotated.
    tokenizer,
    # pyre-fixme[2]: Parameter must be annotated.
    image_token_index=IMAGE_TOKEN_INDEX,
    # pyre-fixme[2]: Parameter must be annotated.
    return_tensors=None,
):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    for x in insert_separator(prompt_chunks, [image_token_index]):
        input_ids.extend(x)

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def preprocess_qwen(
    # pyre-fixme[2]: Parameter must be annotated.
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    system_message: str = "You are a helpful assistant.",
    # pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
    #  `typing.Dict[<key type>, <value type>]` to avoid runtime subscripting errors.
) -> Dict:
    # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)

    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    im_start, im_end = tokenizer.additional_special_tokens_ids
    # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
    unmask_tokens_idx = [198, im_start, im_end]
    nl_tokens = tokenizer("\n").input_ids

    # Reset Qwen chat templates so that it won't include system message every time we apply
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # _system = tokenizer("system").input_ids + nl_tokens
    # _user = tokenizer("user").input_ids + nl_tokens
    # _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess_llama3(
    # pyre-fixme[2]: Parameter must be annotated.
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    system_message: str = "You are a helpful assistant.",
    # pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
    #  `typing.Dict[<key type>, <value type>]` to avoid runtime subscripting errors.
) -> Dict:
    # roles = {"human": "<|start_header_id|>user<|end_header_id|>", "gpt": "<|start_header_id|>assistant<|end_header_id|>"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    unmask_tokens = [
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "\n\n",
    ]
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    # After update, calling tokenizer of llama3 will
    # auto add bos id for the tokens. ヽ(｀⌒´)ﾉ
    # pyre-fixme[53]: Captured variable `bos_token_id` is not annotated.
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def safe_tokenizer_llama3(text):
        input_ids = tokenizer(text).input_ids
        if input_ids[0] == bos_token_id:
            input_ids = input_ids[1:]
        return input_ids

    nl_tokens = tokenizer.convert_tokens_to_ids("\n\n")

    # chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{%- if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{%- endif %}"
    chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    tokenizer.chat_template = chat_template

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
            # pyre-fixme[6]: For 1st argument expected `Union[int, str]` but got `slice`.
        )[:-4]

        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)

            conv = [{"role": role, "content": content}]
            # First is bos token we don't need here
            # pyre-fixme[6]: For 1st argument expected `Union[int, str]` but got
            #  `slice`.
            encode_id = tokenizer.apply_chat_template(conv)[1:-4]
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    print("input_ids", input_ids, flush=True)
    print("targets", targets, flush=True)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess_llama_3_1(
    # pyre-fixme[2]: Parameter must be annotated.
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    # pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
    #  `typing.Dict[<key type>, <value type>]` to avoid runtime subscripting errors.
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            if sentence["from"] == "Answer":
                sentence["from"] = "gpt"  # data bug
            role = roles[sentence["from"]]
            # assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [
                longvu_tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    # remove the first bos token
    if input_ids[0][0] == input_ids[0][1] == tokenizer.bos_token_id:
        input_ids = input_ids[:, 1:]
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3_1

    # Mask targets
    sep = "<|start_header_id|>" + conv.roles[1] + "<|end_header_id|>" + "\n\n"
    # sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.shape[0])

        rounds = conversation.split(conv.tokenizer.eos_token)
        rounds = [rounds[0]] + [
            rounds[idx] + rounds[idx + 1] for idx in range(1, len(rounds) - 1, 2)
        ]

        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2 and i != 0:
                break

            if i == 0:
                round_len = len(tokenizer(rou, add_special_tokens=False).input_ids)
                instruction_len = len(
                    tokenizer(rou, add_special_tokens=False).input_ids
                )

            else:
                parts[0] += sep
                if has_image:
                    round_len = len(longvu_tokenizer_image_token(rou, tokenizer)) + 1
                    instruction_len = len(longvu_tokenizer_image_token(parts[0], tokenizer))
                else:
                    round_len = len(tokenizer(rou).input_ids) + 1
                    instruction_len = len(tokenizer(parts[0]).input_ids)

            # if i > 0: round_len += 1
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX
        cur_len = cur_len + len(tokenizer(sep, add_special_tokens=False).input_ids)

        # if cur_len > tokenizer.model_max_length: print(f"WARNING: max length context")
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_llama_3_2(
    # pyre-fixme[2]: Parameter must be annotated.
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    # pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
    #  `typing.Dict[<key type>, <value type>]` to avoid runtime subscripting errors.
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [
                longvu_tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    # remove the first bos token
    if input_ids[0][0] == input_ids[0][1] == tokenizer.bos_token_id:
        input_ids = input_ids[:, 1:]
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3_2

    # Mask targets
    sep = "<|start_header_id|>" + conv.roles[1] + "<|end_header_id|>" + "\n\n"
    # sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.shape[0])

        rounds = conversation.split(conv.tokenizer.eos_token)
        rounds = [rounds[0]] + [
            rounds[idx] + rounds[idx + 1] for idx in range(1, len(rounds) - 1, 2)
        ]

        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2 and i != 0:
                break

            if i == 0:
                round_len = len(tokenizer(rou, add_special_tokens=False).input_ids)
                instruction_len = len(
                    tokenizer(rou, add_special_tokens=False).input_ids
                )

            else:
                parts[0] += sep
                if has_image:
                    round_len = len(longvu_tokenizer_image_token(rou, tokenizer)) + 1
                    instruction_len = len(longvu_tokenizer_image_token(parts[0], tokenizer))
                else:
                    round_len = len(tokenizer(rou).input_ids) + 1
                    instruction_len = len(tokenizer(parts[0]).input_ids)

            # if i > 0: round_len += 1
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX
        cur_len = cur_len + len(tokenizer(sep, add_special_tokens=False).input_ids)

        # if cur_len > tokenizer.model_max_length: print(f"WARNING: max length context")
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_phi3(
    # pyre-fixme[2]: Parameter must be annotated.
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    # pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
    #  `typing.Dict[<key type>, <value type>]` to avoid runtime subscripting errors.
) -> Dict:
    conv = conversation_lib.conv_templates["phi3"].copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack(
            [
                longvu_tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(
                conv.sep.join(rounds[conv_idx : conv_idx + 2])
            )  # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(longvu_tokenizer_image_token(rou, tokenizer))
                instruction_len = len(longvu_tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i == 0:
                round_len += 1
                instruction_len += 1
            else:
                round_len -= 2
                instruction_len -= 2

            if (
                i != 0
                and getattr(tokenizer, "legacy", False)
                and IS_TOKENIZER_GREATER_THAN_0_14
            ):
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    # pyre-fixme[2]: Parameter must be annotated.
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    # pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack(
            [
                longvu_tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(
                conv.sep.join(rounds[conv_idx : conv_idx + 2])
            )  # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            if has_image:
                round_len = len(longvu_tokenizer_image_token(rou, tokenizer))
                instruction_len = len(longvu_tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if (
                i != 0
                and getattr(tokenizer, "legacy", False)
                and IS_TOKENIZER_GREATER_THAN_0_14
            ):
                round_len += 1
                instruction_len += 1
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    # pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
    #  `typing.Dict[<key type>, <value type>]` to avoid runtime subscripting errors.
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        # pyre-fixme[6]: For 1st argument expected `Union[slice, SupportsIndex]` but
        #  got `str`.
        assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
        # pyre-fixme[16]: `str` has no attribute `__setitem__`.
        source[0]["value"] = DEFAULT_IMAGE_TOKEN
        conversation = (
            # pyre-fixme[6]: For 1st argument expected `Union[slice, SupportsIndex]`
            #  but got `str`.
            source[0]["value"]
            # pyre-fixme[6]: For 1st argument expected `Union[slice, SupportsIndex]`
            #  but got `str`.
            + source[1]["value"]
            + conversation_lib.default_conversation.sep
        )
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [
        longvu_tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversations
    ]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        # pyre-fixme[6]: For 1st argument expected `Union[slice, SupportsIndex]` but
        #  got `str`.
        tokenized_len = len(longvu_tokenizer_image_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    # pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
    #  `typing.Dict[<key type>, <value type>]` to avoid runtime subscripting errors.
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if (
        conversation_lib.default_conversation.sep_style
        == conversation_lib.SeparatorStyle.PLAIN
    ):
        return preprocess_plain(sources, tokenizer)
    if (
        conversation_lib.default_conversation.sep_style
        == conversation_lib.SeparatorStyle.LLAMA_2
    ):
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "llama3":
        return preprocess_llama3(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "llama3_1":
        return preprocess_llama_3_1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "llama3_2":
        return preprocess_llama_3_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "phi3":
        return preprocess_phi3(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def get_tokenize_len(prompts):
        return [len(longvu_tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [
            longvu_tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
            for prompt in conversations
        ]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            # pyre-fixme[61]: `header` is undefined, or not always defined.
            # pyre-fixme[6]: For 1st argument expected `Union[slice, SupportsIndex]`
            #  but got `str`.
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn(
                # pyre-fixme[61]: `header` is undefined, or not always defined.
                # pyre-fixme[6]: For 1st argument expected `Union[slice,
                #  SupportsIndex]` but got `str`.
                [header] + [s["value"] for s in source],
                tokenizer,
            )["input_ids_lens"]
        # pyre-fixme[6]: For 1st argument expected `Union[slice, SupportsIndex]` but
        #  got `str`.
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        # pyre-fixme[2]: Parameter must be annotated.
        data_args,
    ) -> None:
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        self.tokenizer = tokenizer
        # pyre-fixme[4]: Attribute must be annotated.
        self.list_data_dict = list_data_dict
        # pyre-fixme[4]: Attribute must be annotated.
        self.data_args = data_args

    @property
    # pyre-fixme[3]: Return type must be annotated.
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self) -> List[int]:
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    def __len__(self) -> int:
        return len(self.list_data_dict)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        has_image = True
        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            full_path = os.path.join(image_folder, image_file)
            if not os.path.exists(full_path):
                print(full_path)
                has_image = False
                sources = copy.deepcopy([e["conversations"] for e in sources])
            else:
                image = Image.open(full_path).convert("RGB")
                if self.data_args.image_aspect_ratio == "sam":
                    image = np.array(image)[:, :, ::-1]
                if self.data_args.image_aspect_ratio == "pad":
                    # pyre-fixme[3]: Return type must be annotated.
                    # pyre-fixme[2]: Parameter must be annotated.
                    def expand2square(pil_img, background_color):
                        width, height = pil_img.size
                        if width == height:
                            return pil_img
                        elif width > height:
                            result = Image.new(
                                pil_img.mode, (width, width), background_color
                            )
                            result.paste(pil_img, (0, (width - height) // 2))
                            return result
                        else:
                            result = Image.new(
                                pil_img.mode, (height, height), background_color
                            )
                            result.paste(pil_img, ((height - width) // 2, 0))
                            return result

                    image = expand2square(
                        image, tuple(int(x * 255) for x in processor.image_mean)
                    )
                    image = processor.preprocess(image, return_tensors="pt")[
                        "pixel_values"
                    ][0]
                else:
                    if self.data_args.image_aspect_ratio != "sam":
                        image = processor.preprocess(image, return_tensors="pt")[
                            "pixel_values"
                        ][0]
                sources = preprocess_multimodal(
                    copy.deepcopy([e["conversations"] for e in sources]), self.data_args
                )
        elif "video" in sources[0]:
            video_file = self.list_data_dict[i]["video"]
            video_folder = self.data_args.image_folder
            if "webvid" in video_folder:
                video_file = os.path.join(video_folder, "videos", video_file)
            elif "ActivityNet" in video_folder:
                video_file = os.path.join(video_folder, "train_val", video_file)
            else:
                video_file = os.path.join(video_folder, video_file)
            if not os.path.exists(video_file):
                print("nonexist: {}".format(video_file), flush=True)
                for sub_folder in os.listdir(video_folder):
                    if os.path.isdir(os.path.join(video_folder, sub_folder)):
                        for sub_sub_folder in os.listdir(
                            os.path.join(video_folder, sub_folder)
                        ):
                            print("folder", sub_folder, sub_sub_folder)
                has_image = False
                sources = copy.deepcopy([e["conversations"] for e in sources])
            else:
                if video_file.endswith(".webm"):
                    has_image = False
                    sources = copy.deepcopy([e["conversations"] for e in sources])
                else:
                    try:
                        # if video_file.endswith(".webm"):
                        #     video_webm = VideoFileClip(video_file)
                        #     video_frames = np.array(list(video_webm.iter_frames()))
                        #     sample_fps = round(video_webm.fps / self.data_args.video_fps)
                        #     frame_idx = [i for i in range(0, len(video_frames), sample_fps)]
                        #     video = video_frames[frame_idx]
                        # else:
                        vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
                        sample_fps = round(vr.get_avg_fps() / self.data_args.video_fps)
                        frame_idx = [i for i in range(0, len(vr), sample_fps)]
                        video = vr.get_batch(frame_idx).asnumpy()
                        if self.data_args.image_aspect_ratio == "sam":
                            image = video[:, :, :, ::-1][:100]
                        else:
                            processor = self.data_args.image_processor
                            image = processor.preprocess(video, return_tensors="pt")[
                                "pixel_values"
                            ]
                        sources = preprocess_multimodal(
                            copy.deepcopy([e["conversations"] for e in sources]),
                            self.data_args,
                        )
                    except:
                        has_image = False
                        sources = copy.deepcopy([e["conversations"] for e in sources])
        else:
            has_image = False
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            # pyre-fixme[6]: For 1st argument expected `Sequence[str]` but got
            #  `Union[Dict[typing.Any, typing.Any], List[typing.Any]]`.
            sources,
            self.tokenizer,
            has_image=has_image,
        )
        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0]
            )

        # image exist in the data
        if has_image:
            if "image" in self.list_data_dict[i]:
                # pyre-fixme[61]: Local variable `image` is undefined, or not always defined.
                data_dict["image"] = image
            elif "video" in self.list_data_dict[i]:
                # pyre-fixme[61]: Local variable `image` is undefined, or not always defined.
                data_dict["image"] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            # crop_size = self.data_args.image_processor.crop_size
            # data_dict["image"] = torch.zeros(3, crop_size["height"], crop_size["width"])
            if self.data_args.image_aspect_ratio == "sam":
                if "video" in self.list_data_dict[i]:
                    data_dict["image"] = np.zeros((1, 1024, 1024, 3)).astype(np.uint8)
                else:
                    data_dict["image"] = np.zeros((1024, 1024, 3)).astype(np.uint8)
            else:
                crop_size = self.data_args.image_processor.crop_size
                if "video" in self.list_data_dict[i]:
                    data_dict["image"] = torch.zeros(
                        1, 3, crop_size["height"], crop_size["width"]
                    )
                else:
                    data_dict["image"] = torch.zeros(
                        3, crop_size["height"], crop_size["width"]
                    )

        if has_image:
            if self.data_args.num_points > 0:
                if "box" in self.list_data_dict[i]:
                    x1, y1, x2, y2 = self.list_data_dict[i]["box"]
                    points = []
                    x = random.uniform(x1, x2)
                    y = random.uniform(y1, y2)
                    points.append(torch.tensor([x, y, 1]))
                    for _ in range(1, self.data_args.num_points):
                        points.append(torch.tensor([0, 0, 0]))
                    points = torch.stack(points, dim=0)
                    data_dict["point"] = points
                else:
                    if "point" in self.list_data_dict[i]:
                        points = torch.tensor(self.list_data_dict[i]["point"])
                        data_dict["point"] = points
                    else:
                        points = []
                        grid = int(np.sqrt(self.data_args.num_points))
                        height, width = image.shape[0], image.shape[1]
                        for i in range(grid):
                            for j in range(grid):
                                points.append(
                                    torch.tensor(
                                        [
                                            width / grid / 2.0 + i / grid * width,
                                            height / grid / 2.0 + j / grid * height,
                                            1,
                                        ]
                                    )
                                )
                        points = torch.stack(points, dim=0)
                        data_dict["point"] = points
        elif self.data_args.is_multimodal:
            if self.data_args.num_points > 0:
                points = []
                grid = int(np.sqrt(self.data_args.num_points))
                height, width = data_dict["image"].shape[0], data_dict["image"].shape[1]
                for i in range(grid):
                    for j in range(grid):
                        points.append(
                            torch.tensor(
                                [
                                    width / grid / 2.0 + i / grid * width,
                                    height / grid / 2.0 + j / grid * height,
                                    1,
                                ]
                            )
                        )
                points = torch.stack(points, dim=0)
                data_dict["point"] = points

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    # pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
    #  `typing.Dict[<key type>, <value type>]` to avoid runtime subscripting errors.
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            # pyre-fixme[6]: For 3rd argument expected `float` but got `Optional[int]`.
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            # pyre-fixme[6]: For 1st argument expected `Tensor` but got `Optional[int]`.
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        # if "image" in instances[0]:
        #     images = [instance["image"] for instance in instances]
        #     if all(x is not None and x.shape == images[0].shape for x in images):
        #         if type(images[0]) is torch.Tensor:
        #             batch["images"] = torch.stack(images)
        #         else:
        #
        #             batch["images"] = np.stack(images)
        #     else:
        #
        #         #  `List[typing.Any]`.
        #         batch["images"] = images

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            # pyre-fixme[6]: For 2nd argument expected `Tensor` but got `List[typing.Any]`.
            batch["images"] = images

            if "point" in instances[0]:
                points = [instance["point"] for instance in instances]
                batch["points"] = torch.stack(points)

        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    # pyre-fixme[2]: Parameter must be annotated.
    data_args,
    # pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
    #  `typing.Dict[<key type>, <value type>]` to avoid runtime subscripting errors.
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )
