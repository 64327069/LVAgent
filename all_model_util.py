from __future__ import annotations
from tqdm import tqdm
import json
import random
import argparse
import base64
import logging
import math
import math
import os
import sys
import time
import warnings
from functools import lru_cache
from io import BytesIO
import re
import requests
import torch
import torchvision
from packaging import version
from PIL import Image
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode



logger = logging.getLogger(__name__)

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        image_obj = Image.open(requests.get(image, stream=True).raw)
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = image_obj.convert("RGB")
    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))

    return image


def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
        nframes = total_frames / video_fps * fps
        nframes = min(max(nframes, min_frames), max_frames)
        nframes = round_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes


def _read_video_torchvision(
    ele: dict,
) -> torch.Tensor:
    """read video using torchvision.io.read_video

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    video_path = ele["video"]
    if version.parse(torchvision.__version__) < version.parse("0.19.0"):
        if "http://" in video_path or "https://" in video_path:
            warnings.warn("torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")
        if "file://" in video_path:
            video_path = video_path[7:]
    st = time.time()
    video, audio, info = io.read_video(
        video_path,
        start_pts=ele.get("video_start", 0.0),
        end_pts=ele.get("video_end", None),
        pts_unit="sec",
        output_format="TCHW",
    )
    total_frames, video_fps = video.size(0), info["video_fps"]
    logger.info(f"torchvision:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    video = video[idx]
    return video


def is_decord_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("decord") is not None


def _read_video_decord(
    ele, sample_idx=None
):
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    import decord
    video_path = ele["video"]
    st = time.time()
    vr = decord.VideoReader(video_path)
    # TODO: support start_pts and end_pts
    if 'video_start' in ele or 'video_end' in ele:
        raise NotImplementedError("not support start_pts and end_pts in decord for now.")
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    if sample_idx:
        video = vr.get_batch(sample_idx).asnumpy()
    else:
        idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
        video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    return video


VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord,
    "torchvision": _read_video_torchvision,
}

FORCE_QWENVL_VIDEO_READER = os.getenv("FORCE_QWENVL_VIDEO_READER", None)


@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    if FORCE_QWENVL_VIDEO_READER is not None:
        video_reader_backend = FORCE_QWENVL_VIDEO_READER
    elif is_decord_available():
        video_reader_backend = "decord"
    else:
        video_reader_backend = "torchvision"
    print(f"qwen-vl-utils using {video_reader_backend} to read video.", file=sys.stderr)
    return video_reader_backend


def fetch_video(ele, sample_idx=None, image_factor= IMAGE_FACTOR):
    if isinstance(ele["video"], str):
        video_reader_backend = get_video_reader_backend()

        if sample_idx:
            video = _read_video_decord(ele,sample_idx)
        else:
            video = VIDEO_READER_BACKENDS[video_reader_backend](ele)
        nframes, _, height, width = video.shape

        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels = ele.get("max_pixels", max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
        return video
    else:
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = [
            fetch_image({"image": video_element, **process_info}, size_factor=image_factor)
            for video_element in ele["video"]
        ]
        nframes = ceil_by_factor(len(images), FRAME_FACTOR)
        if len(images) < nframes:
            images.extend([images[-1]] * (nframes - len(images)))
        return images


def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele["type"] in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
    conversations: list[dict] | list[list[dict]], sample_idx=None
):
    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            if sample_idx:
                video_inputs.append(fetch_video(vision_info, sample_idx))
            else:
                video_inputs.append(fetch_video(vision_info))
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    return image_inputs, video_inputs

def insert_subtitles_into_frames(subtitles):
    interleaved_list = []
    cur_i = 0
    for subtitle in subtitles:
        if 'line' in subtitle.keys():
            subtitle_text = subtitle["line"]
        elif 'text' in subtitle.keys():
            subtitle_text = subtitle["text"]
        interleaved_list.append(subtitle_text)

    return "\n".join(interleaved_list)


def ego_info_history(anno, history_info, previous_info):
    option = f"Options: A: {anno['A']}, B: {anno['B']}, C: {anno['C']}, D: {anno['D']}, E: {anno['E']}\n"
    option_prompt = f"""
    Long video details: Question:\n{anno['question']}\nOptions:\n{option}\n.
    History discussion info:\n{history_info}\nPrevious key info:\n{previous_info}.
    Based on these, re-identify the key information needed to answer the question. Focus on visual cues, context, and temporal relationships within the frames. Limit your response to 50 words.
    """
    return option_prompt

def ego_info(anno):

    question = f"Question: {anno['question']}\n Options: A: {anno['A']}, B: {anno['B']}, C: {anno['C']}, D: {anno['D']}, E: {anno['E']}\n"

    decide_watch = "You are given a single-choice question, multiple-choice options, subtitles, some frames of the long video. You should not only look at the textual information but also consider the input visual information, taking everything into account. If you can answer the question accurately and comprehensively based on the existing information especially the visual information, and further watching the entire video will not significantly improve the quality of the answer, then you don't need to watch the entire video and can answer 'No'. However, if the existing information is not sufficient to fully answer the question, and watching the entire video may obtain information crucial for answering the question, please reply 'Yes'\n" + question + '\nOutput: [Yes/No]'


    prompt_info = f"""
Given four randomly sampled frames from a long video, a question, and multiple-choice options. Please identify the key information needed to answer the question in one sentence. Focus on visual cues, context, and temporal relationships within the frames. Limit your response to 50 words.

Question: {anno['question']}
Options: 
A: {anno["A"]}
B: {anno["B"]}
C: {anno["C"]}
D: {anno["D"]}
E: {anno["E"]}
"""
    return decide_watch, prompt_info, get_ego_prompt(anno)

def mme_info_intern(doc):
    question = doc["question"]
    option = "\n".join([f"{opt}" for i, opt in enumerate(doc["options"])])
    question = question + "\n" + option

    decide_watch = "You are given a single-choice question, options, subtitles, some frames of the long video. You should not only look at the textual information but also consider the input visual information, taking everything into account. If you can answer the question accurately and comprehensively based on the existing information especially the visual information, and further watching the entire video will not significantly improve the quality of the answer, then you don't need to watch the entire video and can answer 'No'. However, if the existing information is not sufficient to fully answer the question, and watching the entire video may obtain information crucial for answering the question, please reply 'Yes'\n" + question + '\nOutput: [Yes/No]'

    option_prompt = "Given four randomly sampled frames from a long video, subtitles, a question, and multiple-choice options, identify the key information needed to answer the question. Focus on visual cues, context, and temporal relationships within the frames. Limit your response to 50 words."
    
    info_prompt = option_prompt + '\n' + question
    return decide_watch, info_prompt, get_mme_wo_subtitle_prompt(doc)

def mme_sub_info_intern_history(doc, history_info, previous_info):
    cache_dir = '/fs-computility/video/shared/wangzikang/videomme'
    subtitle_path = os.path.join(cache_dir, "subtitle", doc["videoID"] + ".srt")
    if os.path.exists(subtitle_path):  # Denote have subtitle
        subtitle = open(subtitle_path).readlines()
    else:
        subtitle = ""
    subtitles_prompt = "This video's subtitles are listed below: \n"
    if subtitle == "":
        subtitle = "No subtitles available"
    else:
        textlist = []
        for ele in subtitle:
            pattern = r'<font color="white" size=".72c">(.*?)</font>'
            matches = re.findall(pattern, ele)
            if matches:
                textlist.append(matches[0])
        subtitle = "\n".join(textlist)
    option = "\n".join([f"{opt}" for i, opt in enumerate(doc["options"])])
    option_prompt = f"""
    Long video details: Subtitle:\n{subtitle}\nQuestion:\n{doc['question']}\nOptions:\n{option}\n.
    History discussion info:\n{history_info}\nPrevious key info:\n{previous_info}.
    Based on these, re-identify the key information needed to answer the question. Focus on visual cues, context, and temporal relationships within the frames. Limit your response to 50 words.
    """
    return option_prompt

def mme_sub_info_intern_wosub_history(doc, history_info, previous_info):
    option = "\n".join([f"{opt}" for i, opt in enumerate(doc["options"])])
    option_prompt = f"""
    Long video details: Question:\n{doc['question']}\nOptions:\n{option}\n.
    History discussion info:\n{history_info}\nPrevious key info:\n{previous_info}.
    Based on these, re-identify the key information needed to answer the question. Focus on visual cues, context, and temporal relationships within the frames. Limit your response to 50 words.
    """
    return option_prompt

def mlvu_info_history(doc, history_info, previous_info):
    option = "A:{}\n B:{}\n C:{}\n D:{}\n".format(doc["candidates"][0], doc["candidates"][1], doc["candidates"][2], doc["candidates"][3])
    option_prompt = f"""
    Long video details: Question:\n{doc['question']}\nOptions:\n{option}\n.
    History discussion info:\n{history_info}\nPrevious key info:\n{previous_info}.
    Based on these, re-identify the key information needed to answer the question. Focus on visual cues, context, and temporal relationships within the frames. Limit your response to 50 words.
    """
    return option_prompt

def lvbench_info_history(doc, history_info, previous_info):
    option = "\n".join([". ".join([f'{chr(ord("A") + i)}', candidate]) for i, candidate in enumerate(doc['candidates'])])

    cache_dir = '/fs-computility/video/shared/wangzikang/longvideobench/subtitles'
    if '_en.json' in  doc["subtitle_path"]:
        subtitle_path = doc["subtitle_path"]
    else:
        subtitle_path = doc["subtitle_path"].replace(".json", "_en.json")
    with open(os.path.join(cache_dir, subtitle_path), "r") as f:
        subtitles = json.load(f)
    
    subtitle = insert_subtitles_into_frames(subtitles)


    option_prompt = f"""
    Long video details: Question:\n{doc['question']}\nOptions:\n{option}\nSubtitle:\n{subtitle}.
    History discussion info:\n{history_info}\nPrevious key info:\n{previous_info}.
    Based on these, re-identify the key information needed to answer the question. Focus on visual cues, context, and temporal relationships within the frames. Limit your response to 50 words.
    """
    return option_prompt


def mme_sub_info_intern(doc):
    cache_dir = '/fs-computility/video/shared/wangzikang/videomme'
    subtitle_path = os.path.join(cache_dir, "subtitle", doc["videoID"] + ".srt")
    if os.path.exists(subtitle_path):  # Denote have subtitle
        subtitle = open(subtitle_path).readlines()
    else:
        subtitle = ""
    subtitles_prompt = "This video's subtitles are listed below: \n"
    if subtitle == "":
        subtitle = "No subtitles available"
    else:
        textlist = []
        for ele in subtitle:
            pattern = r'<font color="white" size=".72c">(.*?)</font>'
            matches = re.findall(pattern, ele)
            if matches:
                textlist.append(matches[0])
        subtitle = "\n".join(textlist)

    option_prompt = "Given four randomly sampled frames from a long video, subtitles, a question, and multiple-choice options, identify the key information needed to answer the question. Focus on visual cues, context, and temporal relationships within the frames. Limit your response to 50 words."

    decide_watch = "You are given a single-choice question, multiple-choice options, subtitles, some frames of the long video. You should not only look at the textual information but also consider the input visual information, taking everything into account. Base on the provided information, your task is to determine whether it is necessary to answer this question by watching the whole video." 

    question = doc["question"]
    option = "\n".join([f"{opt}" for i, opt in enumerate(doc["options"])])
    question = question + "\n" + option
    info_prompt = subtitles_prompt + subtitle + "\n" + option_prompt + "\n" + question + "\n"
    decide_watch = subtitles_prompt + subtitle + "\n" + decide_watch + "\n" + question + "\n" + 'Output: [Yes/No]'
    return decide_watch, info_prompt, get_mme_subtitle_prompt(doc)





def get_ego_prompt(anno):
    time_instruciton = "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question. \n The Answer format is:\n Answer: xx\n"
    option = "Option:\nA: {}\nB: {}\nC: {}\nD: {}\nE: {}".format(anno["A"], anno["B"],anno["C"],anno["D"],anno["E"])
    question = f"{time_instruciton}\n" + f"Question: {anno['question']}\n" + f"{option}"
    return question
    
def get_mlvu_prompt(anno):

    option_prompt = "Given four randomly sampled frames from a long video, subtitles, a question, and multiple-choice options, identify the key information needed to answer the question. Focus on visual cues, context, and temporal relationships within the frames. Limit your response to 50 words."

    decide_watch = "You are given a single-choice question, multiple-choice options, subtitles, some frames of the long video. You should not only look at the textual information but also consider the input visual information, taking everything into account. Base on the provided information, your task is to determine whether it is necessary to answer this question by watching the whole video. Please just answer Yes or No." 

    time_instruciton = f"Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question. \nThe Answer format is: Answer: xx\n"
    option = "A:{}\n B:{}\n C:{}\n D:{}\n".format(anno["candidates"][0], anno["candidates"][1], anno["candidates"][2], anno["candidates"][3])
    question = f"Question: {anno['question']}\n" + f"{option}"
    time_instruciton = f"{time_instruciton}\n" + question
    info_prompt = option_prompt + "\n" + question + "\n"
    decide_watch = decide_watch + "\n" + question + "\n" + 'Output: [Yes/No]'
    return decide_watch, info_prompt, time_instruciton


def get_lvbench_prompt(doc):
    candidates = doc['candidates']

    question = doc["question"] + "\n" + "\n".join([". ".join([f'{chr(ord("A") + i)}', candidate]) for i, candidate in enumerate(candidates)])
    question = question

    cache_dir = '/fs-computility/video/shared/data/LongVideoBench/subtitles'
    if '_en.json' in  doc["subtitle_path"]:
        subtitle_path = doc["subtitle_path"]
    else:
        subtitle_path = doc["subtitle_path"].replace(".json", "_en.json")
    with open(os.path.join(cache_dir, subtitle_path), "r") as f:
        subtitles = json.load(f)
    
    option_prompt = "Given four randomly sampled frames from a long video, subtitles, a question, and multiple-choice options, identify the key information needed to answer the question. Focus on visual cues, context, and temporal relationships within the frames. Limit your response to 50 words."

    decide_watch = "You are given a single-choice question, multiple-choice options, subtitles, some frames of the long video. You should not only look at the textual information but also consider the input visual information, taking everything into account. If you don't need to watch the entire video, you can answer 'No'. However, if the existing information is not sufficient to fully answer the question, and watching the entire video may obtain information crucial for answering the question, please reply 'Yes'"

    subtitle = insert_subtitles_into_frames(subtitles)

    answer_prompt = f"This video's subtitles are listed below: \n{subtitle}.\n Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter of the correct option.\nQuestion: {question}\n" + 'Answer with the options letter from the given choices directly.\n'

    info_prompt = "This video's subtitles are listed below: \n" + subtitle + "\n"  + question + "\n" + option_prompt + "\n"

    decide_watch = "This video's subtitles are listed below: \n" + subtitle + "\n" + question + "\n" + decide_watch + "\n" + 'Output: [Yes/No]'
    return decide_watch, info_prompt, answer_prompt



def get_mme_subtitle_prompt(doc):
    cache_dir = '/fs-computility/video/shared/wangzikang/videomme'
    subtitle_path = os.path.join(cache_dir, "subtitle", doc["videoID"] + ".srt")
    if os.path.exists(subtitle_path):  # Denote have subtitle
        subtitle = open(subtitle_path).readlines()
    else:
        subtitle = ""
    subtitles_prompt = "This video's subtitles are listed below: \n"
    if subtitle == "":
        subtitle = "No subtitles available"
    else:
        textlist = []
        for ele in subtitle:
            pattern = r'<font color="white" size=".72c">(.*?)</font>'
            matches = re.findall(pattern, ele)
            if matches:
                textlist.append(matches[0])
        subtitle = "\n".join(textlist)

    option_prompt = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option."
    question = doc["question"]
    option = "\n".join([f"{opt}" for i, opt in enumerate(doc["options"])])
    question = question + "\n" + option
    full_prompt = subtitles_prompt + subtitle + "\n" + option_prompt + "\n" + question + "\n" + "The best answer is:"
    return full_prompt

def get_mme_wo_subtitle_prompt(doc):
    option_prompt = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option."
    question = doc["question"]
    option = "\n".join([f"{opt}" for i, opt in enumerate(doc["options"])])
    question = question + "\n" + option
    full_prompt = option_prompt + "\n" + question + "\n" + "The best answer is:"
    return full_prompt

