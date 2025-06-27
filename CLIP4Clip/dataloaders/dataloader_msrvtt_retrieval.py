from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import defaultdict
import json
import random
from dataloaders.rawvideo_util import RawVideoExtractor
import decord
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch


import time
import threading




class TimeoutException(Exception):
    pass

def timeout_handler(func, args=(), kwargs={}, timeout_duration=5):
    """
    为函数设置超时机制
    :param func: 要执行的函数
    :param args: 函数的位置参数
    :param kwargs: 函数的关键字参数
    :param timeout_duration: 超时时间（秒）
    :return: 函数的返回值，如果超时则抛出 TimeoutException
    """
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None
            self.exception = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except Exception as e:
                self.exception = e

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.is_alive():
        raise TimeoutException("Data loading timed out.")
    if it.exception:
        raise it.exception
    return it.result

class MSRVTT_DataLoader(Dataset):
    """MSRVTT dataset loader."""
    def __init__(
            self,
            csv_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):

        self.data = pd.read_csv(csv_path)
        self.features_path = features_path
        self.feature_framerate = feature_framerate  #1.0 代表每秒采集几帧
        self.max_words = max_words    #30
        self.max_frames = max_frames   #100
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order  #frame的选择顺序
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return len(self.data)

    def _get_text(self, video_id, sentence):
        choice_video_ids = [video_id]
        n_caption = len(choice_video_ids)

        k = n_caption
        pairs_text = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.int64)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(sentence)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]  #大于30的单词就直接扔掉

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)   #为了pad时使用
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def extract_and_resize_frames(self, video_path):
    # 初始化 VideoReader，使用 CPU 进行解码
        vr = decord.VideoReader(video_path)
        # 提取指定索引的帧
        sorted_frame_idx = random.sample(range(len(vr)), 16)
        sorted_frame_idx = sorted(sorted_frame_idx)
        frames = vr.get_batch(sorted_frame_idx).asnumpy()

        # 定义图像转换操作
            
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

        resized_frames = []
        for frame in frames:
            # 对每一帧应用转换操作
            resized_frame = transform(frame)
            resized_frames.append(resized_frame)
        resized_frames = torch.tensor(np.stack(resized_frames))
        resized_frames = resized_frames.unsqueeze(1)
        return resized_frames
    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.ones((len(choice_video_ids), self.max_frames), dtype=np.int64)

        # Pair x 16 x 1 x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float64)

        for i, video_id in enumerate(choice_video_ids):
            # Individual for YoucokII dataset, due to it video format
            # video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))
            video_path = video_id
            video_slice = self.extract_and_resize_frames(video_path)
            slice_len = video_slice.shape[0]
            video[i][:slice_len, ...] = video_slice


        return video, video_mask


    def __getitem__(self, idx):
        # 这个东西注意一下
        # video 9770  类似于path

        video_id = self.data['video_id'].values[idx]
        # sentence  'a person is connecting something to system'
        sentence = self.data['sentence'].values[idx]
        #print(video_id)
        # video按照原来的就可以了，按照之前弄的clip的做。
        # ClipTokenizer()
        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, sentence)
        try:
            video, video_mask = timeout_handler(self._get_rawvideo, args=(choice_video_ids, ),  timeout_duration=20)
            return pairs_text, pairs_mask, pairs_segment, video, video_mask
        except TimeoutException:
            video_mask = np.ones((len(choice_video_ids), self.max_frames), dtype=np.int64)
            video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float64)
            return np.zeros_like(pairs_text), np.zeros_like(pairs_mask), np.zeros_like(pairs_segment), video, video_mask
            
class MSRVTT_TrainDataLoader(Dataset):
    """MSRVTT train dataset loader."""
    def __init__(
            self,
            csv_path,
            json_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            unfold_sentences=False,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):
        self.csv = pd.read_csv(csv_path)
        self.data = json.load(open(json_path, 'r'))
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.unfold_sentences = unfold_sentences
        self.sample_len = 0
        if self.unfold_sentences:
            train_video_ids = list(self.csv['video_id'].values)
            self.sentences_dict = {}
            for itm in self.data['sentences']:
                if itm['video_id'] in train_video_ids:
                    self.sentences_dict[len(self.sentences_dict)] = (itm['video_id'], itm['caption'])
            self.sample_len = len(self.sentences_dict)
        else:
            num_sentences = 0
            self.sentences = defaultdict(list)
            s_video_id_set = set()
            for itm in self.data['sentences']:
                self.sentences[itm['video_id']].append(itm['caption'])
                num_sentences += 1
                s_video_id_set.add(itm['video_id'])

            # Use to find the clips in the same video
            self.parent_ids = {}
            self.children_video_ids = defaultdict(list)
            for itm in self.data['videos']:
                vid = itm["video_id"]
                url_posfix = itm["url"].split("?v=")[-1]
                self.parent_ids[vid] = url_posfix
                self.children_video_ids[url_posfix].append(vid)
            self.sample_len = len(self.csv)

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, caption=None):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.int64)

        for i, video_id in enumerate(choice_video_ids):
            if caption is not None:
                words = self.tokenizer.tokenize(caption)
            else:
                words = self._get_single_text(video_id)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_single_text(self, video_id):
        rind = random.randint(0, len(self.sentences[video_id]) - 1)
        caption = self.sentences[video_id][rind]
        words = self.tokenizer.tokenize(caption)
        return words
    def extract_and_resize_frames(self, video_path):
    # 初始化 VideoReader，使用 CPU 进行解码
        vr = decord.VideoReader(video_path)
        # 提取指定索引的帧
        sorted_frame_idx = random.sample(range(len(vr)), 16)
        sorted_frame_idx = sorted(sorted_frame_idx)
        frames = vr.get_batch(sorted_frame_idx).asnumpy()

        # 定义图像转换操作
            
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

        resized_frames = []
        for frame in frames:
            # 对每一帧应用转换操作
            resized_frame = transform(frame)
            resized_frames.append(resized_frame)
        resized_frames = torch.tensor(np.stack(resized_frames))
        resized_frames = resized_frames.unsqueeze(1)
        return resized_frames
    

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.ones((len(choice_video_ids), self.max_frames), dtype=np.int64)

        # Pair x 16 x 1 x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float64)

        for i, video_id in enumerate(choice_video_ids):
            # Individual for YoucokII dataset, due to it video format
            # video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))
            video_path = video_id
            video_slice = self.extract_and_resize_frames(video_path)
            slice_len = video_slice.shape[0]
            video[i][:slice_len, ...] = video_slice


        return video, video_mask

    def __getitem__(self, idx):
        if self.unfold_sentences:
            video_id, caption = self.sentences_dict[idx]
        else:
            video_id, caption = self.csv['video_id'].values[idx], None
        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)
        try:
            video, video_mask = timeout_handler(self._get_rawvideo, args=(choice_video_ids, ),  timeout_duration=20)
            return pairs_text, pairs_mask, pairs_segment, video, video_mask
        except TimeoutException:
            video_mask = np.ones((len(choice_video_ids), self.max_frames), dtype=np.int64)

            video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float64)
            return np.zeros_like(pairs_text), np.zeros_like(pairs_mask), np.zeros_like(pairs_segment), video, video_mask

