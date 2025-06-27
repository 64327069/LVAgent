
import torch
import torch.nn as nn 
from torch.utils.data import (SequentialSampler)
import numpy as np
import random
import os
import argparse
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.modeling import CLIP4Clip
from util import parallel_apply, get_logger
from decord import VideoReader, cpu
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
# torch.distributed.init_process_group(backend="nccl")
torch.cuda.init()
#from tensorboardX import SummaryWriter

global logger

def get_args(description='CLIP4Clip on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--max_words', type=int, default=60, help='')
    parser.add_argument('--max_frames', type=int, default=16, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')


    parser.add_argument('--text_num_hidden_layers', type=int, default=16, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=16, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=4, help="Layer NO. of cross.")


    parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"],
                        help="linear projection of flattened patches.")
    parser.add_argument('--sim_header', type=str, default="seqTransf",
                        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf"],
                        help="choice a similarity header.")

    args = parser.parse_args()     #获取命令行参数


    return args


def init_model(model_path, args):      #用来返回CLIP4模型
    model_state_dict = torch.load(model_path, map_location='cpu')
    model = CLIP4Clip.from_pretrained("cross-base", cache_dir="", state_dict=model_state_dict, task_config=args)
    model.to('cuda')
    return model

SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
def _get_text(tokenizer,  video_id, sentence):
    choice_video_ids = [video_id]
    n_caption = len(choice_video_ids)
    k = n_caption
    pairs_text = np.zeros((k, 77), dtype=np.int64)
    pairs_mask = np.zeros((k, 77), dtype=np.int64)
    pairs_segment = np.zeros((k, 77), dtype=np.int64)

    words = tokenizer.tokenize(sentence)

    words = [SPECIAL_TOKEN["CLS_TOKEN"]] + words
    total_length_with_CLS = 76
    if len(words) > total_length_with_CLS:
        words = words[:total_length_with_CLS]
    words = words + [SPECIAL_TOKEN["SEP_TOKEN"]]  #大于30的单词就直接扔掉

    input_ids = tokenizer.convert_tokens_to_ids(words)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)   #为了pad时使用
    while len(input_ids) < 77:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    pairs_text[0] = np.array(input_ids)
    pairs_mask[0] = np.array(input_mask)
    pairs_segment[0] = np.array(segment_ids)
    pairs_text= torch.Tensor(pairs_text).cuda()
    pairs_mask= torch.Tensor(pairs_mask).cuda()
    pairs_segment= torch.Tensor(pairs_segment).cuda()
    return pairs_text.long(), pairs_mask.long(), pairs_segment.long(), choice_video_ids



def extract_and_resize_frames(video_path, frame_indices):
    # 初始化 VideoReader，使用 CPU 进行解码
    vr = VideoReader(video_path, ctx=cpu(0))
    # 提取指定索引的帧
    frames = vr.get_batch(frame_indices).asnumpy()
    # 定义图像转换操作：将图像调整为 224x224 大小
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor()
    ])
    resized_frames = []
    for frame in frames:
        # 对每一帧应用转换操作
        resized_frame = transform(frame)
        resized_frames.append(resized_frame)
    # 将处理后的帧堆叠成一个张量
    resized_frames = torch.stack(resized_frames)
    return resized_frames

# 示例使用


def _get_rawvideo(self, choice_video_ids, video_path):
    video_mask = np.zeros((len(choice_video_ids), 16), dtype=np.int64)
    max_video_length = [0] * len(choice_video_ids)

    # 1 16 1 3 224 224
    video = np.zeros((len(choice_video_ids), 16, 1, 3, 224, 224), dtype=np.float64)


    raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
    raw_video_data = raw_video_data['video']
    if len(raw_video_data.shape) > 3:
        raw_video_data_clip = raw_video_data
        # L x T x 3 x H x W
        raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
        if self.max_frames < raw_video_slice.shape[0]:
            if self.slice_framepos == 0:
                video_slice = raw_video_slice[:self.max_frames, ...]
            elif self.slice_framepos == 1:
                video_slice = raw_video_slice[-self.max_frames:, ...]
            else:
                sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                video_slice = raw_video_slice[sample_indx, ...]
        else:
            video_slice = raw_video_slice
        # 注意slice_len
        video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

        slice_len = video_slice.shape[0]
        max_video_length[i] = max_video_length[0] if max_video_length[0] > slice_len else slice_len
        if slice_len < 1:
            pass
        else:
            video[i][:slice_len, ...] = video_slice

    # 1 1 16 video mask (一共几帧就送几个0)
    for i, v_length in enumerate(max_video_length):
        video_mask[i][:v_length] = [1] * v_length

    return video, video_mask

if __name__ == "__main__":
    args = get_args()
    model = init_model('/mnt/workspace/internvl2.5/InternVL/CLIP4Clip/output/2_20/pytorch_model.bin.8', args)
    text = "Does this surveillance footage contain any anomalies? If yes, which kind of anomaly?"
    sample_idx = get_frame_idx_path(video_path)

    # 基本上就是tokenizer弄一下
    # 最重要的代码基本上已经弄通了，yes
    tokenizer = ClipTokenizer()  
    # 输入video path还有sentence，输出文本的一些东西
    input_ids, input_mask, segment_ids, choice_video_ids = _get_text(tokenizer, video_path, text)
    # get_rawvideo这个怎么回事看看。
    video = extract_and_resize_frames(video_path, sample_idx).cuda()
    # visual_output 1.16 512
    video_mask = torch.Tensor([[1] * 16]).cuda()
    token_type_ids = torch.Tensor([[0] * 16]).cuda()
    visual_output = model.get_visual_output(video, video_mask=video_mask, shaped=True, video_frame=16)
    text_feat = model.get_sequence_output(input_ids, segment_ids, input_mask, shaped=True)

    b1b2_logits, *_tmp = model.get_similarity_logits(text_feat, visual_output, input_mask, video_mask,
                                                                     loose_type=True, eval = 'myeval')
    print(b1b2_logits)
                                                                     
    
    