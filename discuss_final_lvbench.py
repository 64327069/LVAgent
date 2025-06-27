
import os
import random
import json
from all_model_agent import  InternVL8B,InternVL78B, Llava72B
from all_model_util import *
from tqdm import tqdm
import copy
import time
import pandas as pd
import re
from decord import VideoReader, cpu
import threading 
from PIL import Image
import torchvision.transforms as T


# 加载clip模型
import torch
import torch.nn as nn
import numpy as np
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.modeling import CLIP4Clip

SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                 "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

def init_model(model_path, args):
    model_state_dict = torch.load(model_path, map_location='cpu')
    model = CLIP4Clip.from_pretrained("cross-base", cache_dir="", state_dict=model_state_dict, task_config=args)
    model.to('cuda')
    return model

def _get_text(tokenizer, video_id, sentence):
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
    words = words + [SPECIAL_TOKEN["SEP_TOKEN"]]

    input_ids = tokenizer.convert_tokens_to_ids(words)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)
    while len(input_ids) < 77:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    pairs_text[0] = np.array(input_ids)
    pairs_mask[0] = np.array(input_mask)
    pairs_segment[0] = np.array(segment_ids)
    pairs_text = torch.Tensor(pairs_text).cuda()
    pairs_mask = torch.Tensor(pairs_mask).cuda()
    pairs_segment = torch.Tensor(pairs_segment).cuda()
    return pairs_text.long(), pairs_mask.long(), pairs_segment.long(), choice_video_ids

def extract_and_resize_frames(video_path, frame_indices):
    vr = VideoReader(video_path, ctx=cpu(0))
    frames = vr.get_batch(frame_indices).asnumpy()
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor()
    ])
    resized_frames = []
    for frame in frames:
        resized_frame = transform(frame)
        resized_frames.append(resized_frame)
    resized_frames = torch.stack(resized_frames)
    return resized_frames

def run_clip4clip(model, video_path, text, sample_idx):
    # 定义 args 字典
    args = {
        'video_dim': 1024,
        'max_words': 60,
        'max_frames': 16,
        'feature_framerate': 1,
        'margin': 0.1,
        'hard_negative_rate': 0.5,
        'negative_weighting': 1,
        'n_pair': 1,
        'text_num_hidden_layers': 16,
        'visual_num_hidden_layers': 16,
        'cross_num_hidden_layers': 4,
        'linear_patch': "2d",
        'sim_header': "seqTransf"
    }
    tokenizer = ClipTokenizer()
    input_ids, input_mask, segment_ids, choice_video_ids = _get_text(tokenizer, video_path, text)
    video = extract_and_resize_frames(video_path, sample_idx).cuda()
    video_mask = torch.Tensor([[1] * 16]).cuda()
    token_type_ids = torch.Tensor([[0] * 16]).cuda()
    visual_output = model.get_visual_output(video, video_mask=video_mask, shaped=True, video_frame=16)
    text_feat = model.get_sequence_output(input_ids, segment_ids, input_mask, shaped=True)
    b1b2_logits, *_tmp = model.get_similarity_logits(text_feat, visual_output, input_mask, video_mask,
                                                     loose_type=True, eval='myeval')
    return b1b2_logits

model_path = 'CLIP4Clip\clip4clip_checkpoint\mViTT_new\pytorch_model_0.0011.bin.25'
    # 定义 args 字典
args = {
    'video_dim': 1024,
    'max_words': 60,
    'max_frames': 16,
    'feature_framerate': 1,
    'margin': 0.1,
    'hard_negative_rate': 0.5,
    'negative_weighting': 1,
    'n_pair': 1,
    'text_num_hidden_layers': 16,
    'visual_num_hidden_layers': 16,
    'cross_num_hidden_layers': 4,
    'linear_patch': "2d",
    'sim_header': "seqTransf"
}
asp_clip = init_model(model_path, args)
# 示例调用


def get_max_frame_block(video_path, text_prompt, sample_frame=16, sample_dict = None):
    best_clip_score_idx = []
    best_clip_score = 0
    select_block = 0
    sample_all_frames = []


    for i in range(1, 7):
        if sample_dict:
            sorted_frame_idx = sample_dict['all_samp'][i - 1]
        else:
            sorted_frame_idx = get_frame_idx_path(video_path, i, sample_frame)
        clip_score = run_clip4clip(asp_clip, video_path, text_prompt, sorted_frame_idx)
        if clip_score > best_clip_score:
            best_clip_score = clip_score
            best_clip_score_idx = sorted_frame_idx
            select_block = i
        sample_all_frames.append(sorted_frame_idx)
    if sample_dict:
        print('sample_block {} sample {}'.format(sample_dict['block'], select_block))

    return best_clip_score_idx, select_block, sample_all_frames

def get_frame_idx_path(video_path, round = 0, sample_frame=16, judge_whole=False):
    # 计算每一帧和text的相似度，然后选出最相似的帧，返回这些帧的路径
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    if len(vr) <= 96 or judge_whole:
        sorted_frame_idx = random.sample(range(len(vr)), sample_frame)
        sorted_frame_idx = sorted(sorted_frame_idx)
        return sorted_frame_idx
    inter = len(vr) // 6
    if 1<= round <= 6:
        sp = round * inter
        try:
            sorted_frame_idx = random.sample(range(sp-inter, sp - 1), sample_frame)
        except:
            sorted_frame_idx = random.sample(range(len(vr)), sample_frame)
    else:
        sorted_frame_idx = random.sample(range(len(vr)), sample_frame)

    sorted_frame_idx = sorted(sorted_frame_idx)
    return sorted_frame_idx
# 这个之后更改一下。

def get_result_first_round(agent_set, anno):
    answer_dict = {}
    sample_dict = {}
    decide_watch, info_prompt, get_mme_answer = get_lvbench_prompt(anno)
    # TODO: 请更改视频路径
    video_path = os.path.join('/fs-computility/video/shared/wangzikang/longvideobench/videos', anno['video_path'])

    def process_agent(agent):
        agent_name = agent.get_model_name()
        watch_ori = anno[agent_name]['watch'][0] if isinstance(anno[agent_name]['watch'], list) else anno[agent_name]['watch']
        watch = agent.get_answer(video_path, decide_watch, anno[agent_name]["watch_samp"])
        print(watch_ori,watch)
        watch = watch[0] if isinstance(watch, list) else watch

        if 'Yes' in watch:
            if 'sample_idx' in anno[agent_name].keys():
                sample_idx = anno[agent_name]['sample_idx']
            else:
                sample_idx = get_frame_idx_path(video_path, round=0, sample_frame=16)
            result = agent.get_answer(video_path, get_mme_answer, sample_idx)
        else:
            text_prompt_ori = anno[agent_name]['info']
            text_prompt = agent.get_answer(video_path, info_prompt, anno[agent_name]["watch_samp"])
            print(text_prompt_ori, "\n", text_prompt)

            if isinstance(text_prompt, list): text_prompt = text_prompt[0]
            if 'sample_dict' in anno[agent_name].keys():
                print('ori_samp')
                best_clip_score_idx, select_block, sample_all_frames = get_max_frame_block(video_path, text_prompt, 16, anno[agent_name]['sample_dict'])
            else:
                best_clip_score_idx, select_block, sample_all_frames = get_max_frame_block(video_path, text_prompt, sample_frame=16)
            if select_block == anno[agent_name]['block'] and 'sample_idx' in anno[agent_name].keys():
                result = agent.get_answer(video_path, get_mme_answer, anno[agent_name]['sample_idx'])
            else:
                result = agent.get_answer(video_path, get_mme_answer, sample_all_frames[select_block - 1])
        if isinstance(result, list):
            result = result[0]
        answer_dict[agent_name] = result.split('Answer: ')[-1][0]
        if 'Yes' not in watch and answer_dict[agent_name] == chr(ord('A') + anno['correct_choice']):
            sample_dict[agent_name] = {'all_samp': sample_all_frames, 'block': select_block}
            

    threads = []
    for agent in agent_set:
        thread = threading.Thread(target=process_agent, args=(agent,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    answer_set = set(answer_dict.values())
    print(answer_dict)
    return answer_set, answer_dict, sample_dict
# 把这个函数改好

# first round进行一部分的更改。
def get_result_second_round(agent_set, anno, history_info=None):
    answer_dict = {}
    sample_dict = {}
    decide_watch, info_prompt, get_mme_answer = get_lvbench_prompt(anno)
    # TODO: 请更改视频路径
    video_path = os.path.join('/fs-computility/video/shared/wangzikang/longvideobench/videos', anno['video_path'])

    def process_agent(agent):
        agent_name = agent.get_model_name()
        watch = anno[agent_name]['watch'][0] if isinstance(anno[agent_name]['watch'], list) else anno[agent_name]['watch']

        if 'Yes' in watch:

            sample_idx = get_frame_idx_path(video_path, round=0, sample_frame=16)
            result = agent.get_answer(video_path, get_mme_answer, sample_idx)
        else:
            text_prompt = history_info[agent_name]
            if isinstance(text_prompt, list): text_prompt = text_prompt[0]
            if agent_name == 'intern_78b':
                if 'sample_dict' in anno[agent_name].keys():
                    best_clip_score_idx, select_block, sample_all_frames = get_max_frame_block(video_path, text_prompt, 16, anno[agent_name]['sample_dict'])
                else:
                    best_clip_score_idx, select_block, sample_all_frames = get_max_frame_block(video_path, text_prompt, sample_frame=16)
                if select_block == anno[agent_name]['block']:
                    result = agent.get_answer(video_path, get_mme_answer, anno[agent_name]['sample_idx'])
                else:
                    result = agent.get_answer(video_path, get_mme_answer, sample_all_frames[select_block - 1])
            # if 'sample_dict' in anno[agent_name].keys():
            #     best_clip_score_idx, select_block, sample_all_frames = get_max_frame_block(video_path, text_prompt, 16, anno[agent_name]['sample_dict'])
            else:
                best_clip_score_idx, select_block, sample_all_frames = get_max_frame_block(video_path, text_prompt, sample_frame=16)

            
            # if select_block == anno[agent_name]['block']:
            #     result = agent.get_answer(video_path, get_mme_answer, anno[agent_name]['sample_idx'])
            # else:
            result = agent.get_answer(video_path, get_mme_answer, sample_all_frames[select_block - 1])

        if isinstance(result, list):
            result = result[0]
        answer_dict[agent_name] = result.split('Answer: ')[-1][0]
        if 'Yes' not in watch and answer_dict[agent_name] == chr(ord('A') + anno['correct_choice']):
            sample_dict[agent_name] = {'all_samp': sample_all_frames, 'block': select_block}
  

    threads = []
    for agent in agent_set:
        thread = threading.Thread(target=process_agent, args=(agent,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    answer_set = set(answer_dict.values())
    return answer_set, answer_dict, sample_dict
# TODO: Add the history info here.
def reason_process(agent_set, anno, answer_dict, sample_idx=None, history_info=None):
    # TODO: 请更改视频路径
    video_path = os.path.join('/fs-computility/video/shared/wangzikang/longvideobench/videos', anno['video_path'])
    
    ans_dict = {}

    def process_agent(agent):
        agent_name = agent.get_model_name()
        reason_prompt = "Given the video frames you've seen, and the question along with your answer, deeply analyze the logical steps and evidence from the frames that led you to provide this particular answer. The Question is: {}\n, The predict answer is {}\n.".format(
            anno['question'], anno['candidates'][ord(answer_dict[agent_name]) - ord('A')])
        if 'sample_idx' not in anno[agent_name]:
            rand_block = random.randint(1, 6)
            local_sample_idx = get_frame_idx_path(video_path, round=rand_block, sample_frame=16, judge_whole=True)
        else:
            local_sample_idx = anno[agent_name]['sample_idx']
        try:
            result = agent.get_answer(video_path, reason_prompt, local_sample_idx)
        except:
            print(anno['video_path'])
            rand_block = random.randint(1, 6)
            local_sample_idx = get_frame_idx_path(video_path, round=rand_block, sample_frame=16, judge_whole=True)
            result = agent.get_answer(video_path, reason_prompt, local_sample_idx)
        if isinstance(result, list):
            result = result[0]
        ans_dict[agent_name] = result

    threads = []
    for agent in agent_set:
        thread = threading.Thread(target=process_agent, args=(agent,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return ans_dict

def parse_json(text):
    if isinstance(text, list):
        text = text[0]
    text = re.sub(r"[\n\t]", "", text)
    text = text.replace('```json', '').replace('```', '')
    try:
        # First, try to directly parse the text as JSON
        return json.loads(text)
    except json.JSONDecodeError:
        # If direct parsing fails, use regex to extract JSON
        json_pattern = r"\{.*?\}|\[.*?\]"  # Pattern for JSON objects and arrays

        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                match = match.replace("'", '"')
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # If no JSON structure is found
        print("No valid JSON found in the text.")
        return None

def agent_back_process(agent_set, data):
    scores = {}

    for model in data.keys():
        score = 0
        for sub_dict in data.values():
            if model in sub_dict:
                score += int(sub_dict[model])
        scores[model] = score
    priority_order = ['intern_8b', 'llava_72b', 'intern_78b']
    min_score = min(scores.values())
    lowest_score_keys = [key for key, score in scores.items() if score == min_score]
    if len(lowest_score_keys) > 1:
        for priority_key in priority_order:
            if priority_key in lowest_score_keys:
                lowest_score_key = priority_key
                break
    else:
        lowest_score_key = lowest_score_keys[0]

    # 3. 排除掉分数最低的字典
    new_data = [key for key in data.keys() if key != lowest_score_key]

    return new_data, lowest_score_key, scores


def discuss_text_process(agent_set, anno, answer_dict, reason_dict):
    all_agent_name = [agent.get_model_name() for agent in agent_set]
    other_agent_name = copy.deepcopy(all_agent_name)
    sys_prompt = ""
    for key in reason_dict.keys():
        if isinstance(reason_dict[key], list):
            reason_dict[key] = reason_dict[key][0]
    discuss_dict = {}
    def process_agent(agent):
        agent_name = agent.get_model_name()
        local_other_agent_name = copy.deepcopy(other_agent_name)
        local_other_agent_name.remove(agent_name)
        if len(local_other_agent_name) == 1:
            answer_format = {agent_name: "1-10", local_other_agent_name[0]: "1-10"}
            discuss_prompt = f"""Given the answers and the reasoning for judgment from this model and two other models, please rate this model and the other two models. The score ranges from 1-10. Output in dictionary format.
            The question is: {anno['question']}, 
            The answer of this model is {answer_dict[agent_name]}, the reason is {reason_dict[agent_name]}.
            The answer of {local_other_agent_name[0]} model is {answer_dict[local_other_agent_name[0]]}, the reason is {reason_dict[local_other_agent_name[0]]}.
            The intern_8b model perform best. 
            You do not need to explain your answer, just give me scores as your answer following the answer_format.
            Please strictly follow the answer format! The answer_format is:
            {answer_format}
            """
        if len(local_other_agent_name) > 1:
            answer_format = {agent_name: "1-10", local_other_agent_name[0]: "1-10", local_other_agent_name[1]: "1-10"}
            discuss_prompt = f"""Given the answers and the reasoning for judgment from this model and two other models. 
            The question is: {anno['question']}
            The answer of this model is {answer_dict[agent_name]}, the reason is {reason_dict[agent_name]}.
            The answer of {local_other_agent_name[0]} model is {answer_dict[local_other_agent_name[0]]}, the reason is {reason_dict[local_other_agent_name[0]]}.
            The answer of {local_other_agent_name[1]} model is {answer_dict[local_other_agent_name[1]]}, the reason is {reason_dict[local_other_agent_name[1]]}.
            Please score the performance of this model an other two models base on their reasoning. The score ranges from 1-10. Output in dict format.
            You do not need to explain your answer, just give me scores as your answer following the answer_format.
            Please strictly follow the answer format! The answer_format is:
            {answer_format}
            """

        if agent_name == 'llava_72b':
            video_path = os.path.join('/fs-computility/video/shared/wangzikang/longvideobench/videos', anno['video_path'])
            temp = agent.get_answer(video_path, discuss_prompt, anno['llava_72b']['watch_samp'])
        else:
            temp = agent.get_text_answer(discuss_prompt)
        print(temp)
        temp = parse_json(temp)
        is_valid = True
        for key, value in temp.items():
            try:
                temp[key] = int(value)
            except:
                temp[key] = 'no_val'

        for value in temp.values():
            if not isinstance(value, int) or value < 1 or value > 10:
                is_valid = False
                break
        if is_valid:
            discuss_dict[agent_name] = temp
        else:
            discuss_dict[agent_name] = {'intern_78b': 8, 'intern_8b': 6, 'llava_72b': 7}

    threads = []
    for agent in agent_set:
        thread = threading.Thread(target=process_agent, args=(agent,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return discuss_dict

# 整理历史信息，要注意一下。
def generate_history_info(agent_set, anno, new_data, lowest_score_key, scores, reason_dict, answer_dict):

    all_prompt = " Discussion History Summary:\n"
    for data in new_data:
        all_prompt += "{}'s answer: {}\n Reason: {}\n The final score is {}.\n".format(data, anno['candidates'][ord(answer_dict[data]) - ord('A')], reason_dict[data], scores[data])
    all_prompt += "Removed Answer ({})\n Answer: {}\n Reason {}\n However, this reason was deemed unconvincing, so this answer was removed from the discussion.".format(
        lowest_score_key, anno['candidates'][ord(answer_dict[lowest_score_key]) - ord('A')], reason_dict[lowest_score_key])
    # 请从这些history中提取出要回答这个问题需要什么关键信息。
    # 给之前的info, 问题，答案
    history_info = {}
    for agent in agent_set:
        agent_name = agent.get_model_name()
        
        history_generate_prompt = lvbench_info_history(anno, all_prompt, anno[agent_name]['info'])
        if agent_name == 'llava_72b':
            # TODO: 更改视频路径
            video_path = os.path.join('/fs-computility/video/shared/wangzikang/longvideobench/videos', anno['video_path'])
            history_info[agent_name] = agent.get_answer(video_path, history_generate_prompt, anno['llava_72b']['watch_samp'])
        else:
            history_info[agent_name] = agent.get_text_answer(history_generate_prompt)
        
        if isinstance(history_info[agent_name], list): history_info[agent_name] = history_info[agent_name][0] 
    return history_info

internvl8b = InternVL8B()
internvl78b = InternVL78B()
llava72b = Llava72B()

all_anno_path = '/fs-computility/video/shared/wangzikang/Qwen2-VL-main/VideoAgent-master/concat_result/lvbench_all_result.json'

result_anno = json.load(open(all_anno_path))
all_num = 0
correct = 0
agent_set_ori = [internvl8b, internvl78b, llava72b]
import copy
ori_anno = copy.deepcopy(result_anno)

for kkk, anno in tqdm(enumerate(result_anno), desc="processing items"):
    agent_set = agent_set_ori
    all_num += 1
    print(anno['video_path'])
    print(anno['question'])
    print(anno['candidates'])
    # if 'On the left side of the screen, there is an image with several pieces of paper' not in anno['question']:
    #     continue
    answer_set, answer_dict, sample_dict = get_result_first_round(agent_set, anno)
    print('intern78 {}, intern_8 {}, llava_72b {}'.format(anno['intern_78b']['intern_78b'], anno['intern_8b']['intern_8b'], anno['llava_72b']['llava_72b']))
    anno['first_samp'] = sample_dict
    print(chr(ord('A') + anno['correct_choice']))
    if len(answer_set) < 3:
        values = list(answer_dict.values())
        for ans in answer_set:
            # 统计该值在列表中出现的次数
            if values.count(ans) >= 2:
                answer_set = ans
                break
        anno['final_answer'] = str(answer_set)
        if anno['final_answer'] == chr(ord('A') + anno['correct_choice']):
            correct += 1
        print("first round, correct{}, all_num {}, {}".format(correct, all_num, correct / all_num))
        with open(all_anno_path, 'w') as f:
            json.dump(result_anno, f)
        continue

    if 'first_round' in anno.keys():
        history_info = anno['first_round']['history_info']
        reason_dict = anno['first_round']['reason_dict']
        discuss_dict = discuss_text_process(agent_set, anno, answer_dict, reason_dict)
        anno['first_round']['discuss_dict'] = discuss_dict
        new_data, lowest_score_key, scores = agent_back_process(agent_set, discuss_dict)

    else:
        reason_dict = reason_process(agent_set, anno, answer_dict)
        discuss_dict = discuss_text_process(agent_set, anno, answer_dict, reason_dict)
        new_data, lowest_score_key, scores = agent_back_process(agent_set, discuss_dict)
        history_info = generate_history_info(agent_set, anno, new_data, lowest_score_key, scores, reason_dict, answer_dict)
        anno['first_round'] = {}
        anno['first_round']['answer_dict'] = answer_dict
        anno['first_round']['scores'] = scores
        anno['first_round']['reason_dict'] = reason_dict
        anno['first_round']['discuss_dict'] = discuss_dict
        anno['first_round']['history_info'] = history_info

    new_agent_set = []
    for agent in agent_set:
        if agent.get_model_name() in new_data:
            new_agent_set.append(agent)
    agent_set = new_agent_set
    
    # 第一轮结束，第二轮开始
    answer_set, answer_dict, sample_dict = get_result_second_round(agent_set, anno, history_info)
    anno['second_samp'] = sample_dict
    print(answer_set)
    if len(answer_set) == 1:
        answer_set = next(iter(answer_set))
        answer_set = str(answer_set)
        if answer_set == chr(ord('A') + anno['correct_choice']):
            correct += 1
        print("second round, correct{}, all_num {}, {}".format(correct, all_num, correct / all_num))
        anno['final_answer'] = str(answer_set)
        with open(all_anno_path, 'w') as f:
            json.dump(result_anno, f)
        continue
    
    print(answer_dict, chr(ord('A') + anno['correct_choice']))

    reason_dict = reason_process(agent_set, anno, answer_dict)
    discuss_dict = discuss_text_process(agent_set, anno, answer_dict, reason_dict)
    new_data, lowest_score_key, scores = agent_back_process(agent_set, discuss_dict)
    history_info = generate_history_info(agent_set, anno, new_data, lowest_score_key, scores, reason_dict, answer_dict)
    anno['second_round'] = {}
    anno['second_round']['answer_dict'] = answer_dict
    anno['second_round']['scores'] = scores
    anno['second_round']['reason_dict'] = reason_dict
    anno['second_round']['discuss_dict'] = discuss_dict
    anno['second_round']['history_info'] = history_info
        
    new_agent_set = []
    for agent in agent_set:
        if agent.get_model_name() in new_data:
            new_agent_set.append(agent)
    agent_set = new_agent_set
    # 在这里重新生成info。
    # 第二轮结束
    answer_set, answer_dict, sample_dict = get_result_second_round(agent_set, anno, history_info)
    anno['third_samp'] = sample_dict

    anno['last_round'] = {}   
    anno['last_round']['answer_dict'] = answer_dict
    print(answer_dict)
    final_answer = answer_dict[agent_set[0].get_model_name()]
    print(final_answer)
    anno['final_answer'] = str(final_answer)
    if final_answer == chr(ord('A') + anno['correct_choice']):
        correct += 1
    
    print(answer_dict, chr(ord('A') + anno['correct_choice']))
    print("third round, correct{}, all_num {}, {}".format(correct, all_num, correct / all_num))
    try:
        with open(all_anno_path, 'w') as f:
            json.dump(result_anno, f)
    except:
        print(anno)
        anno = ori_anno[kkk]

with open(all_anno_path, 'w') as f:
    json.dump(result_anno, f)