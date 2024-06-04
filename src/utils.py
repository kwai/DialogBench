import re
import json
import torch
import random
from transformers.generation.logits_process import LogitsProcessor


MAX_LEN = 2048
NUM_BEAMS = 1
DO_SAMPLE = False
MAX_NEW_TOKENS = 1
MAX_NEW_TOKENS_SFT = 128
REPETITION_PENALTY = 1.0
SYSTEM_STR = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions. "
GEN_KWARGS = {
    "num_beams": NUM_BEAMS,
    "do_sample": DO_SAMPLE,
    "max_new_tokens": MAX_NEW_TOKENS_SFT,
    "repetition_penalty": REPETITION_PENALTY}


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


def extract_choice(response, language="Chinese"):
    if "en" in language.lower():
        return extract_choice_en(response)
    choices = ["A", "B", "C", "D"]
    response = str(response).strip()
    if len(response) == 0:
        return None, None
    if response[0] in choices:
        return response[0], response[0]
    # 1. Single match
    patterns = [
        (r'答案(选项)?(是|为)：? ?([ABCD])', 3),
        (r'答案(是|为)选项 ?([ABCD])', 2),
        (r'故?选择?：? ?([ABCD])', 1),
        (r'([ABCD]) ?选?项(是|为)?正确', 1),
        (r'正确的?选项(是|为) ?([ABCD])', 2),
        (r'答案(应该)?(是|为)([ABCD])', 3),
        (r'选项 ?([ABCD]) ?(是|为)?正确', 1),
        (r'选择答案 ?([ABCD])', 1),
        (r'答案?：?([ABCD])', 1),
        (r'([ABCD])(选?项)?是?符合题意', 1),
        (r'答案选项：? ?([ABCD])', 1),  # chatglm
        (r'答案(选项)?为(.*?)([ABCD])', 3),  # chatgpt
    ]
    for pattern, idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            answer = m.group(idx)
            assert answer in choices
            return None, answer
    # 2. Recursive match
    patterns = [
        (r'([ABCD])(.*?)当选', 1),
        (r'([ABCD])(.*?)正确', 1),
    ]
    for pattern, idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            while m:
                answer = m.group(idx)
                m = re.search(pattern, m.group(0)[1:], re.M)
            assert answer in choices
            return None, answer
    # 3. Weak single match
    patterns = [
        (r'[^不]是：? ?([ABCD])', 1),
    ]
    for pattern, idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            answer = m.group(idx)
            assert answer in choices
            return None, answer
    # 4. Check the only mentioend choices
    pattern = r'^[^ABCD]*([ABCD])[^ABCD]*$'
    m = re.match(pattern, response)
    if m:
        answer = m.group(1)
        assert answer in choices
        return None, answer
    return None, None


def get_model_path(model_name):
    with open('./config/model_path_config.json', 'r', encoding='utf8') as f:
        path_dict = json.load(f)
    f.close()
    model_path = path_dict.get(model_name, model_name)
    return model_path


def get_choice_content(item_v):
    for key in ["answer", "text", "content", "person", "person1", "person2"]:
        content = item_v.get(key, None)
        if content is not None:
            return content
    return None


def get_option_and_label(option, label, index=-1, shuffle=False):
    choices = {}
    try:
        flag = 0
        if type(option) == dict:
            for item_k, item_v in option.items():
                if str.isnumeric(item_k):
                    item_k = chr(int(item_k) + 64)
                    flag = 1
                choices[item_k] = item_v if type(item_v) != dict else get_choice_content(item_v)
        elif type(option) == list:
            for opt in option:
                for item_k, item_v in opt.items():
                    if str.isnumeric(item_k):
                        item_k = chr(int(item_k) + 64)
                    choices[item_k] = item_v
        if flag:
            try:
                label = ','.join([chr(int(l) + 64) for l in label.split(',')])
            except:
                pass
        return shuffle_choices(label, choices, index, shuffle)
    except:
        return None, None, {}


def shuffle_choices(label, choices, index=-1, shuffle=False):
    """
    if index == -1, random shuffle choices
    elif index in [0, 1, 2, 3], the correct answer will appear at the index-th position.
    """
    if len(choices) == 0 or len(choices) > 4:
        return None, None, {}
    option_str = ""
    new_choices = {}
    label_str = choices[label]
    dict_key_ls = list(choices.keys())
    dict_value_ls = list(choices.values())
    if shuffle:
        if index == -1 or index >= len(dict_key_ls):
            random.shuffle(dict_value_ls)
        else:
            current_position = dict_key_ls.index(label)
            correct_position = index
            dict_value_ls[current_position], dict_value_ls[correct_position] = dict_value_ls[correct_position], \
            dict_value_ls[current_position]
            label = dict_key_ls[index]
        for i in range(len(dict_key_ls)):
            new_choices[dict_key_ls[i]] = dict_value_ls[i]
            if dict_value_ls[i] is None or dict_value_ls[i] in ["None", 'none']:
                return None, None, {}
            if label_str == dict_value_ls[i]:
                label = dict_key_ls[i]
            option_str += '%s. %s ' % (dict_key_ls[i], dict_value_ls[i])
        return option_str.strip(), label, new_choices
    else:
        for i in range(len(dict_key_ls)):
            if dict_value_ls[i] is None or dict_value_ls[i] in ["None", 'none']:
                return None, None, {}
            option_str += '%s. %s ' % (list(choices.keys())[i], list(choices.values())[i])
        return option_str.strip(), label, choices


def evalate_step(result_dict, pred_answer, pred_answer_hard, label, correct_num, correct_num_hard):
    if pred_answer not in result_dict:
        result_dict[pred_answer] = {'label_size': 0, 'pred_size': 0, 'score': 0}
    if label not in result_dict:
        result_dict[label] = {'label_size': 0, 'pred_size': 0, 'score': 0}
    result_dict[label]['label_size'] += 1
    result_dict[pred_answer]['pred_size'] += 1
    if pred_answer == label:
        correct_num += 1
        result_dict[pred_answer]['score'] += 1
    if pred_answer_hard == label:
        correct_num_hard += 1
    return result_dict, correct_num, correct_num_hard


def calculate_score(result_dict):
    for key, value in result_dict.items():
        result_dict[key]['recall'] = round(value['score'] / (value['label_size'] + 1e-8), 4) * 100
        result_dict[key]['percision'] = round(value['score'] / (value['pred_size'] + 1e-8), 4) * 100
        f1_score = 2 * result_dict[key]['recall'] * result_dict[key]['percision'] / (
                result_dict[key]['recall'] + result_dict[key]['percision'] + 1e-8)
        result_dict[key]['f1_score'] = f1_score
    return result_dict


def read_json(data_path):
    with open(data_path, "r", encoding="utf8") as f:
        lines = f.readlines()
    f.close()
    return [json.loads(line) for line in lines]


def extract_choice_en(response):
    choices = ["A", "B", "C", "D"]
    response = str(response)
    pattern = r'^\s*[ABCD](?![a-zA-Z])'
    m = re.search(pattern, response)
    if m != None:
        return m.group(0).strip(), m.group(0).strip()

    # 1. Single match
    patterns = [
        (r'is option ([ABCD])', 1),
        (r'answer: ?([ABCD])', 1),
        (r'is:\n\n?([ABCD])', 1),
        (r'is ?([ABCD])', 1),
        (r'is ([ABCD])', 1),
        (r'so, choose: ?([ABCD])', 1),
        (r'choose ([ABCD])', 1),
        (r'([ABCD])\.', 1),
        (r'would be ([ABCD])', 1),
        (r'be ([ABCD])', 1),
        (r'option ([ABCD])', 1),
        (r'Option ([ABCD])', 1),
        (r'Option ([ABCD])', 1),
        (r'【Candidate options】([ABCD])', 1),
        (r'\[([ABCD])\]', 1),
        (r'\(([ABCD])\)', 1),
        (r'\{([ABCD])\}', 1),
        (r'is ?\(([ABCD])\)', 1),
        (r'is \(([ABCD])\)', 1)
    ]
    for pattern, idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            answer = m.group(idx)
            assert answer in choices
            return None, answer
    return None, None
