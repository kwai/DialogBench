import os
import torch
import argparse
from models import *
from utils import read_json


def main(model_name, data_path, data_dir, output_path, method, cuda_device, language="Chinese"):
    """
        :param model_name: if not in model_path_config, will read model_name directly
        :param data_path: should be json or None
        :param data_dir: should be a directory containing json files, or None
        :param method: sft or pretrain
        :param cuda_device: your device
        :param language: for English, just pass "en"
    """

    cuda_device = torch.device("cuda" + cuda_device)

    if not os.path.exists(output_path + model_name):
        os.mkdir(output_path + model_name)
    # load evaluator
    if method == "pretrain":
        evaluator = EvalModel(model_name=model_name, method=method, cuda_device=cuda_device, language=language)
    elif "kwaiyii" in model_name.lower():
        evaluator = EvalKwaiYii(model_name=model_name, method=method, cuda_device=cuda_device, language=language)
    elif "chatglm" in model_name.lower():
        evaluator = EvalchatGLM(model_name=model_name, method=method, cuda_device=cuda_device, language=language)
    elif "baichuan" in model_name.lower():
        evaluator = EvalBaichuan(model_name=model_name, method=method, cuda_device=cuda_device, language=language)
    elif "qwen" in model_name.lower():
        evaluator = EvalQwen(model_name=model_name, method=method, cuda_device=cuda_device, language=language)
    elif "moss" in model_name.lower():
        evaluator = EvalMoss(model_name=model_name, method=method, cuda_device=cuda_device, language=language)
    elif "internlm" in model_name.lower():
        evaluator = EvalInternlm(model_name=model_name, method=method, cuda_device=cuda_device, language=language)
    elif "llama" in model_name.lower():
        evaluator = EvalLlama(model_name=model_name, method=method, cuda_device=cuda_device, language=language)
    else:
        evaluator = EvalModel(model_name=model_name, method=method, cuda_device=cuda_device, language=language)
    # get path list
    path_list = []
    if data_path is not None and data_dir is None:
        path_list = [data_path]
    elif data_path is None and data_dir is not None:
        path_list = [os.path.join(data_dir, f'{data_path}') for data_path in os.listdir(data_dir)]
    # run
    for data_path in path_list:
        print(data_path)
        try:
            data_list = read_json(data_path)
        except:
            continue
        task = data_path.split("/")[-1].replace(".json", "")
        save_result_dir = os.path.join(output_path + model_name, f"{task}_{method}")
        if not os.path.exists(save_result_dir):
            os.mkdir(save_result_dir)
        evaluator.evaluate(data_list, save_result_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="./output")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen-7B-Chat")
    parser.add_argument("--method", type=str, default="sft")
    parser.add_argument("--cuda_device", type=str, default="0")
    parser.add_argument("--language", type=str, default="Chinese")

    args = parser.parse_args()
    main(args.model_name, args.data_path, args.data_dir, args.output_path, args.method, args.cuda_device, args.language)
