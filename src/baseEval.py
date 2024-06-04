import time
import math
from utils import *
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, GenerationConfig


class EvalModel:
    def __init__(self, model_name, method, cuda_device, language="Chinese"):
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.model_name = model_name
        self.method = method
        self.device = cuda_device
        self.language = language
        self.model_path = get_model_path(self.model_name)
        self.model, self.tokenizer = self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        print("======================== Loading Model ========================")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, use_fast=False)
        if 'chatglm' in self.model_path.lower():
            model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).to(self.device)
        else:
            model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True).to(self.device)
            if self.method == "sft":
                if "qwen" in self.model_path.lower() or (
                        "baichuan" in self.model_path.lower() and "sft" in self.method.lower()):
                    model.generation_config = GenerationConfig.from_pretrained(self.model_path, trust_remote_code=True)
                    model.generation_config.update(**GEN_KWARGS)
        print("====================== Already Load Model ======================")
        return model.eval(), tokenizer

    def evaluate(self, data_list, save_result_dir):
        save_result_dir = save_result_dir + "/" + time.strftime("%m%d_%H%M", time.localtime(time.time())) + ".json"
        if self.method.lower() == "sft":
            return self._evaluate_SFT(data_list, save_result_dir)
        elif self.method.lower() == "pretrain" or self.method.lower() == "pt":
            return self._evaluate_pretrain(data_list, save_result_dir)
        else:
            print("================================ ERROR ================================")
            print("====== Pleaser check your method, make sure it in ('pt', 'sft')) ======")
            print("================================ ERROR ================================")
            return None

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole_enc = self.tokenizer.encode(context + continuation)
        context_enc = self.tokenizer.encode(context)
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    def _evaluate_SFT(self, data_list, save_result_dir):
        count = 0
        correct_num = 0
        correct_num_hard = 0
        result_dict = {}
        f = open(save_result_dir, "w", encoding="utf8")
        result_list = []
        for data in tqdm(data_list):
            count += 1
            inputs = self.tokenizer([data["prompt_SFT"]], return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                **GEN_KWARGS)
            model_output = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                                 clean_up_tokenization_spaces=True, skip_special_tokens=True)
            pred_answer_hard, pred_answer_easy = extract_choice(model_output, language=self.language)
            result_list.append({
                "prompt": data["prompt_SFT"],
                "model_output": model_output,
                "label": data["label"],
                "pred_answer_hard": pred_answer_hard,
                "pred_answer_easy": pred_answer_easy
            })
            result_dict, correct_num, correct_num_hard = evalate_step(result_dict, pred_answer_easy, pred_answer_hard,
                                                                      data["label"], correct_num, correct_num_hard)
        correct_ratio = 100 * correct_num / count
        correct_ratio_hard = 100 * correct_num_hard / count
        result_dict = calculate_score(result_dict)
        result = {'score': correct_ratio, 'socre_hard': correct_ratio_hard, 'detail': result_dict, 'data': result_list}
        json.dump(result, f, ensure_ascii=False, indent=4)
        print("======================== ACC: ", round(correct_ratio, 2), " ========================")
        f.close()

    def _evaluate_pretrain(self, data_list, save_result_dir):
        count = 0
        correct_num = 0

        result_dict = {}
        f = open(save_result_dir, "w", encoding="utf8")
        result_list = []
        for data in tqdm(data_list):
            count += 1
            ppls = []
            with torch.no_grad():
                for index, target in data["choices"].items():
                    context_enc, target_enc = self._encode_pair(data["prompt_pretrain"], target)
                    # when too long to fit in context, truncate from the left
                    input_ids = torch.tensor(
                        (context_enc + target_enc)[-(MAX_LEN + 1):][:-1],
                        dtype=torch.long, ).to(self.device)
                    (input_length,) = input_ids.shape
                    logits = self.model(input_ids.unsqueeze(0))["logits"].cpu()
                    target_length = len(target_enc)
                    cont_tok = torch.tensor(target_enc, dtype=torch.long).unsqueeze(0)
                    logits = logits[:, input_length - target_length: input_length, :]  # [1, seq, vocab]
                    loss = self.cross_entropy(logits.squeeze(0), cont_tok.squeeze(0))
                    ppls.append((math.exp(loss.item()), index))
            pred_answer = sorted(ppls, key=lambda x: x[0])[0][1]
            result_list.append({
                "prompt": data["prompt_pretrain"],
                "choices": data["choices"],
                "ppl_score": ppls,
                "label": data["label"],
                "pred_answer": pred_answer
            })
            result_dict, correct_num, _ = evalate_step(result_dict, pred_answer, None, data["label"], correct_num, 0)
        correct_ratio = 100 * correct_num / count
        result_dict = calculate_score(result_dict)
        result = {"score": correct_ratio, "detail": result_dict, "data": result_list}
        json.dump(result, f, ensure_ascii=False, indent=4)
        print("======================== ACC: ", round(correct_ratio, 2), " ========================")
        f.close()
