import json
import copy
from tqdm import tqdm
from baseEval import EvalModel
from utils import extract_choice, evalate_step, calculate_score, GEN_KWARGS, shuffle_choices


class EvalQwen(EvalModel):
    def _evaluate_SFT(self, data_list, save_result_dir):
        count = 0
        correct_num = 0
        correct_num_hard = 0
        result_dict = {}
        f = open(save_result_dir, "w", encoding="utf8")
        result_list = []
        for data in tqdm(data_list):
            try:
                option_str = data['query'].split("【候选选项】")[-1]
                option_str = option_str.strip()
            except:
                continue
            for index in range(len(data["choices"])):
                count += 1
                new_option_str, label, choices = shuffle_choices(data["label"], data['choices'], index=index, shuffle=True)
                query = data["query"].replace(option_str, new_option_str)
                history = copy.deepcopy(data["history"])
                model_output, history = self.model.chat(self.tokenizer, query, history=history)
                pred_answer_hard, pred_answer_easy = extract_choice(model_output,language=self.language)
                result_list.append({
                    "id": data["id"] + "###" + str(index),
                    "domain": data["domain"],
                    "query": query,
                    "history": history,
                    "choices": choices,
                    "model_output": model_output,
                    "label": label,
                    "pred_answer_hard": pred_answer_hard,
                    "pred_answer_easy": pred_answer_easy
                })
                result_dict, correct_num, correct_num_hard = evalate_step(result_dict, pred_answer_easy, pred_answer_hard, label, correct_num, correct_num_hard)
        correct_ratio = 100 * correct_num / count
        correct_ratio_hard = 100 * correct_num_hard / count
        result_dict = calculate_score(result_dict)
        result = {'score': correct_ratio, 'socre_hard': correct_ratio_hard, 'detail': result_dict, 'data': result_list}
        json.dump(result, f, ensure_ascii=False, indent=4)
        print("======================== ACC: ", round(correct_ratio, 2), " ========================")
        f.close()
