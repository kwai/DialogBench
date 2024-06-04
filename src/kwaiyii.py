import json
from utils import *
from tqdm import tqdm
from baseEval import EvalModel


class EvalKwaiYii(EvalModel):
    def _evaluate_SFT(self, data_list, save_result_dir):
        count = 0
        correct_num = 0
        correct_num_hard = 0
        result_dict = {}
        f = open(save_result_dir, "w", encoding="utf8")
        result_list = []
        for data in tqdm(data_list):
            count += 1
            query = data["query"]
            history = []
            for ls in data["history"]:
                history.append("user####" + ls[0])
                history.append("ASSISTANT####" + ls[1])
            model_output = self.model(query=query, history=history)
            pred_answer_hard, pred_answer_easy = extract_choice(model_output,language=self.language)
            result_list.append({
                "prompt": data["prompt_KuaiYi"],
                "choices": data["choices"],
                "label": data["label"],
                "model_output": model_output,
                "pred_answer_hard": pred_answer_hard,
                "pred_answer_easy": pred_answer_easy
            })
            result_dict, correct_num, correct_num_hard = evalate_step(result_dict, pred_answer_easy, pred_answer_hard, data["label"], correct_num, correct_num_hard)
        correct_ratio = 100 * correct_num / count
        correct_ratio_hard = 100 * correct_num_hard / count
        result_dict = calculate_score(result_dict)
        result = {'score': correct_ratio, 'socre_hard': correct_ratio_hard, 'detail': result_dict, 'data': result_list}
        json.dump(result, f, ensure_ascii=False, indent=4)
        print("======================== ACC: ", round(correct_ratio, 2), " ========================")
        self.model.close()
        f.close()
