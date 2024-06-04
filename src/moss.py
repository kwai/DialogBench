import json
from tqdm import tqdm
from baseEval import EvalModel
from utils import extract_choice, evalate_step, calculate_score, GEN_KWARGS

class EvalMoss(EvalModel):
    def _evaluate_SFT(self, data_list, save_result_dir):
        count = 0
        correct_num = 0
        correct_num_hard = 0
        result_dict = {}
        f = open(save_result_dir, "w", encoding="utf8")
        result_list = []
        for data in tqdm(data_list):
            count += 1
            inputs = self.tokenizer([data['prompt_MOSS']], return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                inputs=inputs['input_ids'], 
                attention_mask=inputs['attention_mask'],
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                **GEN_KWARGS)
            model_output = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], clean_up_tokenization_spaces=True, skip_special_tokens=True)
            pred_answer_hard, pred_answer_easy = extract_choice(model_output,language=self.language)
            result_list.append({
                "prompt": data["prompt_MOSS"],
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
        f.close()
