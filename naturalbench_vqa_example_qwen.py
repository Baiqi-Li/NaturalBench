import torch
import re
from tqdm import tqdm
import json
import os

from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info #pip install qwen-vl-utils[decord];  Refer to https://github.com/QwenLM/Qwen2.5-VL


SUFFIX_FOR_VQA = {
    "yes_no": "Please answer Yes or No.",
    "multiple_choice": "Please output the letter corresponding to the correct option."
} 

def extract_answer(output_string, task_type="yes_no"):
    """
    Extracts the answer from the output string based on the task type.

    Parameters:
    output_string (str): The output string.
    task_type (str): The type of task. Must be either "yes_no" or "multiple_choice".

    Returns:
    int: 
        1 if "yes" or "A" 
        0 if "no" or "B"
        -1 if no relevant answer is found.
        Raises a ValueError if an unsupported task_type is provided.
    """

    def find_word_position(string, word):
        pattern = r'\b' + re.escape(word) + r'\b'
        match = re.search(pattern, string, re.IGNORECASE)
        if match:
            return match.start()
        return -1
    
    if task_type not in ["yes_no", "multiple_choice"]:
        raise ValueError("Task type not supported. Must be 'yes_no' or 'multiple_choice'.")
    
    if task_type == "yes_no":
        position_yes_and_a = find_word_position(output_string, "yes")
        position_no_and_b = find_word_position(output_string, "no")
    elif task_type == "multiple_choice":
        position_yes_and_a = find_word_position(output_string, "A")
        position_no_and_b = find_word_position(output_string, "B")

    if position_yes_and_a == -1 and position_no_and_b == -1:
        print(f"No answer found in the output string: {output_string}.")
        return -1
    elif position_yes_and_a != -1 and position_no_and_b != -1:
        return 1 if position_yes_and_a < position_no_and_b else 0
    else:
        return 0 if position_yes_and_a == -1 else 1

def get_scores(scores):
    """
    Calculate various scores based on the given results.

    Args:
        scores (dict or list): A dictionary or list containing results where each result can be:
            - dict: {id: {"q0_i0": 1 or 0, "q0_i1": 1 or 0, "q1_i0": 1 or 0, "q1_i1": 1 or 0}, ...}
            - list: [[q0_i0 (1 or 0), q0_i1 (1 or 0), q1_i0 (1 or 0), q1_i1 (1 or 0)], ...]

    The keys "q0_i0", "q0_i1", "q1_i0", "q1_i1" represent combinations of questions and images:
        - "q0_i0" means question_0 on image_0 
        - "q0_i1" means question_0 on image_1 
        - "q1_i0" means question_1 on image_0 
        - "q1_i1" means question_1 on image_1 

    Returns:
        dict: A dictionary containing the calculated scores:
            - 'Q_Acc': Average question score
            - 'I_Acc': Average image score
            - 'Acc': Average binary VQA score
            - 'G_Acc': Average group score
    """
    Q_Acc = 0.0
    I_Acc = 0.0
    Acc = 0.0
    G_Acc = 0.0

    num_samples = len(scores)

    def calculate_image_score(result):
        image_correct = 0
        if isinstance(result, dict):
            if result["q0_i0"] == 1.0 and result["q1_i0"] == 0.0:
                image_correct += 1
            if result["q1_i1"] == 1.0 and result["q0_i1"] == 0.0:
                image_correct += 1
        elif isinstance(result, list):
            if result[0] == 1.0 and result[2] == 0.0:
                image_correct += 1
            if result[3] == 1.0 and result[1] == 0.0:
                image_correct += 1
        return image_correct
    
    def calculate_question_score(result):
        text_correct = 0
        if isinstance(result, dict):
            if result["q0_i0"] == 1.0 and result["q0_i1"] == 0.0:
                text_correct += 1
            if result["q1_i1"] == 1.0 and result["q1_i0"] == 0.0:
                text_correct += 1
        else:
            if result[0] == 1.0 and result[1] == 0.0:
                text_correct += 1
            if result[3] == 1.0 and result[2] == 0.0:
                text_correct += 1
        return text_correct

    def calculate_binary_score(result):
        binary_score_correct = 0
        if isinstance(result, dict):
            binary_score_correct += 1 if result["q0_i0"] == 1.0 else 0
            binary_score_correct += 1 if result["q0_i1"] == 0.0 else 0
            binary_score_correct += 1 if result["q1_i0"] == 0.0 else 0
            binary_score_correct += 1 if result["q1_i1"] == 1.0 else 0
        else:
            binary_score_correct += 1 if result[0] == 1.0 else 0
            binary_score_correct += 1 if result[1] == 0.0 else 0
            binary_score_correct += 1 if result[2] == 0.0 else 0
            binary_score_correct += 1 if result[3] == 1.0 else 0

        return binary_score_correct

    def calculate_group(result):
        group_correct = 0
        if calculate_question_score(result) == 2 and calculate_image_score(result) == 2:
            group_correct += 1
        
        return group_correct
    
    if isinstance(scores, dict):
        for _, result in scores.items():
            Q_Acc += calculate_question_score(result)
            I_Acc += calculate_image_score(result)
            Acc += calculate_binary_score(result)
            G_Acc += calculate_group(result)
    else:
        for result in scores:
            Q_Acc += calculate_question_score(result)
            I_Acc += calculate_image_score(result)
            Acc += calculate_binary_score(result)
            G_Acc += calculate_group(result)

    results = {
        'Q_Acc': Q_Acc / float(num_samples * 2),
        'I_Acc': I_Acc / float(num_samples * 2),
        'Acc': Acc / float(num_samples * 4),
        'G_Acc': G_Acc / num_samples
    }

    return results

if __name__ == "__main__":
    #1.Load dataset from HuggingFace
    dataset = load_dataset("BaiqiL/NaturalBench")

    #2.Use NaturalBench: construct 1900*4 [question, image, correct_answer, question_type] samples from the dataset with 1900 samples
    naturalbench = []
    for item in dataset["train"]:
        naturalbench.append([item["Question_0"] + SUFFIX_FOR_VQA[item["Question_Type"]], item["Image_0"], item["Image_0_Question_0"], item['Question_Type']])
        naturalbench.append([item["Question_0"] + SUFFIX_FOR_VQA[item["Question_Type"]], item["Image_1"], item["Image_1_Question_0"], item['Question_Type']])
        naturalbench.append([item["Question_1"] + SUFFIX_FOR_VQA[item["Question_Type"]], item["Image_0"], item["Image_0_Question_1"], item['Question_Type']])
        naturalbench.append([item["Question_1"] + SUFFIX_FOR_VQA[item["Question_Type"]], item["Image_1"], item["Image_1_Question_1"], item['Question_Type']])
    
    # 3. Test Models: use the naturalbench dataset to test your own models and get the "output_file" of your model
    ## 3.1 Load the model
    model_name = "Qwen/Qwen2.5-VL-72B-Instruct"#"Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2", 
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(model_name, min_pixels=256*28*28, max_pixels=1280*28*28)
 
    ## 3.2 Run the model on NaturalBench
    output_file_path = "./Qwen_2_5_VL_72B.json"

    if os.path.exists(output_file_path):
        with open(output_file_path, 'r', encoding='utf-8') as file:
            output_file = json.load(file)
    else:
        output_file = []
        for i in tqdm(range(len(naturalbench))):
            item = naturalbench[i]
            question = item[0]
            image = item[1]
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image,
                            },
                            {"type": "text", "text": question},
                        ],
                    }
                ]

                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(model.device)

                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=128)

                generated_ids_trimmed = [
                    out_ids[len(in_ids) :].cpu() for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

                output_file.append(output_text[0] if isinstance(output_text, list) else output_text)

                del inputs, generated_ids
                torch.cuda.empty_cache()

            except Exception as e:
                # In case of an error, append an empty string and continue
                print(f"Error processing item {i}: {e}")
                output_file.append("")
                continue
 
        with open(output_file_path, 'w') as json_file:
            json.dump(output_file, json_file, indent=4)

    # 4. Extract the answer: extract the answer from the outputs (you could also use LLMs such as ChatGPT to extract the answer)
    assert len(output_file) == 1900*4
    answers = {}
    number_answered_samples = len(output_file)//4
    for i in range(number_answered_samples):
        answers[i] = {
            "q0_i0": extract_answer(output_file[i*4], naturalbench[i*4][3]),
            "q0_i1": extract_answer(output_file[i*4+1], naturalbench[i*4+1][3]),
            "q1_i0": extract_answer(output_file[i*4+2], naturalbench[i*4+2][3]),
            "q1_i1": extract_answer(output_file[i*4+3], naturalbench[i*4+3][3])
        }

    #5. Calculate the scores
    scores = get_scores(answers)
    print(scores)