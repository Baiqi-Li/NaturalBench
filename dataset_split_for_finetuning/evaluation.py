import re
import os
import zipfile
import json

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


def ensure_images_directory():
    # Get the current script's directory
    current_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Define the path to the images folder
    images_folder_path = os.path.join(current_directory, 'images')
    
    # Create the images folder if it doesn't exist
    if not os.path.exists(images_folder_path):
        print("The images folder does not exist. Creating it...")
        os.makedirs(images_folder_path)
    
    # Define the path to the images.zip file
    zip_file_path = os.path.join(current_directory, 'images.zip')
    
    # Check if the images.zip file exists
    if os.path.exists(zip_file_path):
        print("Extracting images.zip into the images folder...")
        # Extract the images.zip file into the images folder
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(images_folder_path)
        print("Extraction completed.")
    else:
        print("The images.zip file does not exist. Cannot extract.")


def read_json_file(file_path):
    """
    Read and return the content of a JSON file.

    :param file_path: Path to the JSON file.
    :return: Parsed JSON content as a Python dictionary.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the file {file_path}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")



if __name__ == "__main__":

    ensure_images_directory()

    current_directory = os.path.dirname(os.path.abspath(__file__))
    finetune_test_set = read_json_file(os.path.join(current_directory, 'test_set.json'))

    # # 1. Test Models: use the finetune_test_set to test your own models and get the "model output"(string) into "output_file"(list)
    
    # # 2. Extract the answer: extract the answer from the outputs (you could also use LLMs such as ChatGPT to extract the answer)
    assert len(output_file) == len(finetune_test_set)
    answers = {}
    number_answered_samples = len(output_file)//4
    for i in range(number_answered_samples):
        answers[i] = {
            "q0_i0": extract_answer(output_file[i*4], finetune_test_set[i*4][3]),
            "q0_i1": extract_answer(output_file[i*4+1], finetune_test_set[i*4+1][3]),
            "q1_i0": extract_answer(output_file[i*4+2], finetune_test_set[i*4+2][3]),
            "q1_i1": extract_answer(output_file[i*4+3], finetune_test_set[i*4+3][3])
        }

    scores = get_scores(answers)
    print(scores)
