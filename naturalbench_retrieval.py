import argparse
import os
import t2v_metrics
import json
from torch.utils.data import Dataset

def get_retrieval_scores(scores_i2t):
    ids = list(range(scores_i2t.shape[0]))
    retrieval_scores = []
    for id, score_i2t in zip(ids, scores_i2t):
        retrieval_scores.append({
            "id" : id,
            "c0_i0": score_i2t[0][0],
            "c0_i1": score_i2t[1][0],
            "c1_i0": score_i2t[0][1],
            "c1_i1": score_i2t[1][1]}
        )
    return retrieval_scores

def get_retrieval_acc(scores):
    text_correct_count = 0
    image_correct_count = 0
    group_correct_count = 0
    def text_correct(result):
        return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]

    def image_correct(result):
        return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]

    def group_correct(result):
        return image_correct(result) and text_correct(result)
    
    for result in scores:
        text_correct_count += 1 if text_correct(result) else 0
        image_correct_count += 1 if image_correct(result) else 0
        group_correct_count += 1 if group_correct(result) else 0

    denominator = len(scores)
    result = {
        'text': text_correct_count/denominator,
        'image': image_correct_count/denominator,
        'group': group_correct_count/denominator,
    }
    return result

class NaturalBench_Retrieval(Dataset):
    def __init__(self,
                 root_dir='./datasets',
                 download=True,
                 image_preprocess=None,
                 return_image_paths=True):
        self.root_dir = root_dir
        self.dataset_dir = os.path.join(root_dir, "NaturalBench-Retrieval")
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.metadata_path = os.path.join(self.dataset_dir, 'metadata.json')

        self.download_links =  "https://huggingface.co/datasets/BaiqiL/NaturalBench/resolve/main/NaturalBench-Retrieval.zip"

        if not os.path.exists(self.dataset_dir):
            if download:
                import subprocess
                model_file_name = "NaturalBench-Retrieval.zip"
                image_zip_file = os.path.join(self.root_dir, model_file_name)
                if not os.path.exists(image_zip_file):
                    subprocess.call(
                        ["wget", self.download_links, "-O", model_file_name], cwd=self.root_dir
                    )
                subprocess.call(["unzip", "-q", model_file_name], cwd=self.root_dir)

        with open(self.metadata_path, 'r', encoding='utf-8') as file:
            self.metadata = json.load(file)

        self.return_image_paths = return_image_paths
        
        if return_image_paths:
            assert image_preprocess is None
            self.preprocess = None
        self.preprocess = image_preprocess
    
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        assert self.metadata[idx]['index'] == idx
        image_0_path = os.path.join(self.image_dir, self.metadata[idx]['image_0'])
        image_1_path = os.path.join(self.image_dir, self.metadata[idx]['image_1'])
        if self.return_image_paths:
            image_0 = image_0_path
            image_1 = image_1_path
        else:
            image_0 = self.preprocess(self.image_loader(image_0_path))
            image_1 = self.preprocess(self.image_loader(image_1_path))
        
        caption_0 = self.metadata[idx]['caption_0']
        caption_1 = self.metadata[idx]['caption_1']
        item = {
            "images": [image_0, image_1],
            "texts": [caption_0, caption_1]
        }
        return item
    
    def evaluate_scores(self, scores):
        retrieval_scores = get_retrieval_scores(scores)
        acc = get_retrieval_acc(retrieval_scores)
        print("NaturalBench-Retrieval performance (overall)")
        print(f"{'Dataset': <70} {'Text': <10} {'Image': <10} {'Group': <10}")
        print(f"{'NaturalBench-Retrieval': <70} {acc['text']: <10.2%} {acc['image']: <10.2%} {acc['group']: <10.2%}")
        results = {}
        results['all'] = acc
        return results

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="./datasets", type=str,
                        help='Root directory for saving datasets.')
    parser.add_argument("--cache_dir", default=t2v_metrics.constants.HF_CACHE_DIR, type=str) 
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--model", default="openai:ViT-L-14", type=str) #VQAScore:"clip-flant5-xxl"
    parser.add_argument("--question", default=None, type=str)
    parser.add_argument("--answer", default=None, type=str)
    return parser.parse_args()

def main():
    args = config()
    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir)
    
    score_func = t2v_metrics.get_score_model(model=args.model, device=args.device, cache_dir=args.cache_dir)

    kwargs = {}
    if args.question is not None:
        print(f"Using question template: {args.question}")
        kwargs['question_template'] = args.question
    if args.answer is not None:
        print(f"Using answer template: {args.answer}")
        kwargs['answer_template'] = args.answer
    
    print(f"Performance of {args.model}.")
        
    dataset = NaturalBench_Retrieval(root_dir=args.root_dir)
    scores = score_func.batch_forward(dataset, batch_size=args.batch_size, **kwargs).cpu()
    dataset.evaluate_scores(scores)

if __name__ == "__main__":
    main()