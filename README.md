# <span style="color:red">(NeurIPS24)</span> NaturalBench: Evaluating Vision-Language Models on Natural Adversarial Samples

## Links:

<div align="center">

| [🏠**Home Page**](https://linzhiqiu.github.io/papers/naturalbench) | [&#129303;**HuggingFace**](https://huggingface.co/datasets/BaiqiL/NaturalBench) | [**🏆Leaderboard**](https://huggingface.co/datasets/BaiqiL/NaturalBench#Leaderboard) | [**📖Paper**](https://arxiv.org/abs/2410.14669) | [🖥️ **Code**](https://github.com/Baiqi-Li/NaturalBench/blob/main/example.py) |

</div>

<div align="center">
  <img src="https://huggingface.co/datasets/BaiqiL/NaturalBench/resolve/main/pictures/natural_teaser.jpg" style="width: 80%; height: auto;">
</div>


## 🚩 **News**
- 🎉 Sep. 26, 2024.  NaturalBench was accepted by ***NeurIPS***!
- ✅ NaturalBench-Retrieval Dataset: the [download link](https://huggingface.co/datasets/BaiqiL/NaturalBench/resolve/main/NaturalBench-Retrieval.zip?download=true) from [huggingface homepage](https://huggingface.co/datasets/BaiqiL/NaturalBench).
- ✅ We have integrated NaturalBench into [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) and [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).
  ```
  # Evaluation with lmms-eval and VLMEvalKit
  ## Please refer to the official documentation of lmms-eval and VLMEvalKit

  ## lmms-eval:
      python3 -m accelerate.commands.launch \
          --num_processes=1 \
          -m lmms_eval \
          --model llava_onevision \
          --model_args pretrained="lmms-lab/llava-onevision-qwen2-7b-ov" \
          --tasks naturalbench \
          --batch_size 1 \
          --log_samples \
          --log_samples_suffix llava_onevision_naturalbench \
          --output_path ./logs/

  ## VLMEvalKit:
      python run.py --data NaturalBenchDataset --model llava-onevision-qwen2-7b-ov-hf --verbose
  ```
  
## Usages
You can learn how to use and evaluate NaturalBench by reviewing the simple examples in [example.py](https://github.com/Baiqi-Li/NaturalBench/blob/main/example.py).

## Citation Information
```
@inproceedings{naturalbench,
  title={NaturalBench: Evaluating Vision-Language Models on Natural Adversarial Samples},
  author={Li, Baiqi and Lin, Zhiqiu and Peng, Wenxuan and Nyandwi, Jean de Dieu and Jiang, Daniel and Ma, Zixian and Khanuja, Simran and Krishna, Ranjay and Neubig, Graham and Ramanan, Deva},
  booktitle={The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2024},
  url={https://openreview.net/forum?id=Dx88A9Zgnv}
}
```
