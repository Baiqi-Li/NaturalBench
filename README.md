# <span style="color:red">(NeurIPS24)</span> NaturalBench: Evaluating Vision-Language Models on Natural Adversarial Samples

## Links:

<div align="center">

| [🏠**Home Page**](https://linzhiqiu.github.io/papers/naturalbench) | [&#129303;**HuggingFace**](https://huggingface.co/datasets/BaiqiL/NaturalBench) | [**🏆Leaderboard**](https://huggingface.co/datasets/BaiqiL/NaturalBench#Leaderboard) | [**📖Paper**](https://arxiv.org/abs/2410.14669) | [🖥️ **Code**](https://github.com/Baiqi-Li/NaturalBench/blob/main/example.py) |

</div>

<div align="center">
  <img src="https://huggingface.co/datasets/BaiqiL/NaturalBench/resolve/main/pictures/natural_teaser.jpg" style="width: 80%; height: auto;">
</div>


## 🚩 News
- 🎉 Sep. 26, 2024.  NaturalBench was accepted by ***NeurIPS***!
- ✅ We have integrated NaturalBench into [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) and [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).
- ✅ We have integrated NaturalBench-Retrieval Dataset into [t2v_metric](https://github.com/linzhiqiu/t2v_metrics/tree/main).
- ✅ NaturalBench-Retrieval Dataset: the [download link](https://huggingface.co/datasets/BaiqiL/NaturalBench/resolve/main/NaturalBench-Retrieval.zip?download=true) from [huggingface homepage](https://huggingface.co/datasets/BaiqiL/NaturalBench).

## Usages

- ### VQA Task

  There are two approaches to use and evaluate NaturalBench benchmark:

    #### 1. Evaluation based on the example code: 

  Learn how to use and evaluate NaturalBench by reviewing the simple example in [naturalbench_vqa.py](https://github.com/Baiqi-Li/NaturalBench/blob/main/naturalbench_vqa.py).


    #### 2. Evaluation with `lmms-eval` and `VLMEvalKit`:

    Please refer to the official documentation of `lmms-eval` and `VLMEvalKit` for more details.

    - **lmms-eval**:
      ```bash
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
      ```

  - **VLMEvalKit**:
    ```bash
    python run.py --data NaturalBenchDataset --model llava-onevision-qwen2-7b-ov-hf --verbose
    ```

- ### Retrieval Task

  To use the retrieval task, install [t2v_metric](https://github.com/linzhiqiu/t2v_metrics/tree/main) package, then run the evaluation code:

  ```
  python naturalbench_retrieval.py
  ```


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
