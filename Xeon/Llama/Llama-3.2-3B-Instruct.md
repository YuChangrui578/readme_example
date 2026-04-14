# Running Llama-3.2-3B-Instruct Series on Xeon® with SGLang

## Model Overview

The Llama-3.2-3B-Instruct series offers multiple precision options to balance accuracy, performance, and memory efficiency. This series includes W8A8 quantization, FP8, and AWQ INT4 variants, each optimized for different deployment scenarios on Intel Xeon® processors.

| Data Type | Model Card | Precision | Use Case |
|:---:|:---|:---:|:---|
| W8A8 | [RedHatAI/Llama-3.2-3B-quantized.w8a8](https://huggingface.co/RedHatAI/Llama-3.2-3B-quantized.w8a8) | 8-bit | Optimized for reduced memory footprint with minimal accuracy loss |
| FP8 | [RedHatAI/Llama-3.2-3B-Instruct-FP8](https://huggingface.co/RedHatAI/Llama-3.2-3B-Instruct-FP8) | 8-bit Floating Point | Balanced precision for general-purpose inference |
| AWQ_INT4 | [AMead10/Llama-3.2-3B-Instruct-AWQ](https://huggingface.co/AMead10/Llama-3.2-3B-Instruct-AWQ) | 4-bit | Maximum compression for memory-constrained environments |

## Model Acquisition

You can access the models on huggingface.

The models can be downloaded to local storage by command

```
hf download --resume <MODEL_ID> --local-dir 'path/to/local/dir'
```

*Note:* You may need to log in your authorized HuggingFace account to access the model files.
Please refer to [HuggingFace login](https://huggingface.co/docs/huggingface_hub/quick-start#login).

## Model Deployment

The BKMs for model service deployment & benchmarking for the supported data types are as follows:

[W8A8](#w8a8) | [FP8](#fp8) | [AWQ_INT4](#awq_int4)

### W8A8

The W8A8 (8-bit weight, 8-bit activation) quantization provides excellent memory efficiency while maintaining high inference quality. This format is ideal for production deployments where memory bandwidth is a bottleneck.

#### Environment Setup

`Llama-3.2-3B-quantized.w8a8` W8A8 model has been supported by official SGLang.
Please refer to the `Installation` section in
[the official SGLang CPU server document](https://docs.sglang.io/platforms/cpu_server.html#installation).

#### Launch of the Serving Engine

An example command to launch SGLang server with `RedHatAI/Llama-3.2-3B-quantized.w8a8` would be like:

```bash
sglang serve                     \
    --model <MODEL_ID_OR_PATH>   \
    --trust-remote-code          \
    --disable-overlap-schedule   \
    --device cpu                 \
    --enable-torch-compile       \
    --host 0.0.0.0               \
    --tp 6
```

The `<MODEL_ID_OR_PATH>` can be either the model ID (a.k.a. `RedHatAI/Llama-3.2-3B-quantized.w8a8`) or the path of the pre-downloaded model folder.

Please read the `Notes` part in the serving engine launching section in
[the official SGLang CPU server document](https://docs.sglang.io/platforms/cpu_server.html#launch-of-the-serving-engine)
to better understand how to configure the arguments, especially for TP (tensor parallel) and numa binding settings.

### FP8

The FP8 (8-bit floating point) precision offers a balanced approach between accuracy and performance. This format is suitable for general-purpose inference workloads where numerical stability is crucial.

#### Environment Setup

`Llama-3.2-3B-Instruct-FP8` FP8 model has been supported by official SGLang.
Please refer to the `Installation` section in
[the official SGLang CPU server document](https://docs.sglang.io/platforms/cpu_server.html#installation).

#### Launch of the Serving Engine

An example command to launch SGLang server with `RedHatAI/Llama-3.2-3B-Instruct-FP8` would be like:

```bash
sglang serve                     \
    --model <MODEL_ID_OR_PATH>   \
    --trust-remote-code          \
    --disable-overlap-schedule   \
    --device cpu                 \
    --enable-torch-compile       \
    --host 0.0.0.0               \
    --tp 6
```

The `<MODEL_ID_OR_PATH>` can be either the model ID (a.k.a. `RedHatAI/Llama-3.2-3B-Instruct-FP8`) or the path of the pre-downloaded model folder.

Please read the `Notes` part in the serving engine launching section in
[the official SGLang CPU server document](https://docs.sglang.io/platforms/cpu_server.html#launch-of-the-serving-engine)
to better understand how to configure the arguments, especially for TP (tensor parallel) and numa binding settings.

### AWQ_INT4

The AWQ INT4 (Activation-aware Weight Quantization with 4-bit) provides the highest compression ratio, making it ideal for memory-constrained environments. This format significantly reduces memory usage while preserving most of the model's capabilities.

#### Environment Setup

`Llama-3.2-3B-Instruct-AWQ` AWQ INT4 model is supported in [a dev branch](https://github.com/jianan-gu/sglang/tree/cpu_optimized).

You can pull the docker image if you have access to `gar-registry.caas.intel.com`:

```bash
docker pull gar-registry.caas.intel.com/pytorch/pytorch-ipex-spr:intel-sglang-cpu-optimized
```

Or you can build the image from the Dockerfile:

```bash
git clone -b cpu_optimized https://github.com/jianan-gu/sglang.git
cd sglang/docker
# May need to add some other (e.g. proxy) settings
docker build -t sglang:intel-cpu-optimized -f xeon.Dockerfile .
```

#### Launch of the Serving Engine

An example command to launch SGLang server with `AMead10/Llama-3.2-3B-Instruct-AWQ` would be like:

```bash
python -m sglang.launch_server   \
    --model <MODEL_ID_OR_PATH>   \
    --trust-remote-code          \
    --disable-overlap-schedule   \
    --device cpu                 \
    --enable-torch-compile       \
    --host 0.0.0.0               \
    --tp 6
```

The `<MODEL_ID_OR_PATH>` can be either the model ID (a.k.a. `AMead10/Llama-3.2-3B-Instruct-AWQ`) or the path of the pre-downloaded model folder.

Please read the `Notes` part in the serving engine launching section in
[the official SGLang CPU server document](https://docs.sglang.io/platforms/cpu_server.html#launch-of-the-serving-engine)
to better understand how to configure the arguments, especially for TP (tensor parallel) and numa binding settings.

## Benchmarking

Open another terminal and run the `sglang.bench_serving` command.
An example command would be like:

```bash
python -m sglang.bench_serving                                 \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json   \
    --dataset-name random                                      \
    --random-input-len 1024                                    \
    --random-output-len 1024                                   \
    --num-prompts 1                                            \
    --max-concurrency 1                                        \
    --request-rate inf                                         \
    --random-range-ratio 1.0
```

In the example command

- `--request_rate inf` indicates that all requests should be sent simultaneously.
- `--num-prompts 1` and `--max-concurrency 1` indicates 1 request is sent in this test round, can be adjusted for testing with different request concurrency number.
- `--dataset-name random` is set to randomly select samples from the dataset.
- `--random-input 1024`, `--random-output 1024` and `--random-range-ratio 1.0` settings are for fixed 1024-in/1024-out token size limit (realized by truncating or repeating the original sample).
 
Please adjust the settings per your benchmarking scenarios. Detailed descriptions for the arguments of `bench_serving` are available via the command:

```bash
python -m sglang.bench_serving -h
```

## Precision Comparison Summary

| Data Type | Memory Efficiency | Accuracy | Deployment Complexity | Recommended Scenario |
|:---:|:---:|:---:|:---:|:---|
| W8A8 | High | Excellent | Low (Official SGLang) | Production with memory constraints |
| FP8 | High | Excellent | Low (Official SGLang) | General-purpose inference |
| AWQ_INT4 | Highest | Very Good | Medium (Dev branch) | Maximum compression, edge devices |