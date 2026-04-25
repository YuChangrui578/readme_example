# Llama 3.2

## 1. Model Introduction

Llama 3.2 is a collection of lightweight models released by Meta, designed to bring advanced reasoning and instruction-following capabilities to edge devices and smaller-scale deployments. This family includes optimized variants such as quantized W8A8, FP8, and AWQ versions to maximize performance on CPU-based architectures.

Key features include:
- **Optimized for Efficiency**: Small-scale models (e.g., 3B) tailored for low-latency applications.
- **Quantization Support**: Multiple quantization formats (W8A8, FP8, AWQ) available for varying hardware constraints.
- **Enhanced Reasoning**: Improved instruction following and general intelligence compared to previous lightweight iterations.

For further details, please refer to the [Llama 3.2 model card on HuggingFace](https://huggingface.co/meta-llama).

## 2. Model Acquisition

You can access the models on huggingface.

| Data Type | Model ID |
|:---:|:---:|
| W8A8 | `Llama-3.2-3B-quantized.w8a8` |
| FP8 | `Llama-3.2-3B-Instruct-FP8` |
| AWQ | `Llama-3.2-3B-Instruct-AWQ` |

The models can be downloaded to local storage by command:

```bash
hf download --resume <MODEL_ID> --local-dir 'path/to/local/dir'
```

Then set `--model-path <LOCAL_MODEL_PATH>` instead of `--model-path <MODEL_ID>` in the `sglang.launch_server` command.

## 3. SGLang Installation

For optimized CPU deployment, please refer to the [official SGLang installation guide](https://docs.sglang.io/platforms/cpu_server.html#installation).

For specific optimizations like AWQ, please refer to the [CPU-optimized branch of SGLang](https://github.com/jianan-gu/sglang/tree/cpu_optimized).

You can pull the docker image if you have access to `gar-registry.caas.intel.com`:

```bash
docker pull gar-registry.caas.intel.com/pytorch/pytorch-ipex-spr:intel-sglang-cpu-optimized
```

## 4. Model Deployment

This section provides deployment configurations optimized for the hardware platforms and use cases.

**Interactive Command Generator**: Use the configuration selector below to generate a launch command for the Llama 3.2 collection of models.

import Llama32ConfigGenerator from '@site/src/components/autoregressive/Llama32ConfigGenerator';

<Llama32ConfigGenerator />

Please read the `Notes` part in the serving engine launching section in [the official SGLang CPU server document](https://docs.sglang.io/platforms/cpu_server.html#launch-of-the-serving-engine) to better understand how to configure the arguments, especially for TP (tensor parallel) and NUMA binding settings.

## 5. Model Invocation

SGLang exposes an OpenAI-compatible endpoint. First, start the server:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="EMPTY",
)

resp = client.chat.completions.create(
    model="Llama-3.2-3B-Instruct-FP8",
    messages=[
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a Python function that retries a request with exponential backoff."},
    ],
    temperature=0.2,
    max_tokens=512,
)

print(resp.choices[0].message.content)
```

## 6. Benchmarking

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

Detailed descriptions for the arguments of `bench_serving` are available in [this doc](../../base/benchmarks/lm_benchmark.md).

---