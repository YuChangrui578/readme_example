# Llama 3.2

## 1. Model Introduction

Llama 3.2 is a collection of lightweight, highly capable models from Meta, designed to bring advanced intelligence to edge devices and smaller scale deployments. This family includes optimized variants such as quantized W8A8, FP8, and AWQ versions to maximize throughput on CPU-based inference engines.

Key features of the Llama 3.2 family:

- **Efficiency**: Optimized for low-latency inference on diverse hardware.
- **Quantization Support**: Wide range of quantization formats including W8A8, FP8, and AWQ to balance precision and performance.
- **Versatility**: Suitable for a variety of text-based tasks and instruction following.

For further details, please refer to the official [Llama 3.2 release information](https://ai.meta.com/blog/meta-llama-3-2/).

## 2. Model Acquisition

You can access the models on Hugging Face.

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

For CPU-optimized inference, please refer to the [official SGLang installation guide](https://docs.sglang.io/platforms/cpu_server.html#installation).

For AWQ-based models, ensure you are using the optimized branch:

```bash
git clone -b cpu_optimized https://github.com/jianan-gu/sglang.git
```

Or pull the specialized Docker image:

```bash
docker pull gar-registry.caas.intel.com/pytorch/pytorch-ipex-spr:intel-sglang-cpu-optimized
```

## 4. Model Deployment

This section provides deployment configurations optimized for the hardware platforms and use cases.

**Interactive Command Generator**: Use the configuration selector below to generate a launch command for the Llama 3.2 collection.

import Llama32ConfigGenerator from '@site/src/components/autoregressive/Llama32ConfigGenerator';

<Llama32ConfigGenerator />

Please read the `Notes` part in the serving engine launching section in [the official SGLang CPU server document](https://docs.sglang.io/platforms/cpu_server.html#launch-of-the-serving-engine) to understand how to configure arguments like TP (tensor parallel) and NUMA binding.

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
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain the benefits of quantization in LLMs."},
    ],
    temperature=0.2,
    max_tokens=512,
)

print(resp.choices[0].message.content)
```

## 6. Benchmarking

To benchmark the performance, use the `sglang.bench_serving` command:

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

Adjust the settings according to your specific hardware and testing requirements. Detailed argument descriptions are available in [this doc](../../base/benchmarks/lm_benchmark.md).

---