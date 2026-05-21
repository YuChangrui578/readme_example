# Llama 3.2

## 1. Model Introduction

Llama 3.2 is a collection of lightweight, optimized models designed for efficient deployment. This family includes highly quantized versions (W8A8, FP8, AWQ) specifically tailored for high-performance inference on various hardware platforms.

These models bring enhanced efficiency and intelligence to edge and server-side deployments:

- **Optimized Quantization**: Support for W8A8, FP8, and AWQ quantization formats to balance precision and performance.
- **High Efficiency**: Designed for low-latency applications and reduced memory footprint.
- **Instruction Tuned**: Availability of Instruct variants for conversational and task-oriented workflows.

For further details, please refer to the [Llama 3.2 model cards on Hugging Face](https://huggingface.co/RedHatAI/Llama-3.2-3B-quantized.w8a8).

## 2. Model Acquisition

You can access the models on Hugging Face.

| Data Type | Model Card |
|:---:|:---:|
| W8A8 | [RedHatAI/Llama-3.2-3B-quantized.w8a8](https://huggingface.co/RedHatAI/Llama-3.2-3B-quantized.w8a8) |
| FP8 | [RedHatAI/Llama-3.2-3B-Instruct-FP8](https://huggingface.co/RedHatAI/Llama-3.2-3B-Instruct-FP8) |
| AWQ | [AMead10/Llama-3.2-3B-Instruct-AWQ](https://huggingface.co/AMead10/Llama-3.2-3B-Instruct-AWQ) |

The models can be downloaded to local storage by command:

```bash
hf download --resume <MODEL_ID> --local-dir 'path/to/local/dir'
```

Then set `--model-path <LOCAL_MODEL_PATH>` instead of `--model-path <MODEL_ID>` in the `sglang.launch_server` command.

*Note:* You may need to log in to your authorized HuggingFace account to access the model files. Please refer to [HuggingFace login](https://huggingface.co/docs/huggingface_hub/quick-start#login).

## 3. SGLang Installation

For the supported data types (W8A8, FP8), please refer to the [official SGLang installation guide](https://docs.sglang.io/platforms/cpu_server.html#installation) for installation instructions.

For AWQ quantized models, ensure you are using the optimized branch:

```bash
git clone -b cpu_optimized https://github.com/jianan-gu/sglang.git
```

You can pull the docker image if you have access to `gar-registry.caas.intel.com`:

```bash
docker pull gar-registry.caas.intel.com/pytorch/pytorch-ipex-spr:intel-sglang-cpu-optimized
```

## 4. Model Deployment

This section provides deployment configurations optimized for the hardware platforms and use cases.

**Interactive Command Generator**: Use the configuration selector below to generate a launch command for the Llama 3.2 collection of models.

import Llama32ConfigGenerator from '@site/src/components/autoregressive/Llama32ConfigGenerator';

<Llama32ConfigGenerator />

Please read the `Notes` part in the serving engine launching section in [the official SGLang CPU server document](https://docs.sglang.io/platforms/cpu_server.html#launch-of-the-serving-engine) to better understand how to configure the arguments, especially for TP (tensor parallel) and numa binding settings.

## 5. Model Invocation

SGLang exposes an OpenAI-compatible endpoint. First, start the server:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="EMPTY",
)

resp = client.chat.completions.create(
    model="RedHatAI/Llama-3.2-3B-quantized.w8a8",
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

Please adjust the settings per your benchmarking scenarios. Detailed descriptions for the arguments of `bench_serving` are available in [this doc](../../base/benchmarks/lm_benchmark.md).