# inference/

The code that produced the model responses reported in the manuscript. It is organised by serving
arrangement, because that is what differs between models: three code paths cover the six models and
the hosted-API comparison.

| Folder | Models | Hardware | Engine |
|---|---|---|---|
| `A100-Transformers/` | Gemma-3-12B, Gemma-3-27B, Mistral-Small-3.2, Qwen-2.5 | 1x NVIDIA A100-SXM4-80GB each | Hugging Face `transformers` |
| `H200-vLLM/` | Llama-4-Maverick, Llama-4-Scout | 2-4x NVIDIA H200 each | vLLM |
| `API/` | All six, through one gateway | hosted | OpenRouter |

## Why the six local models take two code paths

Llama-4-Maverick and Llama-4-Scout cannot be served the way the other four are.

`transformers` decompresses a quantized checkpoint to bf16 before generation, whatever the hardware
supports. Maverick's fp8 checkpoint is 416.8 GB on disk and about 803 GB resident on that path,
which is more than any allocation available here can hold. vLLM consumes the fp8 weights directly,
at a resident size close to the on-disk figure.

Scout does fit on the `transformers` path, since it ships at bf16, but generates at roughly
37,900 ms per image there against roughly 24 ms per image under vLLM. That gap decides whether the
image-variant and sensitivity runs are affordable for it at all.

Both models are therefore served with vLLM on H200 GPUs, which is why `H200-vLLM/worker_vllm.py` is
a separate script from `A100-Transformers/run_worker.py`. The two share the prompts, the image
preprocessing and the parser, and differ only in the inference engine underneath.

Each subfolder has its own README describing what its code does.
