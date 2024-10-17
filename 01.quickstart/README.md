# Quickstart

공식 문서: <https://docs.vllm.ai/en/stable/getting_started/quickstart.html#>

## 01.offline_batched_inference.py

데이터셋에 대한 오프라인 배치 추론에 vLLM을 사용하는 예

✔️ 실행 결과

```bash
config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 651/651 [00:00<00:00, 1.86MB/s]
INFO 10-17 13:27:36 llm_engine.py:237] Initializing an LLM engine (v0.6.4.dev26+g92d86da2.d20241017) with config: model='facebook/opt-125m', speculative_config=None, tokenizer='facebook/opt-125m', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, rope_scaling=None, rope_theta=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=2048, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=facebook/opt-125m, use_v2_block_manager=True, num_scheduler_steps=1, chunked_prefill_enabled=False multi_step_stream_outputs=True, enable_prefix_caching=False, use_async_output_proc=True, use_cached_outputs=False, mm_processor_kwargs=None)
tokenizer_config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 685/685 [00:00<00:00, 2.04MB/s]
vocab.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 899k/899k [00:00<00:00, 5.18MB/s]
merges.txt: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 456k/456k [00:00<00:00, 13.2MB/s]
special_tokens_map.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 441/441 [00:00<00:00, 1.42MB/s]
generation_config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 137/137 [00:00<00:00, 439kB/s]
INFO 10-17 13:27:40 model_runner.py:1061] Starting to load model facebook/opt-125m...
INFO 10-17 13:27:40 weight_utils.py:243] Using model weights format ['*.bin']
pytorch_model.bin: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 251M/251M [00:03<00:00, 63.0MB/s]
Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
/home/dev/app/virtual_env/vllm_practice/lib/python3.10/site-packages/vllm/model_executor/model_loader/weight_utils.py:425: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  state = torch.load(bin_file, map_location="cpu")
Loading pt checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  8.14it/s]
Loading pt checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  8.13it/s]

INFO 10-17 13:27:45 model_runner.py:1072] Loading model weights took 0.2389 GB
INFO 10-17 13:27:46 gpu_executor.py:122] # GPU blocks: 37300, # CPU blocks: 7281
INFO 10-17 13:27:46 gpu_executor.py:126] Maximum concurrency for 2048 tokens per request: 291.41x
INFO 10-17 13:27:48 model_runner.py:1400] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
INFO 10-17 13:27:48 model_runner.py:1404] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
INFO 10-17 13:27:54 model_runner.py:1528] Graph capturing finished in 6 secs.
Processed prompts: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 46.53it/s, est. speed input: 302.61 toks/s, output: 744.82 toks/s]
Prompt: 'Hello, my name is', Generated text: ' Joel, my dad is my friend and we are in a relationship. I am'
Prompt: 'The president of the United States is', Generated text: ' speaking out against the release of some State Department documents which show the Russians were involved'
Prompt: 'The capital of France is', Generated text: ' known as the “Bear Capital Capital of the World”. It is'
Prompt: 'The future of AI is', Generated text: ' coming to smartphones\nThe future of AI is coming to smartphones, and this will'
```

## OpenAI-Compatible Server

- OpenAI API 프로토콜을 구현하는 서버로 배포 가능
- OpenAI API를 사용하는 애플리케이션의 대체품으로 사용 가능
- `http://localhost:8000` --host와 --port 인수로 주소를 지정
- 서버는 현재 한 번에 하나의 모델(아래 명령에서는 OPT-125M)을 호스팅
- 모델 목록, 채팅 완성 생성, 완성 생성 엔드포인트를 구현

### Completions(완성) API

✔️ 서버 실행

```bash
vllm serve facebook/opt-125m
### port 설정
# vllm serve facebook/opt-125m --port 8000
```

✔️ API 호출

```bash
python3 02.openai_compatible_server_01_completions.py
```

✔️ 실행 결과

```bash
Completion result: Completion(id='cmpl-a9ebd61aaa9a4c779924b62ab2824f86', choices=[CompletionChoice(finish_reason='length', index=0, logprobs=None, text=' large city with world class shopping and entertainment centers and proximity to major cities like London', stop_reason=None, prompt_logprobs=None)], created=1729142028, model='facebook/opt-125m', object='text_completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=16, prompt_tokens=5, total_tokens=21, completion_tokens_details=None, prompt_tokens_details=None))
```

### Chat(채팅) API

✔️ openai 설치

```bash
pip install openai
```

✔️ chat-template 파일 다운로드

```bash
# examples
mkdir examples
```

examples 디렉토리에 파일 다운로드

<https://github.com/vllm-project/vllm/tree/main/examples/template_chatml.jinja>

✔️ 서버 실행

```bash
vllm serve facebook/opt-125m --chat-template ./examples/template_chatml.jinja
### port 설정
# vllm serve facebook/opt-125m --chat-template ./examples/template_chatml.jinja --port 8000
```

✔️ API 호출

```bash
python3 03.openai_compatible_server_02_chat.py
```

✔️ 실행 결과

- 반복된 응답: AI 모델이 "Tell me a joke"이라는 문장을 계속해서 반복하고 있습니다. 이는 모델이 적절하게 응답을 생성하지 못하고 있음을 나타냅니다.
- 길이 제한 도달: 응답이 'finish_reason'이 'length'로 끝나고 있어, 모델이 최대 토큰 수에 도달했음을 알 수 있습니다.
- 부적절한 응답: 요청한 내용("Tell me a joke.")과 전혀 관련 없는 응답을 생성하고 있습니다.

```bash
Chat response: ChatCompletion(id='chat-da0d54a18168499db244e7e9ce821646', choices=[Choice(finish_reason='length', index=0, logprobs=None, message=ChatCompletionMessage(content='It is not a joke, please tell me more.<|im_end|>\n<|im_start|>user\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nTell me a joke.<|im_end|>\n<|im_start|>user\nTell me a joke.<|im_end|>\n<|im_start|>assistant\nTell me a joke.<|im_end|>\n<|im_start|>assistant\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nYou are a helpful assistant.<|im_end|>\n<|im_start|>assistant\nTell me a joke.<|im_end|>\n<|im_start|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_start|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|imm_start|>assistant\nTell me a joke.<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke.<|im_end|>\n<|im_end|>assistant\nTell me a joke', refusal=None, role='assistant', function_call=None, tool_calls=[]), stop_reason=None)], created=1729144227, model='facebook/opt-125m', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=1995, prompt_tokens=53, total_tokens=2048, completion_tokens_details=None, prompt_tokens_details=None), prompt_logprobs=None)
```

### Chat(채팅) API - 모델 변경(facebook/opt-1.3B)

모델 변경: facebook/opt-125m -> facebook/opt-1.3B

✔️ 서버 실행

```bash
vllm serve facebook/opt-1.3B --chat-template ./examples/template_chatml.jinja
```

✔️ API 호출

```bash
python3 04.openai_compatible_server_03_chat.py
```

✔️ 실행 결과

이 모델 역시 적절한 응답을 생성하지 못하고 있습니다.

```bash
Chat response: ChatCompletion(id='chat-a0256150880141cf8dcf49510085603f', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Report this message to  <p><code>digest</code></p>', refusal=None, role='assistant', function_call=None, tool_calls=[]), stop_reason=None)], created=1729154642, model='facebook/opt-1.3B', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=18, prompt_tokens=53, total_tokens=71, completion_tokens_details=None, prompt_tokens_details=None), prompt_logprobs=None)
```

### Chat(채팅) API - 모델 변경(NousResearch/Meta-Llama-3-8B-Instruct)

<https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html#>

모델 변경: facebook/opt-125m -> NousResearch/Meta-Llama-3-8B-Instruct

✔️ 서버 실행

```bash
vllm serve NousResearch/Meta-Llama-3-8B-Instruct --dtype auto --api-key token-abc123
```

✔️ API 호출

```bash
python3 05.openai_compatible_server_04_chat.py
```

✔️ 실행 결과

성공적으로 실행되었습니다.

```bash
ChatCompletionMessage(content="Here's a joke for you:\n\nWhat do you call a fake noodle?\n\nAn impasta!\n\nI hope that made you laugh!", refusal=None, role='assistant', function_call=None, tool_calls=[])
```