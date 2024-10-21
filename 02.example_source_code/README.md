# Example Source Code

## 기본 예제 코드

01.basic_inference.py

- 결과는 나오지만 내용은 이상하다.

```bash
 How did you get your first job? You probably don't know about the robot, right? Except you're not using a computer to learn painting.

Well, when I started working on the robot, I was already building a system for a table. I couldn't figure out what I was doing, so I designed a series of components. I don't think there's a simple reason for this, but in the beginning I was starting from scratch to build the system. I then realized that
```

## 기본 예제 코드 - 모델 변경(meta-llama/Llama-3.1-8B-Instruct)

02.basic_inference_llama-3.1-8B-Instruct.py

- Hugging Face의 token을 입력 하지 않음
- 오류 발생

```bash
WARNING 10-18 09:59:51 arg_utils.py:956] Chunked prefill is enabled by default for models with max_model_len > 32K. Currently, chunked prefill might not work with some features or models. If you encounter any issues, please disable chunked prefill by setting --enable-chunked-prefill=False.
INFO 10-18 09:59:51 config.py:1023] Chunked prefill is enabled with max_num_batched_tokens=512.
INFO 10-18 09:59:51 llm_engine.py:237] Initializing an LLM engine (v0.6.4.dev26+g92d86da2.d20241017) with config: model='meta-llama/Llama-3.1-8B-Instruct', speculative_config=None, tokenizer='meta-llama/Llama-3.1-8B-Instruct', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, rope_scaling=None, rope_theta=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=131072, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=meta-llama/Llama-3.1-8B-Instruct, use_v2_block_manager=True, num_scheduler_steps=1, chunked_prefill_enabled=True multi_step_stream_outputs=True, enable_prefix_caching=False, use_async_output_proc=True, use_cached_outputs=False, mm_processor_kwargs=None)
INFO 10-18 09:59:52 model_runner.py:1061] Starting to load model meta-llama/Llama-3.1-8B-Instruct...
INFO 10-18 09:59:53 weight_utils.py:243] Using model weights format ['*.safetensors']
Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:01,  1.68it/s]
Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.61it/s]
Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:01<00:00,  1.59it/s]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  2.14it/s]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.91it/s]

INFO 10-18 09:59:55 model_runner.py:1072] Loading model weights took 14.9888 GB
INFO 10-18 09:59:56 gpu_executor.py:122] # GPU blocks: 2575, # CPU blocks: 2048
INFO 10-18 09:59:56 gpu_executor.py:126] Maximum concurrency for 131072 tokens per request: 0.31x
[rank0]: Traceback (most recent call last):
[rank0]:   File "/home/dev/app/source_code/vllm_practice/02.example_source_code/02.basic_inference_llama-3.1-8B-Instruct.py", line 5, in <module>
[rank0]:     llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
[rank0]:   File "/home/dev/app/virtual_env/vllm_practice/lib/python3.10/site-packages/vllm/entrypoints/llm.py", line 177, in __init__
[rank0]:     self.llm_engine = LLMEngine.from_engine_args(
[rank0]:   File "/home/dev/app/virtual_env/vllm_practice/lib/python3.10/site-packages/vllm/engine/llm_engine.py", line 574, in from_engine_args
[rank0]:     engine = cls(
[rank0]:   File "/home/dev/app/virtual_env/vllm_practice/lib/python3.10/site-packages/vllm/engine/llm_engine.py", line 349, in __init__
[rank0]:     self._initialize_kv_caches()
[rank0]:   File "/home/dev/app/virtual_env/vllm_practice/lib/python3.10/site-packages/vllm/engine/llm_engine.py", line 497, in _initialize_kv_caches
[rank0]:     self.model_executor.initialize_cache(num_gpu_blocks, num_cpu_blocks)
[rank0]:   File "/home/dev/app/virtual_env/vllm_practice/lib/python3.10/site-packages/vllm/executor/gpu_executor.py", line 129, in initialize_cache
[rank0]:     self.driver_worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)
[rank0]:   File "/home/dev/app/virtual_env/vllm_practice/lib/python3.10/site-packages/vllm/worker/worker.py", line 262, in initialize_cache
[rank0]:     raise_if_cache_size_invalid(num_gpu_blocks,
[rank0]:   File "/home/dev/app/virtual_env/vllm_practice/lib/python3.10/site-packages/vllm/worker/worker.py", line 492, in raise_if_cache_size_invalid
[rank0]:     raise ValueError(
[rank0]: ValueError: The model's max seq len (131072) is larger than the maximum number of tokens that can be stored in KV cache (41200). Try increasing gpu_memory_utilization or decreasing max_model_len when initializing the engine.
```

✔️ 코드 수정

03.basic_inference_llama-3.1-8B-Instruct.py

- `gpu_memory_utilization`, `max_model_len` 설정 하였더니 성공적으로 실행
- 위 2항목의 깊이 있는 분석 필요
- 큰 모델을 사용 했더니 내용도 적절하게 나옴

```python
...
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct", 
    gpu_memory_utilization=0.9,  # Increase GPU memory usage
    max_model_len=40960          # Reduce the maximum sequence length
)
...
```

> 실행 결과

```bash
WARNING 10-18 10:08:47 arg_utils.py:956] Chunked prefill is enabled by default for models with max_model_len > 32K. Currently, chunked prefill might not work with some features or models. If you encounter any issues, please disable chunked prefill by setting --enable-chunked-prefill=False.
INFO 10-18 10:08:47 config.py:1023] Chunked prefill is enabled with max_num_batched_tokens=512.
INFO 10-18 10:08:47 llm_engine.py:237] Initializing an LLM engine (v0.6.4.dev26+g92d86da2.d20241017) with config: model='meta-llama/Llama-3.1-8B-Instruct', speculative_config=None, tokenizer='meta-llama/Llama-3.1-8B-Instruct', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, rope_scaling=None, rope_theta=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=40960, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=meta-llama/Llama-3.1-8B-Instruct, use_v2_block_manager=True, num_scheduler_steps=1, chunked_prefill_enabled=True multi_step_stream_outputs=True, enable_prefix_caching=False, use_async_output_proc=True, use_cached_outputs=False, mm_processor_kwargs=None)
INFO 10-18 10:08:48 model_runner.py:1061] Starting to load model meta-llama/Llama-3.1-8B-Instruct...
INFO 10-18 10:08:48 weight_utils.py:243] Using model weights format ['*.safetensors']
Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:01,  1.79it/s]
Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.65it/s]
Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:01<00:00,  1.61it/s]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  2.17it/s]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.95it/s]

INFO 10-18 10:08:51 model_runner.py:1072] Loading model weights took 14.9888 GB
INFO 10-18 10:08:52 gpu_executor.py:122] # GPU blocks: 2575, # CPU blocks: 2048
INFO 10-18 10:08:52 gpu_executor.py:126] Maximum concurrency for 40960 tokens per request: 1.01x
INFO 10-18 10:08:53 model_runner.py:1400] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
INFO 10-18 10:08:53 model_runner.py:1404] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
INFO 10-18 10:09:02 model_runner.py:1528] Graph capturing finished in 9 secs.
Processed prompts: 100%|█| 1/1 [00:20<00:00, 20.61s/it, est. speed input: 0.63 toks/s, output: 48.5
==============================
 “Robbie’s Brush with Art”

Robbie, a sleek and modern robot, was created to assist in various tasks around the house. But as he hummed along, day in and day out, he began to feel a sense of monotony. The same routine, day in and day out. His creator, an eccentric artist named Dr. Luna, had other plans for Robbie. She wanted to teach him the art of painting.
...
```

## 기본 예제 코드 - 한국어

04.basic_inference_ko.py

- 한국어는 제대로 된 답변이 나오지 않음

```python
...
prompt = "그림을 배우는 로봇에 대한 짧은 이야기를 들려주세요:"
...
```

> 실행 결과

```bash
거룩한 원칙: 
1. 밑그림을 먼저 그리고 집중하세요. 
2. 모서리에서 시작하세요. 
3. 회전 중심점은 회전 중심점입니다. 

로봇의 기술: 
1. 1000 개의 브러시로 그림을 그리기. 
2. 유화 교육을 받고 화면을 연습하기. 

결과: 
로봇의 그림은 능숙하고 간결하다. 
일부는 여전히 도전적으로 보인다. 
로봇은 아직까지 화면을 그리기 위한 좋은 사례가 아니다. 
...
```

## 기본 예제 코드 - 채팅

05.basic_inference_chat.py

`sh2orc/llama-3-korean-8b` 모델 사용: 한국어로 파인튜닝된 모델

- `tokenizer`추가
- `tokenizer.apply_chat_template` 추가
- `stop_token_ids` 추가
- 정상 작동

> 실행 결과

```bash
대한민국의 수도는 서울특별시입니다.

서울특별시는 한국의 수도이자 가장 큰 도시입니다. 면적은 약 605.21 km²이며, 인구는 약 10,218,107명입니다. 서울특별시는 1394년 조선 королKing Taejo가 경복궁을 세우면서 수도가 된 곳으로, 조선 시대에 대표적인 문화와 정치의 중심지로 성장했습니다. 도시의 명칭은 '남쪽의 경성'으로, 경복궁은 왕이 생활하며 정치 활동을 하는 곳으로 모두가 볼 수 있도록 공개하는 제도를 만들었습니다. 

서울특별시에는 다양한 문화유산이 곳곳에 남아 있습니다. 한옥마을, 종로, 종로5가, 경복궁, 전통시장 등이 대표적인 예입니다. 한옥마을은 정착형 한옥을 보존하고, 종로와 종로5가는 서울의 역사와 문화를 살펴볼 수 있는 곳입니다. 경복궁은 조선 시대 왕실의 궁궐로, 전통시장은 다양한 음식과 상품이 판매되는 곳입니다. 

또한, 서울은 젊은 인구가 많은 도시로, 다양한 즐길 거리가 많습니다. 함께 볼 수 있는 곳으로는 명동, 홍대, 이태원, 명륜3사거리 등이 있습니다. 이외에도, 서울에는 다양한 박물관과 미술관, 공원과 공연장이 많습니다.
```

## FastAPI 통합 - meta-llama/Llama-3.1-8B-Instruct

- 오프라인과 동일하게 문장은 생성 되나 생성이 종료 되지 않고 문장이 반복 됨

06.fastapi_server.py

- FastAPI 서버 구현

```bash
python3 06.fastapi_server.py
```

07.fastapi_api_call.py

- FastAPI 서버 호출

```bash
python3 07.fastapi_api_call.py
```

> 실행 결과

```bash
Generated Text:
1. 수도는 서울입니다. 2. 서울은 1,578.7k㎡의 면적을 가지고 있습니다. 3. 서울은 2021년 10월에 10,212,976명의 인구를 가지고
...
```

## FastAPI 통합 - sh2orc/llama-3-korean-8b

- 한국어로 파인튜닝된 모델 사용
- 정상 작동

08.fastapi_server.py

- FastAPI 서버 구현

```bash
python3 08.fastapi_server.py
```

09.fastapi_api_call.py

- FastAPI 서버 호출

```bash
python3 09.fastapi_api_call.py
```

> 실행 결과

```bash
Generated Text:
대한민국의 수도는 서울입니다. 서울은 대한민국의 역사적인 수도였으며, 한국의 경제, 문화, 정치, 교통의 중심지입니다. 서울은 아름다운 자연 경관과 다양한 문화유산을 보유하고 있으며, 많은 문화행사와 축제를 통해 대표되는 도시입니다. 또한, 현대적인 도시로 다양한 시설과 인프라가 잘 발달하고 있습니다.
```