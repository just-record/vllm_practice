from vllm import LLM, SamplingParams

# Initialize the model
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct", 
    gpu_memory_utilization=0.9,  # Increase GPU memory usage
    max_model_len=40960          # Reduce the maximum sequence length
)

# Set up the prompt
# prompt = "Tell me a short story about a robot learning to paint:"
# prompt = "그림을 배우는 로봇에 대한 짧은 이야기를 들려주세요:"
prompt = "대한민국의 수도는 어디인가요? 그리고 그 수도에 대해 설명해주세요."

# Set up sampling parameters
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1000)
sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=1000)

# Generate the output
outputs = llm.generate([prompt], sampling_params)

# Print the generated text
print("=" * 30)
for output in outputs:
    print(output.outputs[0].text)
