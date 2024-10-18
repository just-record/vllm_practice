from vllm import LLM, SamplingParams

# Initialize the model
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct", 
    gpu_memory_utilization=0.9,  # Increase GPU memory usage
    max_model_len=40960          # Reduce the maximum sequence length
)

# Set up the prompt
prompt = "Tell me a short story about a robot learning to paint:"

# Set up sampling parameters
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1000)

# Generate the output
outputs = llm.generate([prompt], sampling_params)

# Print the generated text
print("=" * 30)
for output in outputs:
    print(output.outputs[0].text)
