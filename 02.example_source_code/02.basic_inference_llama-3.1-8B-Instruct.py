from vllm import LLM, SamplingParams

# Initialize the model
# llm = LLM(model="facebook/opt-125m")
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

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
