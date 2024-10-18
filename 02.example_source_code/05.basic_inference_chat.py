from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, pipeline

BASE_MODEL = "sh2orc/llama-3-korean-8b"
llm = LLM(model=BASE_MODEL)

# BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
# llm = LLM(
#     model="meta-llama/Llama-3.1-8B-Instruct", 
#     gpu_memory_utilization=0.9,  # Increase GPU memory usage
#     max_model_len=40960          # Reduce the maximum sequence length
# )

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    # {"role": "user", "content": "Tell me a joke."}, 
    # {"role": "user", "content": "재밌는 농담을 하나 들려주세요."}, 
    {"role": "user", "content": "대한민국의 수도는 어디인가요? 그리고 그 수도에 대해 설명해주세요."}, 
]

prompt_message = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True,
)

eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

outputs = llm.generate(prompt_message, SamplingParams(stop_token_ids=eos_token_id, temperature=0.8, top_p=0.95,max_tokens=512))

for output in outputs:
    propt = output.prompt
    generated_text = output.outputs[0].text
    print(generated_text)