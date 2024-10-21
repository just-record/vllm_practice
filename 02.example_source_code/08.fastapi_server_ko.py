from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import uvicorn
from typing import List

app = FastAPI()

BASE_MODEL = "sh2orc/llama-3-korean-8b"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

# Initialize the model
llm = LLM(model=BASE_MODEL)

class GenerationRequest(BaseModel):
    messages: List
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.95

class GenerationResponse(BaseModel):
    generated_text: str

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    try:
        prompt_message = tokenizer.apply_chat_template(
                request.messages, 
                tokenize=False, 
                add_generation_prompt=True,
        )

        sampling_params = SamplingParams(
            stop_token_ids=eos_token_id,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens
        )
        outputs = llm.generate(prompt_message, sampling_params)
        return GenerationResponse(generated_text=outputs[0].outputs[0].text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)