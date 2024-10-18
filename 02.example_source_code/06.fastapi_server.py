from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import uvicorn

app = FastAPI()

# Initialize the model
# BASE_MODEL = "sh2orc/llama-3-korean-8b"
# llm = LLM(model=BASE_MODEL)

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct", 
    gpu_memory_utilization=0.9,  # Increase GPU memory usage
    max_model_len=40960          # Reduce the maximum sequence length
)

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.95

class GenerationResponse(BaseModel):
    generated_text: str

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    try:
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens
        )
        outputs = llm.generate([request.prompt], sampling_params)
        return GenerationResponse(generated_text=outputs[0].outputs[0].text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)