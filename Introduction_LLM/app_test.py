from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama

# Initialize FastAPI
app = FastAPI()

# Load Llama model
llm = Llama.from_pretrained(
    repo_id="rubythalib33/llama3_1_8b_finetuned_bahasa_indonesia",
    filename="unsloth.Q4_K_M.gguf",
)

# Define request and response models
class ChatRequest(BaseModel):
    instruction: str
    input_data: str = ""

class ChatResponse(BaseModel):
    response: str

# Alpaca-style prompt template
alpaca_prompt = """Di bawah ini adalah instruksi yang menjelaskan tugas, dipasangkan dengan masukan yang memberikan konteks lebih lanjut. Tulis tanggapan yang melengkapi permintaan dengan tepat.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

@app.post("/chat", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    # Format the instruction and input for the model
    prompt = alpaca_prompt.format(request.instruction, request.input_data, "")
    
    # Generate the chat completion
    result = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    # Extract and return the result
    response_text = result['choices'][0]['message']['content']
    
    return ChatResponse(response=response_text)

# Run the app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)