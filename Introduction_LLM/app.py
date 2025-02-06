from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_cpp import Llama

# Initialize FastAPI
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Anda dapat membatasi domain tertentu di sini
    allow_credentials=True,
    allow_methods=["*"],  # Mengizinkan semua metode (GET, POST, dll.)
    allow_headers=["*"],  # Mengizinkan semua header
)

# Path model yang telah disalin ke Google Drive
model_path = "/unsloth.Q4_K_M.gguf"

# Memuat model dari Google Drive
llm = Llama(model_path)

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

@app.get("/")
async def root():
    return {"message": "LLM Model is successfully running"}

# Run the app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)