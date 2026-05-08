from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict   
from rag.chain import run_rag

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    history: Optional[List[Dict[str, str]]] = []  

@app.on_event("startup")
def startup_event():
    print(" Server is running and ready!")

@app.get("/")
def root():
    print("Server is Running")
    return {"message": "Server is running!"}

@app.post("/chat")
def chat(request: ChatRequest):
    answer = run_rag(
        query=request.question,
        history=request.history
    )
    return {"answer": answer}