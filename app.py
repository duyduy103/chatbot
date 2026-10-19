
import os
import hmac

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from query_rag import answer_user_question

app = FastAPI(title="SportShop RAG Chatbot")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)


@app.get("/")
def home():
    return {"status": "ok", "service": "SportShop RAG Chatbot"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/chat")
def chat(
    request: ChatRequest,
    x_api_key: str = Header(default="")
):
    expected_key = os.getenv("CHATBOT_API_KEY", "")

    if not expected_key:
        raise HTTPException(
            status_code=500,
            detail="Chatbot API key is not configured"
        )

    if not hmac.compare_digest(x_api_key, expected_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    answer = answer_user_question(question, top_k=3)

    if isinstance(answer, str) and answer.startswith("Lỗi xử lý RAG:"):
        raise HTTPException(
            status_code=500,
            detail="Failed to process the RAG question"
        )

    return {"answer": answer}
