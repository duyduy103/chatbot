import sys
import os
import io
from dotenv import load_dotenv

# Ép luồng chuẩn stdout dùng UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Lấy đường dẫn thư mục hiện tại chứa chính file query_rag.py

# Nạp file .env từ đường dẫn tương đối
load_dotenv()

from rag_service import RAGPipeline

def answer_user_question(question: str, top_k: int = 3):
    try:
        rag = RAGPipeline()
        return rag.query(question, top_k=top_k)
    except Exception as e:
        return f"Lỗi xử lý RAG: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_query = sys.argv[1]
        print(answer_user_question(user_query, top_k=3))