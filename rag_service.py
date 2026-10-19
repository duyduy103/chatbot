import os
from typing import List, Tuple
from dotenv import load_dotenv
from openai import OpenAI

from models import Document, Chunk
from chunker import RecursiveChunker
from vectorstore import MongoVectorStore
from embedding_helper import embed_tex, batch_embed_text

load_dotenv()


class RAGPipeline:
    """Class điều phối toàn bộ luồng RAG kết hợp Jina Embedding & Custom LLM Endpoint."""
    def __init__(self):
        # Lấy chuỗi kết nối MongoDB Atlas từ biến môi trường
        mongo_uri = os.getenv("MONGO_URI")
        db_name = os.getenv("DB_NAME")

        self.chunker = RecursiveChunker(chunk_size=400, chunk_overlap=40)
        self.vector_store = MongoVectorStore(connection_uri=mongo_uri, db_name=db_name)
        
        # Cấu hình OpenAI client trỏ đến Custom Endpoint (nttcodex.com)
        self.llm_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://nttcodex.com/v1",
            timeout=600.0,
        )

    def ingest(self, document: Document):
        """Bước 1: Phân tách Document -> Vectorize Chunks bằng Jina -> Lưu vào MongoDB Atlas."""
        print(f"📥 [RAG] Đang bắt đầu xử lý Document: {document.doc_id}...")
        
        # 1. Tách document thành danh sách Chunks
        chunks = self.chunker.split(document)
        if not chunks:
            print("⚠️ [RAG] Không có chunk nào được tạo từ tài liệu!")
            return

        # 2. Lấy cấu hình Jina từ biến môi trường (hỗ trợ cả JINA_API_KEY và JINA_EMBEDING)
        jina_key = os.getenv("JINA_API_KEY") or os.getenv("JINA_EMBEDING", "jina_71b19e11f1c14eafbf5258052bb84058pMlnvr58Kd2UPXyGkZju02ukv8Cp")
        model_emb = os.getenv("MODEL_EMBEDING", "jina-embeddings-v5-text-small")

        # 3. Tạo Vector Embeddings hàng loạt cho chunks (task='retrieval.passage')
        texts = [c.content for c in chunks]
        embeddings = batch_embed_text(texts, model=model_emb, api_key=jina_key, task="retrieval.passage")
        
        for i, emb in enumerate(embeddings):
            chunks[i].embedding = emb
        
        # 4. Lưu cả Document và Chunks vào VectorStore (MongoDB Atlas)
        self.vector_store.add_data(document, chunks)
        print(f"✅ [RAG] Đã lưu Document '{document.doc_id}' và {len(chunks)} Chunks vào MongoDB Atlas thành công!")

    def retrieve(self, question: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Bước 2: Vector hóa câu hỏi bằng Jina (task='retrieval.query') và tìm kiếm Cosine Similarity."""
        jina_key = os.getenv("JINA_API_KEY") or os.getenv("JINA_EMBEDING")
        model_emb = os.getenv("MODEL_EMBEDING")

        query_embedding = embed_tex(question, model=model_emb, api_key=jina_key, task="retrieval.query")
        results = self.vector_store.search_similar(query_embedding, top_k=top_k)
        return results

    def query(self, question: str, top_k: int = 3) -> str:
        """Bước 3: Nhận câu hỏi -> Retrieve Context -> Gọi LLM tạo câu trả lời hoàn chỉnh."""
        similar_chunks_with_scores = self.retrieve(question, top_k=top_k)

        if not similar_chunks_with_scores:
            return "Rất tiếc, chưa có dữ liệu nào trong hệ thống để trả lời câu hỏi của bạn."

        # Ghép các nội dung chunk thành Context
        context_text = "\n---\n".join([chunk.content for chunk, _ in similar_chunks_with_scores])

        # Xây dựng Prompt cho LLM
        prompt = f"""Bạn là trợ lý AI thông minh cho hệ thống SportShop. Hãy sử dụng thông tin được cung cấp trong phần Context dưới đây để trả lời câu hỏi của người dùng một cách chính xác và tự nhiên nhất.
Nếu thông tin không nằm trong Context, hãy lịch sự thông báo rằng bạn không tìm thấy thông tin phù hợp.

Context:
{context_text}

Câu hỏi: {question}
Trả lời:"""

        # Gọi mô hình qua Custom Endpoint
        response = self.llm_client.chat.completions.create(
            model="gpt-6-luna",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        return response.choices[0].message.content