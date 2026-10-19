import numpy as np
from typing import List, Dict, Tuple, Optional
from pymongo import MongoClient
from models import Document, Chunk

class MongoVectorStore:
    """Class quản lý VectorStore với cấu trúc:
       - documents: Dict[str, Document] (Key: doc_id)
       - chunks: List[Chunk]
    """
    def __init__(self, connection_uri: str, db_name: str):
        self.client = MongoClient(connection_uri)
        self.db = self.client[db_name]
        
        # Tạo 2 collection riêng biệt trong MongoDB
        self._doc_collection = self.db["documents"]
        self._chunk_collection = self.db["chunks"]

    # ----------------------------------------------------
    # PROPERTIES THEO ĐÚNG YÊU CẦU CẤU TRÚC
    # ----------------------------------------------------
    @property
    def documents(self) -> Dict[str, Document]:
        """Lấy tất cả Documents dưới dạng dict[str, Document]."""
        raw_docs = list(self._doc_collection.find({}))
        doc_dict = {}
        for d in raw_docs:
            doc_obj = Document.from_dict(d)
            doc_dict[doc_obj.doc_id] = doc_obj
        return doc_dict

    @property
    def chunks(self) -> List[Chunk]:
        """Lấy tất cả Chunks dưới dạng List[Chunk]."""
        raw_chunks = list(self._chunk_collection.find({}))
        return [Chunk.from_dict(c) for c in raw_chunks]

    # ----------------------------------------------------
    # HÀM LƯU DỮ LIỆU
    # ----------------------------------------------------
    def save_document(self, document: Document):
        """Lưu hoặc cập nhật thông tin Document nguyên bản."""
        self._doc_collection.update_one(
            {"doc_id": document.doc_id},
            {"$set": document.to_dict()},
            upsert=True
        )

    def save_chunks(self, chunks: List[Chunk]):
        """Lưu danh sách Chunks kèm Vector Embeddings."""
        docs = [chunk.to_dict() for chunk in chunks]
        if docs:
            self._chunk_collection.insert_many(docs)

    def add_data(self, document: Document, chunks: List[Chunk]):
        """Hàm tiện ích lưu cả Document và danh sách Chunks cùng lúc."""
        self.save_document(document)
        self.save_chunks(chunks)

    # ----------------------------------------------------
    # HÀM TRUY VẤN COSIN SIMILARITY
    # ----------------------------------------------------
    @staticmethod
    def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
        """Tính Cosine Similarity giữa 2 vector."""
        a = np.array(v1)
        b = np.array(v2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def search_similar(self, query_embedding: List[float], top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Tìm kiếm Top K Chunks có Cosine Similarity cao nhất từ self.chunks."""
        scored_chunks = []

        # Truy vấn trực tiếp trên thuộc tính self.chunks (List[Chunk])
        for chunk in self.chunks:
            if chunk.embedding:
                score = self._cosine_similarity(query_embedding, chunk.embedding)
                scored_chunks.append((chunk, score))

        # Sắp xếp giảm dần theo score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]