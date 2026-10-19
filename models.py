from typing import List, Dict, Any, Optional

class Document:
    """Đại diện cho tài liệu nguyên bản đầu vào."""
    def __init__(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        self.doc_id = doc_id
        self.content = content
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thành Dictionary để lưu vào MongoDB."""
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Khôi phục object Document từ MongoDB document."""
        return cls(
            doc_id=data["doc_id"],
            content=data["content"],
            metadata=data.get("metadata", {})
        )

    def __repr__(self):
        return f"<Document id={self.doc_id}, length={len(self.content)}>"


class Chunk:
    """Đại diện cho một đoạn nhỏ sau khi phân tách (chunking)."""
    def __init__(
        self, 
        chunk_id: str, 
        doc_id: str, 
        content: str, 
        embedding: Optional[List[float]] = None, 
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.content = content
        self.embedding = embedding
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Chunk':
        return cls(
            chunk_id=data["chunk_id"],
            doc_id=data["doc_id"],
            content=data["content"],
            embedding=data.get("embedding"),
            metadata=data.get("metadata", {})
        )