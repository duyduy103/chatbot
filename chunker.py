from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from models import Document, Chunk

class RecursiveChunker:
    """Class chịu trách nhiệm phân tách Document thành các Chunks."""
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 40):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def split(self, document: Document) -> List[Chunk]:
        raw_chunks = self.splitter.split_text(document.content)
        chunks = []
        for idx, text in enumerate(raw_chunks):
            chunk_id = f"{document.doc_id}_chunk_{idx}"
            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                content=text,
                metadata={**document.metadata, "chunk_index": idx}
            )
            chunks.append(chunk)
        return chunks