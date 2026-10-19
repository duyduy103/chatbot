from jina_embeddings import jina_embeddings
import os 
from dotenv import load_dotenv
import numpy as np 

load_dotenv()
modelemb = os.getenv('MODEL_EMBEDING')

def embed_tex(text: str, model=modelemb, api_key=None, task="retrieval.query") -> list[float]:
    if api_key is None:
        api_key = os.getenv("JINA_EMBEDING")
    
    resp = jina_embeddings(input=text, model=model, api_key=api_key, task=task)
    return resp.data[0].embedding

def batch_embed_text(input: list[str], model=modelemb, api_key=None, task="retrieval.passage") -> list[list[float]]:
    if api_key is None:
        api_key = os.getenv("JINA_EMBEDING")
        
    resp = jina_embeddings(input=input, model=model, api_key=api_key, task=task)
    return [item.embedding for item in resp.data]