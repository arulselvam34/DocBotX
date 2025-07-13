from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
import os
import numpy as np
from typing import List

class SimpleEmbeddings(Embeddings):
    """Simple embeddings using basic text processing"""
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return [self._embed_text(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self._embed_text(text)
    
    def _embed_text(self, text: str) -> List[float]:
        """Simple embedding based on character frequencies"""
        # Simple hash-based embedding
        text = text.lower()
        embedding = [0.0] * 384  # Standard embedding size
        for i, char in enumerate(text[:384]):
            embedding[i % 384] += ord(char) / 1000.0
        # Normalize
        norm = sum(x*x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x/norm for x in embedding]
        return embedding

def get_groq_llm(model_name="llama-3.1-8b-instant"):
    """Get a Groq LLM"""
    import streamlit as st
    # Try Streamlit secrets first, then environment variable
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    return ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=0.7
    )

def get_free_embeddings():
    """Get simple embeddings that work offline"""
    return SimpleEmbeddings()

def get_available_groq_models():
    """Return list of available Groq models"""
    return [
        "llama-3.1-8b-instant",
        "llama3-8b-8192",
        "llama3-70b-8192"
    ]