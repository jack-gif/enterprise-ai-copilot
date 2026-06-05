from google import genai
import os
from dotenv import load_dotenv
from langchain.embeddings.base import Embeddings


load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))





class GeminiEmbeddingService():
    def embed_documents(self, texts: list[str]):
        """Embeds a list of texts (documents/chunks)."""
        # Using batch processing is faster than a loop
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=texts
        )
        
        # result.embeddings is a list of objects; each has a .values attribute
        return [e.values for e in result.embeddings]

    def embed_query(self, text: str):
        """Embeds a single text (the user question)."""
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text
        )
        
        # For a single query, .embeddings is still an object with .values
        # If the SDK returns it as a list of 1, we grab the first index
        if isinstance(result.embeddings, list):
            return result.embeddings[0].values
        
        return result.embeddings.values

class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return GeminiEmbeddingService().embed_documents(texts)

    def embed_query(self, text):
        return GeminiEmbeddingService().embed_query(text)