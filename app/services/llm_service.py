from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable
from tenacity import retry, stop_after_attempt, wait_exponential
import os
from dotenv import load_dotenv

load_dotenv()


print("LANGCHAIN_API_KEY:", os.getenv("LANGCHAIN_API_KEY"))
print("LANGCHAIN_TRACING_V2:", os.getenv("LANGCHAIN_TRACING_V2"))

class LLMService:
    def __init__(self):
          self.llm = ChatGoogleGenerativeAI(
               model = "gemini-2.5-flash",
               temperature=0.2,
               google_api_key = os.getenv("GOOGLE_API_KEY")
          )
    @traceable(name="LLM Generate")
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_response(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content.strip()