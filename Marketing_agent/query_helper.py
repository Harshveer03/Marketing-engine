import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from dotenv import load_dotenv

load_dotenv()

VECTOR_DB_DIR = "./vectordb"

class ICPQueryHelper:
    def __init__(self, model="models/gemini-2.5-flash", embedding_model="nomic-embed-text"):
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.vectordb = FAISS.load_local(
            VECTOR_DB_DIR,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=0)

    def query(self, question: str, k: int = 5) -> str:
        """Ask a question against ICP/Niche knowledge base"""
        docs = self.vectordb.similarity_search(question, k=k)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
        You are an AI assistant specialized in business analysis and market positioning.

        Your Task:

            1. Carefully read and analyze the provided context about the company’s business niche and Ideal Customer Profile (ICP).

            2. Identify key insights, patterns, and differentiators that define the company’s positioning.

            3. Deliver a clear, structured, and actionable answer to the given question.

        Guidelines:

            1. Ensure your response is concise, accurate, and aligned with CXO-level decision-making.

            2. Highlight strategic relevance (growth, positioning, differentiation, or opportunities).

            3. Avoid generic statements; focus only on what the context strongly supports.

        Present your insights in a direct, professional tone (no fluff).

        Context:
        {context}

        Question:
        {question}
        """
        response = self.llm.invoke(prompt)
        return response.content.strip()

if __name__ == "__main__":
    helper = ICPQueryHelper()
    while True:
        q = input("\n❓ Ask ICP Knowledge Base: ")
        if q.lower() in ["exit", "quit"]:
            break
        answer = helper.query(q)
        print(f"\n💡 Answer: {answer}")
