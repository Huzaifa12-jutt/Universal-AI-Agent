# ==========================
# rag/chatbot.py
# ==========================

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import settings
from app.rag.prompts import RAG_SYSTEM_PROMPT

GROQ_BASE_URL = settings.GROQ_BASE_URL


class ResearchChatbot:
    """
    Thin wrapper around a Groq-hosted chat model plus a retriever.
    Keeps track of the sources used for the most recent answer so the
    UI can show them for transparency.
    """

    def __init__(self, retriever, api_key, model_name, temperature=0.2):
        self.retriever = retriever
        self.last_sources = []

        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=GROQ_BASE_URL,
            temperature=temperature,
        )

        self.prompt = ChatPromptTemplate.from_template(RAG_SYSTEM_PROMPT)

    def _build_context(self, question):
        docs = self.retriever.invoke(question)
        self.last_sources = docs

        context = "\n\n".join(doc.page_content for doc in docs)
        return context

    def ask(self, question):
        """
        Non-streaming call. Returns the full answer text.
        """

        context = self._build_context(question)

        final_prompt = self.prompt.invoke({
            "context": context,
            "question": question,
        })

        response = self.llm.invoke(final_prompt)

        return response.content

    def stream(self, question):
        """
        Streaming generator. Yields text chunks as they arrive so the
        UI can render the answer progressively.
        """

        context = self._build_context(question)

        final_prompt = self.prompt.invoke({
            "context": context,
            "question": question,
        })

        for chunk in self.llm.stream(final_prompt):
            if chunk.content:
                yield chunk.content


def create_chatbot(retriever, api_key, model_name, temperature=0.2):
    return ResearchChatbot(
        retriever=retriever,
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
    )