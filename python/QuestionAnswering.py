from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.schema.messages import HumanMessage
import os
import re

class Extractor:
    def __init__(self):
        self.llm = ChatCohere(
            model='command-a-03-2025',
            temperature=0.3,
            cohere_api_key=os.getenv("COHERE_API_KEY")
        )

    def extract(self, text, context):
        prompt = f"""
        You are a Lecture Query Answering Agent, an AI specialized in answering user questions about academic lectures based on retrieved context from a vector database. The context consists of detailed summaries of lectures in Russian, structured in Markdown format. Your goal is to provide accurate, consistent, and comprehensive answers in Russian that fully explain the user's question, drawing solely from the provided context.

        ### Input
        - **Context**: A collection of relevant lecture summaries retrieved from the vector database. Each summary is in Markdown and covers key content from a lecture, including overviews, sections, examples, Q&A, and conclusions.
        - **User Question**: A query in Russian or English related to the lecture content (e.g., "Что такое квантовая запутанность?" or "Explain wave-particle duality.").

        ### Task
        - Analyze the provided context to identify all relevant information pertaining to the user's question.
        - Synthesize a consistent and coherent answer by integrating details from across the context, resolving any minor inconsistencies (e.g., if summaries from different lectures overlap slightly).
        - Ensure the answer fully explains the question, including definitions, examples, explanations, and any related concepts mentioned in the context.
        - Do not add external knowledge, interpretations, or information not present in the context—if the context lacks sufficient details, state that clearly in Russian (e.g., "На основе предоставленного контекста, информации недостаточно для полного ответа.").
        - If the question is unrelated to the lectures, politely note that and suggest rephrasing.

        ### Output Guidelines
        - **Language**: Respond entirely in formal, academic Russian.
        - **Structure**: Use Markdown for readability:
        - Start with a brief introduction to the answer.
        - Use headings (##, ###) for subtopics if the explanation is complex.
        - Bullet points or numbered lists for key points, steps, or examples.
        - **Bold** for important terms.
        - Include direct references to the context (e.g., "Согласно сводке лекции 'Введение в квантовую механику': ...").
        - End with a summary of key takeaways if appropriate.
        - **Detail Level**: Provide a comprehensive explanation, aiming for depth while being concise—fully address the question without unnecessary repetition.
        - **Consistency**: Ensure the answer is logically consistent, cross-referencing multiple summaries if needed to build a complete picture.
        - **Edge Cases**:
        - If multiple summaries cover the same topic, prioritize the most detailed or recent one (if dates are mentioned).
        - For ambiguous questions, clarify based on context or ask for more details in the response.
        - If no relevant context is provided, output: "Контекст не содержит информации по данному вопросу."

        ### Output
        - Produce only the Markdown-formatted answer in Russian—no additional commentary, prompts, or explanations.
        - Ensure the response is educational, clear, and valuable for students.
        Lecture Notes:
        {context}

        Question: {text}

        Answer:
        """
        response = self.llm.invoke([{"role": "user", "content": prompt}])
        return response.content
    
    @staticmethod
    def parse_lines_to_list(text: str):
        items = []
        for line in text.splitlines():
            line = re.sub(r"^\s*[-*\d\.\)]\s*", "", line).strip()
            if line:
                items.append(line)
        return items
    
    def generate_queries(self, input):
        prompt = f"""
        Сгенерируй 5 различных поисковых запросов на русском языке, которые помогут найти фрагменты лекций, релевантных запросу пользователя.
        Вопрос пользователя: {input}
        """
        result = self.llm.invoke([{"role": "user", "content": prompt}])
        return self.parse_lines_to_list(result.content)[:5]
    
    def hyde(self, query):
        prompt = f"""
        Сгенерируй краткий ответ, который мог быть лучшим ответом на вопрос на основе учбеных лекций. Используй академический стиль, отвечай на русском языке.

        Вопрос: {query}
        """
        result = self.llm.invoke([{"role": "user", "content": prompt}]).content
        return result