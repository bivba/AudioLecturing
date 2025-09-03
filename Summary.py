from google import genai
from google.genai import types
import os
import asyncio

class Summariser:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    def summarize_text(self, text):
        try:
            response =  self.client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[text],
                config=types.GenerateContentConfig(
                    system_instruction="You are an expert academic summarizer. Your task is to transform lecture transcripts "
                    "into clear, well-structured summaries written in Markdown. Always follow these rules:\n\n"
                    "1. Write in plain, student-friendly language without jargon unless necessary.\n"
                    "2. Structure the summary with Markdown elements:\n"
                    "   - Use #, ##, ### for headings and subheadings.\n"
                    "   - Use bullet points or numbered lists for key ideas.\n"
                    "   - Use **bold** for important terms and definitions.\n"
                    "3. Remove filler words, repetitions, and irrelevant tangents.\n"
                    "4. Emphasize main concepts, examples, and takeaways from the lecture.\n"
                    "5. Keep the flow logical, grouping related ideas together.\n"
                    "6. Where useful, provide short summaries at the end of sections.\n"
                    "7. Output must always be valid Markdown and easy to read.\n\n"
                    "Your goal: produce a concise, accurate, and educational summary "
                    "that helps students quickly understand and review the lecture material."
                )
            )
        except Exception as e:
            print(f"Error occurred while summarizing text: {e}")

        return response.text