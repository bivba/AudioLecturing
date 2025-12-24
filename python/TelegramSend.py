import os
import asyncio
import telegram
import re
from telegram import BotCommand
from io import BytesIO
from vector_db import get_vector_db
from Summary import Summariser
from QuestionAnswering import Extractor
from langchain.schema import Document
from langchain_cohere import CohereRerank

vector_db = get_vector_db()

class TelegramBot:
    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.id = os.getenv("TELEGRAM_CHAT_ID")
        self.summariser = Summariser()
        self.extractor = Extractor()
        self.reranker = CohereRerank(
            model='rerank-v3.5',
            top_n=6, 
            cohere_api_key=os.getenv("COHERE_API_KEY")
        )

        if not self.bot_token or not self.id:
            print("Warning: Telegram credentials not set. Notifier will be disabled.")
            self.bot = None
        else:
            self.bot = telegram.Bot(token=self.bot_token)
            print("Telegram Notifier initialized successfully.")
    
    async def setup_commands(self):
        try:
            await self.bot.set_my_commands([
                BotCommand("ask", "Задайте вопрос по лекциям"),
            ])
            print("Bot commands set successfully.")
        except Exception as e:
            print(f"Failed to set commands: {e}")
    
    async def send_message(self, text, filename):
        try:
            file_to_send = BytesIO(text.encode('utf-8'))
            file_to_send.name = filename
            id = os.getenv("TELEGRAM_CHANNEL")
            await self.bot.send_document(chat_id=id, document=file_to_send, parse_mode='Markdown')
            print("Successfully sent transcription to Telegram.")
        except Exception as e:
            print(f"Error sending message to Telegram: {e}")

    async def send_text_message(self, chat_id, text):
        try:
            if not isinstance(text, str):
                if hasattr(text, "content"):
                    text = text.content
                else:
                    text = str(text)

            markdown_text = escape_markdown_v2(text)
            await self.bot.send_message(chat_id=chat_id, text=markdown_text, parse_mode='MarkdownV2')
            print("Successfully sent text message to Telegram.")
        except Exception as e:
            print(f"Error sending text message to Telegram: {e}")


    def handle_query(self, query):
        docs = self.enhance_query(query)

        top_docs = self.reranker.compress_documents(docs, query=query)

        context = "\n\n----\n\n".join([doc.page_content for doc in top_docs])
        response = self.extractor.extract(query, context)

        return response.content if hasattr(response, "content") else str(response)
    

    def enhance_query(self, query):
        expand = self.extractor.generate_queries(query)
        candidates = set()

        for q in [query] + expand:
            for chunk in vector_db.search(q, k=6):
                candidates.add(chunk)
        for chunk in vector_db.search(self.extractor.hyde(query), k=6):
            candidates.add(chunk)

        if not candidates:
            return "No relevant information found in the lecture notes."
        
        docs = [Document(page_content=t) for t in candidates]

        return docs


    async def poll_messages(self):
        if not self.bot:
            return
        
        await self.setup_commands()

        last_update_id = 0
        ask_mode = {}
        while True:
            try:
                updates = await self.bot.get_updates(offset=last_update_id + 1, timeout=30)
                for update in updates:
                    if update.message and update.message.text:
                        user_id = update.message.from_user.id
                        query = update.message.text.strip()
                        if query == '/ask':
                            ask_mode[user_id] = True
                            await self.send_text_message(user_id, "Please type your question:")
                        elif query.startswith('/'):
                            continue
                        elif ask_mode.get(user_id, False):
                            print(f"Received question from {user_id}: {query}")
                            answer = self.handle_query(query)
                            await self.send_text_message(user_id, answer)
                            ask_mode[user_id] = False
                        else:
                            pass
                    last_update_id = update.update_id
            except Exception as e:
                error_msg = str(e)
                if "Conflict" in error_msg:
                    print("Conflict detected: Another bot instance is running. Stopping this polling.")
                    break
                else:
                    print(f"Error polling messages: {e}")
                    await asyncio.sleep(5)



def escape_markdown_v2(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)

    return re.sub(r'([\[\]\(\)>#+\-=|{}\.!])', r'\\\1', text)
