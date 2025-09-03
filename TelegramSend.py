import os
import asyncio
import telegram
from io import BytesIO

class TelegramBot:
    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.id = os.getenv("TELEGRAM_CHAT_ID")

        if not self.bot_token or not self.id:
            print("Warning: Telegram credentials not set. Notifier will be disabled.")
            self.bot = None
        else:
            self.bot = telegram.Bot(token=self.bot_token)
            print("Telegram Notifier initialized successfully.")

    async def send_message(self, text, filename):
        try:
            file_to_send = BytesIO(text.encode('utf-8'))
            file_to_send.name = filename
            await self.bot.send_document(chat_id=self.id, document=file_to_send, parse_mode='Markdown')
            print("Successfully sent transcription to Telegram.")
        except Exception as e:
            print(f"Error sending message to Telegram: {e}")
