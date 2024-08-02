import json
import os
from channels.generic.websocket import AsyncWebsocketConsumer
from .ai_assistant import AI_Assistant
from django.conf import settings

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self.ai_assistant = AI_Assistant()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        message = json.loads(text_data)
        user_input = message.get('message')

        ai_response, emotion, audio_url, phonemes, lip_sync_file_path = await self.ai_assistant.process_input(user_input)

        lip_sync_file_url = lip_sync_file_path.replace(settings.MEDIA_ROOT, settings.MEDIA_URL)
        lip_sync_file_url = f"{settings.MEDIA_URL}{os.path.basename(lip_sync_file_path)}"
        print(f"Lip Sync File URL: {lip_sync_file_url}")

        response_data = {
            'text': ai_response,
            'emotion': emotion,
            'audio': audio_url,
            'phonemes': phonemes,
            'lipSyncDataUrl': lip_sync_file_url
        }

        await self.send(text_data=json.dumps(response_data))
