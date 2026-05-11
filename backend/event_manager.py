import asyncio
from collections import defaultdict

class EventManager:
    def __init__(self):
        self.queues = defaultdict(asyncio.Queue)
        self.results = {}

    async def publish(self, session_id: str, event: dict):
        await self.queues[session_id].put(event)

    async def subscribe(self, session_id: str):
        while True:
            event = await self.queues[session_id].get()
            yield event
            if event.get("type") == "complete":
                break

event_manager = EventManager()
