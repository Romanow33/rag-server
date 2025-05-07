import asyncio
from typing import Dict, List
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

router = APIRouter()


class ConnectionManager:
    def __init__(self):
        self.active: Dict[str, List[asyncio.Queue]] = {}

    async def connect(self, upload_id: str) -> asyncio.Queue:
        q = asyncio.Queue()
        self.active.setdefault(upload_id, []).append(q)
        return q

    def disconnect(self, upload_id: str, q: asyncio.Queue):
        lst = self.active.get(upload_id, [])
        if q in lst:
            lst.remove(q)
        if not lst:
            self.active.pop(upload_id, None)

    async def send(self, upload_id: str, message: str):
        for q in self.active.get(upload_id, []):
            await q.put(message)


manager = ConnectionManager()


@router.get("/notifications/{upload_id}")
async def notifications(upload_id: str):
    q = await manager.connect(upload_id)

    async def event_generator():
        try:
            while True:
                msg = await q.get()
                yield {"data": msg}
                if msg == "completed":
                    break
        except asyncio.CancelledError:
            pass
        finally:
            manager.disconnect(upload_id, q)

    return EventSourceResponse(event_generator(), ping=0)
