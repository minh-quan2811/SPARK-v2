from pydantic import BaseModel

class SubmitResponse(BaseModel):
    session_id: str
