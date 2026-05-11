import os, uuid, shutil, asyncio, json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from agent_runner import run_pipeline
from event_manager import event_manager

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/api/submit")
async def submit(
    background: str = Form(""),
    preferences: str = Form(""),
    plan_preferences: str = Form(""),
    faculty: str = Form(""),
    year: str = Form(""),
    cv_file: UploadFile = File(...)
):
    session_id = str(uuid.uuid4())
    pdf_path = os.path.join(UPLOAD_DIR, f"{session_id}.pdf")
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(cv_file.file, f)

    form_data = {
        "background": background,
        "preferences": preferences,
        "plan_preferences": plan_preferences,
        "faculty": faculty,
        "year": year,
    }

    asyncio.create_task(run_pipeline(session_id, form_data, pdf_path))
    await asyncio.sleep(0)
    return {"session_id": session_id}

@app.get("/api/stream/{session_id}")
async def stream(session_id: str):
    async def event_generator():
        async for event in event_manager.subscribe(session_id):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")
