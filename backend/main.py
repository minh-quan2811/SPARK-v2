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

# Pending pipelines: session_id → (form_data, pdf_path)
# The pipeline starts only when the SSE consumer connects, eliminating the
# race condition where events are published before anyone is listening.
_pending: dict[str, tuple[dict, str]] = {}


@app.on_event("startup")
async def startup():
    print("Warming up agents...")
    from agents import cv_agent, job_agent, curriculum_agent, plan_agent
    await asyncio.to_thread(cv_agent.get_llm)
    await asyncio.to_thread(job_agent.get_llm)
    await asyncio.to_thread(curriculum_agent.get_llm)
    await asyncio.to_thread(curriculum_agent.get_graph)
    await asyncio.to_thread(plan_agent.get_llm)
    print("All agents initialized")


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

    # Store for when SSE consumer connects — don't start yet
    _pending[session_id] = (form_data, pdf_path)

    return {"session_id": session_id}


@app.get("/api/stream/{session_id}")
async def stream(session_id: str):
    async def event_generator():
        # Start the pipeline now that a consumer is listening
        entry = _pending.pop(session_id, None)
        if entry:
            form_data, pdf_path = entry
            asyncio.create_task(run_pipeline(session_id, form_data, pdf_path))

        async for event in event_manager.subscribe(session_id):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")