from agents import cv_agent, job_agent, curriculum_agent, plan_agent
from event_manager import event_manager

async def _publish(session_id, payload):
    await event_manager.publish(session_id, payload)

async def _emit(session_id, agent_name):
    async def emit(node, message):
        await _publish(session_id, {
            "agent": agent_name,
            "node": node,
            "message": message,
            "status": "running",
        })
    return emit

async def run_pipeline(session_id, form_data, pdf_path):
    # ── CV Agent ──────────────────────────────────────────────
    cv_data = await cv_agent.run(pdf_path, await _emit(session_id, "cv_agent"))
    await _publish(session_id, {
        "agent": "cv_agent",
        "status": "done",
        "output": cv_data,
    })

    # ── Job Agent ─────────────────────────────────────────────
    job_data = await job_agent.run(
        cv_data,
        form_data.get("preferences", ""),
        form_data.get("background", ""),
        await _emit(session_id, "job_agent"),
    )
    await _publish(session_id, {
        "agent": "job_agent",
        "status": "done",
        "output": job_data,
    })

    # ── Curriculum Agent ──────────────────────────────────────
    major = (cv_data.get("education") or {}).get("major", "")
    faculty = form_data.get("faculty", "")
    year = form_data.get("year", "")
    top_jobs = ", ".join(job_data.get("top_job_titles", [])[:3])

    skills_needed_set = []
    for job in job_data.get("jobs", [])[:3]:
        ts = job.get("technical_skills")
        if isinstance(ts, list):
            for s in ts:
                if isinstance(s, str) and s not in skills_needed_set:
                    skills_needed_set.append(s)
    skills_needed = ", ".join(skills_needed_set[:15])

    curriculum_question = (
        f"What courses are available"
        f"{f' in the {faculty} faculty' if faculty else ''}"
        f"{f' for year {year} students' if year else ''}"
        f"{f' related to {major}' if major else ''}"
        f"{f' that cover skills like {skills_needed[:200]}' if skills_needed else ''}?"
    )

    curriculum_data = await curriculum_agent.run(
        curriculum_question,
        await _emit(session_id, "curriculum_agent"),
    )
    await _publish(session_id, {
        "agent": "curriculum_agent",
        "status": "done",
        "output": curriculum_data,
    })

    # ── Plan Agent ────────────────────────────────────────────
    plan_curriculum = {"courses": curriculum_data.get("records", [])}

    await _publish(session_id, {
        "agent": "plan_agent",
        "node": "generate_plan",
        "message": "Generating personalized roadmap",
        "status": "running",
    })
    markdown = plan_agent.run(cv_data, job_data, plan_curriculum, form_data)
    await _publish(session_id, {
        "agent": "plan_agent",
        "status": "done",
        "markdown": markdown,
    })

    await _publish(session_id, {"type": "complete"})