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
    cv_data = await cv_agent.run(pdf_path, await _emit(session_id, "cv_agent"))
    await _publish(session_id, {
        "agent": "cv_agent",
        "status": "done",
        "output": cv_data,
    })

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

    # Derive current_semester from academic year (year 1 → semester 1, year 2 → semester 3, etc.)
    academic_year = int(form_data.get("year", 1) or 1)
    current_semester = (academic_year - 1) * 2 + 1

    curriculum_data = await curriculum_agent.run(
        await _emit(session_id, "curriculum_agent"),
        program=form_data.get("faculty", ""),
        current_semester=current_semester,
        plan_preferences=form_data.get("plan_preferences", ""),
    )
    await _publish(session_id, {
        "agent": "curriculum_agent",
        "status": "done",
        "output": curriculum_data,
    })

    plan_curriculum = {"courses": curriculum_data.get("database_records", [])}

    plan_emit = await _emit(session_id, "plan_agent")

    markdown = await plan_agent.run(
        cv_data,
        job_data,
        plan_curriculum,
        form_data,
        plan_emit,
    )

    await _publish(session_id, {
        "agent": "plan_agent",
        "status": "done",
        "markdown": markdown,
    })

    await _publish(session_id, {"type": "complete"})