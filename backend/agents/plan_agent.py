"""
Plan Agent — LangGraph workflow
================================
Nodes:
  1. analyze_profile   — distil CV into a compact skill/experience summary
  2. analyze_jobs      — extract job patterns and required skills from job listings
  3. select_courses    — match curriculum courses to skill gaps
  4. draft_sections    — write each plan section independently (parallel-ish)
  5. assemble_plan     — merge sections + user preferences into final markdown

Each node receives ONLY the data it needs, avoiding context overload.
"""

import os
import asyncio
import json
import re
from typing import TypedDict, Optional
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

load_dotenv()

# ─────────────────────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────────────────────

_llm = None


def get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview",
            temperature=0.3,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            max_retries=5,
        )
    return _llm


async def _call_llm(system: str, user: str) -> str:
    """Async wrapper around a single LLM call."""
    llm = get_llm()
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    response = await asyncio.to_thread(llm.invoke, messages)
    content = response.content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        ).strip()
    return str(content).strip()


# ─────────────────────────────────────────────────────────────
# Graph State
# ─────────────────────────────────────────────────────────────

class PlanState(TypedDict, total=False):
    # ── raw inputs ──
    cv_data: dict
    job_data: dict
    curriculum_data: dict
    user_data: dict          # form fields: faculty, year, preferences, plan_preferences, background

    # ── intermediate ──
    profile_summary: str     # compact skill/experience paragraph
    job_insights: str        # key job patterns, required skills, seniority
    course_selection: str    # chosen courses with rationale

    # ── drafted sections ──
    section_profile: str
    section_jobs: str
    section_skills: str
    section_courses: str
    section_action_plan: str

    # ── final output ──
    markdown: str
    error: Optional[str]

    # ── emit callback (not serialised by langgraph, injected at runtime) ──
    _emit: object


# ─────────────────────────────────────────────────────────────
# Helper — safe emit
# ─────────────────────────────────────────────────────────────

async def _emit(state: PlanState, node: str, message: str):
    cb = state.get("_emit")
    if cb:
        await cb(node, message)


# ─────────────────────────────────────────────────────────────
# Node 1 — analyze_profile
# ─────────────────────────────────────────────────────────────

async def analyze_profile(state: PlanState) -> PlanState:
    await _emit(state, "analyze_profile", "Analysing CV and skills profile…")

    cv = state["cv_data"]
    edu = cv.get("education") or {}

    tech_skills = cv.get("technical_skills", [])
    soft_skills = cv.get("soft_skills", [])
    experience  = cv.get("experience", [])
    projects    = cv.get("projects", [])
    certs       = cv.get("certifications", [])

    # Compact serialisation — only the fields the LLM needs
    cv_compact = {
        "degree":          edu.get("degree"),
        "major":           edu.get("major"),
        "gpa":             edu.get("gpa"),
        "academic_year":   edu.get("academic_year"),
        "graduation_year": edu.get("graduation_year"),
        "technical_skills": tech_skills[:25],
        "soft_skills":      soft_skills[:12],
        "experience": [
            {
                "company":    e.get("company"),
                "position":   e.get("position"),
                "duration":   e.get("duration"),
                "skills_used": e.get("skills_used", [])[:6],
            }
            for e in experience[:4]
        ],
        "projects": [
            {
                "name":        p.get("name"),
                "skills_used": p.get("skills_used", [])[:6],
            }
            for p in projects[:5]
        ],
        "certifications": certs[:5],
    }

    user = state.get("user_data", {})

    system = (
        "You are a career coach analysing a student's CV. "
        "Produce a concise PROFILE SUMMARY (max 200 words) that covers:\n"
        "1. Education background\n"
        "2. Current technical skill level and breadth\n"
        "3. Notable experience and projects\n"
        "4. Soft skills and certifications\n"
        "5. Overall readiness level (beginner / developing / competent)\n\n"
        "Be specific. Use the student's actual data. No fluff."
    )

    user_msg = (
        f"Student context:\n"
        f"- Faculty: {user.get('faculty', 'Not specified')}\n"
        f"- Academic year: {user.get('year', 'Not specified')}\n"
        f"- Background note: {user.get('background', 'None')}\n\n"
        f"CV data (JSON):\n{json.dumps(cv_compact, ensure_ascii=False, indent=2)}"
    )

    summary = await _call_llm(system, user_msg)
    await _emit(state, "analyze_profile", "Profile summary complete.")
    return {**state, "profile_summary": summary}


# ─────────────────────────────────────────────────────────────
# Node 2 — analyze_jobs
# ─────────────────────────────────────────────────────────────

async def analyze_jobs(state: PlanState) -> PlanState:
    await _emit(state, "analyze_jobs", "Analysing job market requirements…")

    jobs = state["job_data"].get("jobs", [])[:6]   # top 6 listings max

    # Strip heavy fields to keep context small
    jobs_compact = []
    for j in jobs:
        jobs_compact.append({
            "title":               j.get("title"),
            "company":             j.get("company"),
            "seniority":           j.get("seniority"),
            "technical_skills":    (j.get("technical_skills") or [])[:12],
            "requirements":        (j.get("requirements") or [])[:5],
            "years_of_experience": j.get("years_of_experience"),
        })

    user = state.get("user_data", {})

    system = (
        "You are a job market analyst. "
        "Given a list of job listings, produce a concise JOB INSIGHTS report (max 250 words) covering:\n"
        "1. Most common job titles / roles\n"
        "2. Top 10 technical skills required (ranked by frequency)\n"
        "3. Common experience / seniority expectations\n"
        "4. Any notable industry trends visible in the listings\n\n"
        "Output plain text. Be specific and data-driven."
    )

    user_msg = (
        f"Student preferences: {user.get('preferences', 'None')}\n\n"
        f"Job listings (JSON):\n{json.dumps(jobs_compact, ensure_ascii=False, indent=2)}"
    )

    insights = await _call_llm(system, user_msg)
    await _emit(state, "analyze_jobs", "Job insights extracted.")
    return {**state, "job_insights": insights}


# ─────────────────────────────────────────────────────────────
# Node 3 — select_courses
# ─────────────────────────────────────────────────────────────

async def select_courses(state: PlanState) -> PlanState:
    await _emit(state, "select_courses", "Matching curriculum to skill gaps…")

    records = state["curriculum_data"].get("courses", [])[:30]

    # Extract one readable label per record
    course_labels = []
    for r in records:
        if isinstance(r, dict):
            label = next(
                (v for v in r.values() if isinstance(v, str) and 2 < len(v) < 150),
                None,
            )
            if label and label not in course_labels:
                course_labels.append(label)

    current_skills = state["cv_data"].get("technical_skills", [])
    profile_summary = state.get("profile_summary", "")
    job_insights    = state.get("job_insights", "")

    system = (
        "You are a curriculum advisor. "
        "Given a student's current skills, job market requirements, and available courses, "
        "select and rank the TOP 8 most valuable courses for this student. "
        "For each chosen course provide:\n"
        "- Course name\n"
        "- Why it matters (one sentence)\n"
        "- Suggested timing (e.g. 'Month 1-2')\n\n"
        "Output as a numbered list. Be specific and practical."
    )

    user_msg = (
        f"Student current skills: {', '.join(current_skills[:20]) or 'None listed'}\n\n"
        f"Profile summary:\n{profile_summary}\n\n"
        f"Job market requirements:\n{job_insights}\n\n"
        f"Available courses ({len(course_labels)} total):\n"
        + "\n".join(f"- {c}" for c in course_labels)
    )

    selection = await _call_llm(system, user_msg)
    await _emit(state, "select_courses", "Course selection complete.")
    return {**state, "course_selection": selection}


# ─────────────────────────────────────────────────────────────
# Node 4 — draft_sections  (runs 5 sub-calls concurrently)
# ─────────────────────────────────────────────────────────────

async def draft_sections(state: PlanState) -> PlanState:
    await _emit(state, "draft_sections", "Drafting individual roadmap sections…")

    profile_summary = state.get("profile_summary", "")
    job_insights    = state.get("job_insights", "")
    course_selection = state.get("course_selection", "")
    user = state.get("user_data", {})
    cv   = state["cv_data"]

    current_skills  = cv.get("technical_skills", [])[:20]
    top_job_titles  = state["job_data"].get("top_job_titles", [])[:5]
    plan_prefs      = user.get("plan_preferences", "") or user.get("preferences", "") or ""
    academic_year   = user.get("year", "")
    faculty         = user.get("faculty", "")

    # Compute skill gaps
    job_skills: set[str] = set()
    for j in state["job_data"].get("jobs", [])[:5]:
        for s in (j.get("technical_skills") or []):
            job_skills.add(s.lower())
    current_lower = {s.lower() for s in current_skills}
    gaps = sorted(job_skills - current_lower)[:15]

    # ── 5 concurrent section drafts ─────────────────────────

    async def draft_profile_section():
        sys = (
            "You write one section of a career development plan. "
            "Write a STUDENT PROFILE section (max 150 words) in markdown. "
            "Use a heading ## Student Profile. "
            "Summarise who the student is, their strengths, and their starting point."
        )
        msg = (
            f"Faculty: {faculty} | Year: {academic_year}\n\n"
            f"Profile summary:\n{profile_summary}"
        )
        return await _call_llm(sys, msg)

    async def draft_jobs_section():
        sys = (
            "You write one section of a career development plan. "
            "Write a TARGET CAREERS section (max 200 words) in markdown. "
            "Use heading ## Target Careers. "
            "List the recommended job titles with a one-line description each, "
            "then add a short paragraph on market outlook."
        )
        msg = (
            f"Top job titles: {', '.join(top_job_titles)}\n\n"
            f"Job market insights:\n{job_insights}"
        )
        return await _call_llm(sys, msg)

    async def draft_skills_section():
        sys = (
            "You write one section of a career development plan. "
            "Write a SKILLS ANALYSIS section (max 200 words) in markdown. "
            "Use heading ## Skills Analysis. "
            "List current strengths, then clearly highlight skill gaps the student must close."
        )
        msg = (
            f"Current technical skills: {', '.join(current_skills)}\n\n"
            f"Skill gaps identified: {', '.join(gaps) if gaps else 'None significant'}\n\n"
            f"Job market requirements summary:\n{job_insights}"
        )
        return await _call_llm(sys, msg)

    async def draft_courses_section():
        sys = (
            "You write one section of a career development plan. "
            "Write a RECOMMENDED COURSES section in markdown. "
            "Use heading ## Recommended Courses. "
            "Present the course selection clearly with timing guidance. "
            "Keep it practical and specific."
        )
        msg = f"Course selection (already analysed):\n{course_selection}"
        return await _call_llm(sys, msg)

    async def draft_action_plan_section():
        sys = (
            "You write one section of a career development plan. "
            "Write a 6-MONTH ACTION PLAN section in markdown. "
            "Use heading ## 6-Month Action Plan. "
            "Break it into Month 1-2, Month 3-4, Month 5-6. "
            "Each period should have 3-5 concrete, actionable tasks. "
            "Tailor the plan to the student's year, gaps, and preferences. "
            "Be specific — avoid generic advice."
        )
        msg = (
            f"Academic year: {academic_year} | Faculty: {faculty}\n"
            f"Student plan preferences: {plan_prefs or 'None specified'}\n\n"
            f"Profile summary:\n{profile_summary}\n\n"
            f"Skill gaps to close: {', '.join(gaps) if gaps else 'None significant'}\n\n"
            f"Target jobs: {', '.join(top_job_titles)}\n\n"
            f"Selected courses (for timing reference):\n{course_selection}"
        )
        return await _call_llm(sys, msg)

    # Run all 5 section drafts concurrently
    results = await asyncio.gather(
        draft_profile_section(),
        draft_jobs_section(),
        draft_skills_section(),
        draft_courses_section(),
        draft_action_plan_section(),
    )

    await _emit(state, "draft_sections", "All sections drafted.")
    return {
        **state,
        "section_profile":     results[0],
        "section_jobs":        results[1],
        "section_skills":      results[2],
        "section_courses":     results[3],
        "section_action_plan": results[4],
    }


# ─────────────────────────────────────────────────────────────
# Node 5 — assemble_plan
# ─────────────────────────────────────────────────────────────

async def assemble_plan(state: PlanState) -> PlanState:
    await _emit(state, "assemble_plan", "Assembling final career roadmap…")

    user = state.get("user_data", {})
    plan_prefs = user.get("plan_preferences", "") or user.get("preferences", "") or ""

    sections = "\n\n---\n\n".join([
        state.get("section_profile",     ""),
        state.get("section_jobs",        ""),
        state.get("section_skills",      ""),
        state.get("section_courses",     ""),
        state.get("section_action_plan", ""),
    ])

    system = (
        "You are assembling a polished career development plan from pre-written sections. "
        "Your job:\n"
        "1. Combine all sections into one coherent markdown document.\n"
        "2. Add a # Personalized Career Roadmap title at the top.\n"
        "3. Add a short intro paragraph (2-3 sentences) personalised to the student.\n"
        "4. Ensure transitions between sections flow naturally.\n"
        "5. If the student has specific plan preferences, weave them in naturally — "
        "   do NOT just append them at the end.\n"
        "6. Add a ## Key Takeaways section at the end with 3-5 bullet points.\n"
        "7. Keep the total length reasonable (600-900 words).\n\n"
        "Output ONLY the final markdown. No preamble, no code fences."
    )

    user_msg = (
        f"Student plan preferences: {plan_prefs or 'None specified'}\n\n"
        f"Pre-written sections:\n\n{sections}"
    )

    markdown = await _call_llm(system, user_msg)

    # Strip any accidental code fences
    markdown = re.sub(r"^```(?:markdown)?\s*", "", markdown.strip())
    markdown = re.sub(r"\s*```$", "", markdown).strip()

    await _emit(state, "assemble_plan", "Roadmap assembly complete.")
    return {**state, "markdown": markdown}


# ─────────────────────────────────────────────────────────────
# Build LangGraph
# ─────────────────────────────────────────────────────────────

def _build_graph():
    g = StateGraph(PlanState)

    g.add_node("analyze_profile", analyze_profile)
    g.add_node("analyze_jobs",    analyze_jobs)
    g.add_node("select_courses",  select_courses)
    g.add_node("draft_sections",  draft_sections)
    g.add_node("assemble_plan",   assemble_plan)

    # Sequential flow: profile + jobs run first, then course selection,
    # then draft (needs all three), then assemble.
    g.set_entry_point("analyze_profile")
    g.add_edge("analyze_profile", "analyze_jobs")
    g.add_edge("analyze_jobs",    "select_courses")
    g.add_edge("select_courses",  "draft_sections")
    g.add_edge("draft_sections",  "assemble_plan")
    g.add_edge("assemble_plan",   END)

    return g.compile()


_graph = _build_graph()


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

async def run(
    cv_data: dict,
    job_data: dict,
    curriculum_data: dict,
    user_data: dict,
    emit,
) -> str:
    """
    Run the LangGraph plan pipeline.

    Args:
        cv_data:          Output from cv_agent
        job_data:         Output from job_agent
        curriculum_data:  Output from curriculum_agent  (expects key 'courses' or 'records')
        user_data:        Form fields (faculty, year, background, preferences, plan_preferences)
        emit:             Async callback(node: str, message: str) for live updates

    Returns:
        Markdown string of the finished career roadmap.
    """
    # Normalise curriculum_data key — agent_runner passes {courses: records}
    # but curriculum_agent returns {records: [...]}
    if "records" in curriculum_data and "courses" not in curriculum_data:
        curriculum_data = {"courses": curriculum_data["records"], **curriculum_data}

    initial_state: PlanState = {
        "cv_data":          cv_data,
        "job_data":         job_data,
        "curriculum_data":  curriculum_data,
        "user_data":        user_data,
        "_emit":            emit,
    }

    try:
        final_state = await _graph.ainvoke(initial_state)
        return final_state.get("markdown", "# Error\n\nPlan generation failed — no markdown produced.")
    except Exception as e:
        await emit("error", f"Plan agent error: {e}")
        return f"# Error\n\nPlan generation failed: {e}"