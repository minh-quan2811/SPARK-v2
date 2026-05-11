import fitz
import json
import os
import re
import asyncio
from typing import Optional
from pydantic import BaseModel, field_validator
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


class Education(BaseModel):
    degree: Optional[str] = None
    major: Optional[str] = None
    gpa: Optional[float] = None
    graduation_year: Optional[int] = None
    academic_year: Optional[int] = None

    @field_validator("degree", mode="before")
    @classmethod
    def normalize_degree(cls, v):
        if v is None:
            return None
        v_lower = str(v).strip().lower()
        if any(x in v_lower for x in ["engineer", "kỹ sư"]):
            return "Engineer"
        if any(x in v_lower for x in ["bachelor", "cử nhân", "bsc", "b.sc", "b.eng", "undergraduate"]):
            return "Bachelor"
        if any(x in v_lower for x in ["master", "thạc sĩ", "msc", "m.sc"]):
            return "Master"
        if any(x in v_lower for x in ["phd", "doctorate", "tiến sĩ"]):
            return "PhD"
        return str(v).strip()


class Experience(BaseModel):
    company: Optional[str] = None
    position: Optional[str] = None
    duration: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None
    skills_used: list[str] = []


class Project(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    skills_used: list[str] = []


class CVSchema(BaseModel):
    education: Optional[Education] = None
    experience: list[Experience] = []
    technical_skills: list[str] = []
    soft_skills: list[str] = []
    projects: list[Project] = []
    certifications: list[str] = []


def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=5,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


def extract_pdf_text(cv_path: str) -> str:
    doc = fitz.open(cv_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return re.sub(r'\n{3,}', '\n\n', text).strip()


EXTRACTION_PROMPT = """
You are an expert CV parser. Extract ALL information from the CV text below into a single JSON object.

Return ONLY valid JSON — no explanation, no markdown, no code fences.

Required JSON structure:
{{
  "education": {{
    "degree": "Bachelor" | "Engineer" | null,
    "major": "string or null",
    "gpa": number or null,
    "graduation_year": integer or null,
    "academic_year": integer or null
  }},
  "experience": [
    {{
      "company": "string",
      "position": "string",
      "duration": "string",
      "start_date": "string or null",
      "end_date": "string or null",
      "description": "string",
      "skills_used": ["string"]
    }}
  ],
  "technical_skills": ["string"],
  "soft_skills": ["string"],
  "projects": [
    {{
      "name": "string",
      "description": "string",
      "skills_used": ["string"]
    }}
  ],
  "certifications": ["string"]
}}

=== Field-by-field rules ===

EDUCATION:
- degree: Return exactly one of "Bachelor", "Engineer", "Master", "PhD", or null.
  "Bachelor" → B.Sc, B.Eng, Cử nhân, undergraduate.
  "Engineer" → Kỹ sư, 5-year Vietnamese engineering program.
  Null if no degree mentioned.
- major: Field of study as written (e.g. "Electronic and Communication Engineering").
- gpa: Decimal number as-is (e.g. 3.55). Null if not found.
- graduation_year: 4-digit integer. Look for "2025", "graduating 2025", "dự kiến tốt nghiệp 2025". Null if not found.
- academic_year: Current year of study as integer 1–4. Look for "4th year", "năm 4", "năm thứ 4", "year 3". Null if not found.

EXPERIENCE (formal jobs and internships only):
- Only include roles with a named employer/company/institution.
- Do NOT put personal or academic projects here — those go in "projects".
- company: Full name of employer (e.g. "University of Science and Technology - DUT").
- position: Job title exactly as written.
- duration: Date range as written (e.g. "2023 - 2024").
- start_date / end_date: Year or month-year string. Null if not found.
- description: 2–3 sentences summarizing responsibilities and achievements.
- skills_used: Only skills explicitly mentioned for this role.

TECHNICAL_SKILLS:
- All programming languages, tools, frameworks, hardware, protocols, software explicitly listed.
- If "C/C++" appears, split into "C" and "C++" as separate entries.
- No duplicates.

SOFT_SKILLS:
- Interpersonal and professional skills explicitly named or clearly described.
- Include skills described in experience (e.g. "mentored students" → "Mentoring").
- Examples: "English language skills", "Mentoring", "Academic support", "Leadership".

PROJECTS (personal, academic, side projects):
- Only include self-built or academic projects — not formal job roles.
- name: Project title as written.
- description: 1–2 sentences on what was built or accomplished.
- skills_used: Technologies explicitly mentioned for this project only.

CERTIFICATIONS:
- Certificates, competition prizes, awards, honors (e.g. "Maze Runner 2025").
- Do NOT include university degrees here.
- Copy names exactly as written. Return [] if none found.

=== CV Text ===
{cv_text}
"""


def parse_llm_response(response) -> str:
    content = response.content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block["text"].strip()
            if isinstance(block, str):
                return block.strip()
        return ""
    return content.strip()


def clean_json_string(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def validate_cv_output(data: dict) -> dict:
    try:
        edu_data = data.get("education") or {}
        education = Education(**edu_data).model_dump()
    except Exception:
        education = Education().model_dump()

    validated_experience = []
    for exp in data.get("experience", []):
        try:
            validated_experience.append(Experience(**exp).model_dump())
        except Exception:
            validated_experience.append(Experience().model_dump())

    validated_projects = []
    for proj in data.get("projects", []):
        try:
            validated_projects.append(Project(**proj).model_dump())
        except Exception:
            validated_projects.append(Project().model_dump())

    return CVSchema(
        education=education,
        experience=validated_experience,
        technical_skills=data.get("technical_skills", []),
        soft_skills=data.get("soft_skills", []),
        projects=validated_projects,
        certifications=data.get("certifications", []),
    ).model_dump()


async def run(pdf_path: str, emit):
    """
    Extract structured data from CV PDF using single LLM call.
    
    Args:
        pdf_path: Path to the PDF file
        emit: Async callback function to emit progress updates
        
    Returns:
        dict with extracted CV data
    """
    await emit("extract_text", "Reading PDF file")
    cv_text = await asyncio.to_thread(extract_pdf_text, pdf_path)
    
    await emit("llm_extraction", "Extracting CV data with AI")
    llm = get_llm()
    prompt = EXTRACTION_PROMPT.format(cv_text=cv_text)
    response = await asyncio.to_thread(llm.invoke, prompt)
    raw = clean_json_string(parse_llm_response(response))
    
    await emit("parse_json", "Parsing JSON response")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        await emit("json_error", f"JSON parse error: {e}")
        return {}
    
    await emit("validate_output", "Validating data structure")
    result = validate_cv_output(data)
    
    return result