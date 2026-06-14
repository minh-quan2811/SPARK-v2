import os
import json
import re
import asyncio
from typing import List, Dict, Any, Callable, Awaitable, Optional, Set, Tuple

import requests
import cohere
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

Emit = Callable[[str, str], Awaitable[None]]

_llm: Optional[ChatGoogleGenerativeAI] = None


def get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            max_retries=5,
        )
    return _llm


async def _call_llm(system: str, user: str) -> str:
    llm = get_llm()
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    response = await asyncio.to_thread(llm.invoke, messages)
    content = response.content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        ).strip()
    return str(content).strip()


_BLOCKED_DOMAINS = {
    "linkedin.com", "glassdoor.com", "indeed.com",
    "careerbuilder.com", "monster.com", "ziprecruiter.com",
    "simplyhired.com",
}


def _domain(url: str) -> str:
    return re.sub(r"https?://(www\.)?", "", url).split("/")[0].lower()


def _is_scrapable(url: str) -> bool:
    return _domain(url) not in _BLOCKED_DOMAINS


def _scrape(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=12)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header",
                          "aside", "form", "noscript", "iframe"]):
            tag.decompose()
        main = (
            soup.find("main")
            or soup.find("article")
            or soup.find(id=re.compile(r"(content|main|job)", re.I))
            or soup.find(class_=re.compile(r"(job|content|detail|desc)", re.I))
            or soup.body
        )
        text = (main or soup).get_text(separator="\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text[:6000] if text else ""
    except Exception as e:
        return f"SCRAPE_ERROR: {e}"


def _tavily_search(query: str, max_results: int = 8) -> List[Dict[str, str]]:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return []
    client = TavilyClient(api_key=key)
    try:
        results = client.search(query=query, max_results=max_results, topic="general")
        return [
            {"url": r["url"], "title": r.get("title", ""), "snippet": r.get("content", "")[:200]}
            for r in results.get("results", [])
        ]
    except Exception:
        return []


def _clean_json(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


_BUILD_QUERIES_SYSTEM = """You are a job search strategist helping a student find jobs on LinkedIn or Indeed.
Generate exactly 3 search queries using three different angles of the student's profile.

A good query sounds like a job title someone would search for — short, role-focused, and natural.
A bad query is a list of technologies or frameworks strung together.

Query 1 — Role + Domain: Use the student's target role (or most recent role if no target is given) and the domain or industry they work in. 3 to 5 words.
Query 2 — Role + Seniority + Specialty: Use the student's experience level (Fresher / Junior / Mid) combined with their single strongest demonstrated specialty — a skill backed by actual project evidence. 3 to 6 words.
Query 3 — Role + Application Area: Use the role name and the type of system, product, or industry the student's dominant skill cluster applies to. 3 to 5 words.

Rules:
- Every query must begin with a role title or job function
- Maximum 2 technology names per query, and only if they define the role (e.g. "iOS Developer Swift" is fine, "Python LangChain FastAPI AWS" is not)
- If preferences include a location, append it to all queries
- If preferences include an industry, work it into the most relevant query only

Return ONLY a valid JSON array of exactly 3 strings. No explanation, no markdown, no extra keys.
Good example: ["AI Engineer NLP applications", "Junior Machine Learning Engineer computer vision", "LLM Developer conversational AI startup"]
Bad example: ["AI Engineer Python LangChain LlamaIndex RAG", "LangGraph FastAPI AWS cloud AI", "PyTorch Transformers deep learning NLP"]"""

_BUILD_QUERIES_HUMAN = """CV Profile:
{cv_json}

Preferences: {preferences}
Background: {background}"""


async def build_queries(cv_data: dict, preferences: str, background: str) -> List[str]:
    raw = await _call_llm(
        _BUILD_QUERIES_SYSTEM,
        _BUILD_QUERIES_HUMAN.format(
            cv_json=json.dumps(cv_data, ensure_ascii=False),
            preferences=preferences or "None",
            background=background or "None",
        ),
    )
    raw = _clean_json(raw)
    queries = json.loads(raw)
    if not isinstance(queries, list):
        raise ValueError(f"Expected list of queries, got: {type(queries)}")
    return queries[:3]


async def _retrieve_for_query(
    query: str,
    seen_urls: Set[str],
    emit: Emit,
    min_pages: int = 3,
    max_candidates: int = 10,
) -> List[Dict[str, str]]:
    await emit("tool_call", f'Searching: "{query}"')
    candidates = await asyncio.to_thread(_tavily_search, query, max_candidates)
    await emit("tool_result", f"{len(candidates)} candidate URLs found")

    pages: List[Dict[str, str]] = []
    for item in candidates:
        url = item["url"]
        if url in seen_urls:
            continue
        if not _is_scrapable(url):
            seen_urls.add(url)
            continue

        seen_urls.add(url)
        await emit("tool_call", f"Scraping: {url[:80]}")
        text = await asyncio.to_thread(_scrape, url)

        if text.startswith("SCRAPE_ERROR") or len(text) < 200:
            await emit("tool_result", f"{_domain(url)} — failed or too short, skipping")
            continue

        await emit("tool_result", f"{_domain(url)} — {len(text):,} chars extracted")
        pages.append({"url": url, "text": text})

        if len(pages) >= min_pages:
            break

    return pages


_EXTRACT_SYSTEM = """You are a job listing extractor. Extract ALL job postings from the web page text.

Return ONLY a valid JSON array. Each element must follow this schema exactly:
{
  "title": "Job Title",
  "company": "Company Name",
  "location": "City, Country or Remote",
  "salary": "salary range as string, or null",
  "technical_skills": ["skill1", "skill2"],
  "requirements": ["requirement1", "requirement2"],
  "responsibilities": ["duty1", "duty2"],
  "years_of_experience": "X years or null",
  "seniority": "Fresher/Junior/Mid/Senior or null",
  "employment_type": "Full-time/Part-time/Contract or null",
  "remote": true or false,
  "apply_url": "direct application URL or null"
}

Rules:
- Only include real job postings with a named company. Skip ads, nav text, and generic content.
- Include all postings found on the page as separate objects.
- Never invent data. Use null for missing fields.
- Return [] if no real jobs are found.
- Return ONLY the JSON array, no explanation, no markdown fences."""


async def _extract_jobs_from_page(url: str, text: str) -> List[Dict[str, Any]]:
    raw = await _call_llm(
        _EXTRACT_SYSTEM,
        f"Page URL: {url}\n\nPage text (truncated):\n{text[:4000]}",
    )
    raw = _clean_json(raw)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return []


def _deduplicate(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[Tuple[str, str]] = set()
    unique: List[Dict[str, Any]] = []
    for job in jobs:
        key = (
            (job.get("title") or "").lower().strip(),
            (job.get("company") or "").lower().strip(),
        )
        if key in seen or key == ("", ""):
            continue
        seen.add(key)
        unique.append(job)
    return unique


def _build_profile_query(cv_data: dict, preferences: str, background: str) -> str:
    edu = cv_data.get("education") or {}
    tech = cv_data.get("technical_skills", [])
    experience = cv_data.get("experience", [])
    projects = cv_data.get("projects", [])

    exp_roles = ", ".join(e.get("position", "") for e in experience[:3] if e.get("position"))
    proj_tools = ", ".join(s for p in projects[:3] for s in (p.get("skills_used") or [])[:4])

    parts = [
        f"Target role: {edu.get('major', '')}" if edu.get("major") else "",
        f"Skills: {', '.join(tech[:20])}" if tech else "",
        f"Experience: {exp_roles}" if exp_roles else "",
        f"Project tools: {proj_tools}" if proj_tools else "",
        f"Preferences: {preferences}" if preferences else "",
        f"Background: {background}" if background else "",
    ]
    return " | ".join(p for p in parts if p)


def _job_to_document(job: Dict[str, Any]) -> str:
    lines = [
        f"Title: {job.get('title', '')}",
        f"Company: {job.get('company', '')}",
        f"Location: {job.get('location', '')}",
        f"Seniority: {job.get('seniority', '')}",
        f"Skills: {', '.join(job.get('technical_skills') or [])}",
        f"Requirements: {', '.join((job.get('requirements') or [])[:5])}",
        f"Responsibilities: {', '.join((job.get('responsibilities') or [])[:5])}",
    ]
    return "\n".join(l for l in lines if not l.endswith(": "))


def _cohere_rerank(jobs: List[Dict[str, Any]], query: str, top_n: int = 10) -> List[Dict[str, Any]]:
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key or not jobs:
        return jobs[:top_n]

    co = cohere.ClientV2(api_key=api_key)
    documents = [_job_to_document(j) for j in jobs]

    try:
        response = co.rerank(
            model="rerank-v3.5",
            query=query,
            documents=documents,
            top_n=min(top_n, len(documents)),
        )
        reranked: List[Dict[str, Any]] = []
        for r in response.results:
            job = jobs[r.index].copy()
            job["relevance_score"] = round(r.relevance_score, 4)
            reranked.append(job)
        return reranked
    except Exception:
        for j in jobs[:top_n]:
            j.setdefault("relevance_score", None)
        return jobs[:top_n]


_SUMMARY_SYSTEM = (
    "You are a career advisor writing a brief Vietnamese-language summary for a student. "
    "Summarise the job search results in 2-3 sentences covering: "
    "how many jobs were found, which roles/companies stand out, and any key skill requirements."
)


async def _generate_summary(jobs: List[Dict[str, Any]], preferences: str) -> str:
    user_msg = (
        f"Student preferences: {preferences or 'None'}\n\n"
        f"Top job results:\n{json.dumps(jobs[:6], ensure_ascii=False, indent=2)}"
    )
    try:
        return await _call_llm(_SUMMARY_SYSTEM, user_msg)
    except Exception:
        return f"Tìm thấy {len(jobs)} vị trí việc làm phù hợp."


async def run(cv_data: dict, preferences: str, background: str, emit: Emit) -> dict:
    await emit("prepare_query", "Building three-dimensional search queries from CV profile…")

    queries = await build_queries(cv_data, preferences, background)
    for i, q in enumerate(queries, 1):
        await emit("prepare_query", f"Query {i}: {q}")

    await emit("run_agent", "Running retrieval loops for all three queries…")

    seen_urls: Set[str] = set()
    all_pages: List[Dict[str, str]] = []

    for i, query in enumerate(queries, 1):
        await emit("tool_call", f"[Query {i}/3] {query}")
        pages = await _retrieve_for_query(query, seen_urls, emit, min_pages=3)
        await emit("tool_result", f"Query {i} collected {len(pages)} pages")
        all_pages.extend(pages)

    if not all_pages:
        await emit("format_results", "No pages could be scraped — returning empty results")
        return {"jobs": [], "summary": "Không tìm thấy kết quả phù hợp.", "top_job_titles": []}

    await emit("format_results", f"Extracting job listings from {len(all_pages)} pages…")

    per_page_results = await asyncio.gather(
        *[_extract_jobs_from_page(p["url"], p["text"]) for p in all_pages],
        return_exceptions=True,
    )

    raw_jobs: List[Dict[str, Any]] = []
    for result in per_page_results:
        if isinstance(result, list):
            raw_jobs.extend(result)

    await emit("format_results", f"{len(raw_jobs)} raw job entries extracted")

    unique_jobs = _deduplicate(raw_jobs)
    await emit("format_results", f"{len(unique_jobs)} unique jobs after deduplication")

    if not unique_jobs:
        await emit("format_results", "No valid job entries found after deduplication")
        return {"jobs": [], "summary": "Không tìm thấy kết quả phù hợp.", "top_job_titles": []}

    await emit("format_results", "Reranking with Cohere cross-encoder…")

    profile_query = _build_profile_query(cv_data, preferences, background)
    ranked_jobs = await asyncio.to_thread(_cohere_rerank, unique_jobs, profile_query, 10)

    await emit("format_results", f"{len(ranked_jobs)} jobs ranked and ready")

    summary = await _generate_summary(ranked_jobs, preferences)

    return {
        "jobs": ranked_jobs,
        "summary": summary,
        "top_job_titles": [j.get("title") for j in ranked_jobs if j.get("title")],
    }


get_llm()