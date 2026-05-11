import os
import json
import re
import asyncio
from typing import List, Dict, Any, Callable, Awaitable
import requests
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────

Emit = Callable[[str, str], Awaitable[None]]

# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────

_llm = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            max_retries=5,
        )
    return _llm


# ─────────────────────────────────────────────
# Tool implementations (sync, run in threads)
# ─────────────────────────────────────────────

def _do_search_jobs(query: str) -> str:
    try:
        tavily_key = os.getenv("TAVILY_API_KEY")
        if not tavily_key:
            return json.dumps({"error": "TAVILY_API_KEY not set"})
        client = TavilyClient(api_key=tavily_key)
        results = client.search(query=query, max_results=10, topic="general")
        output = []
        for r in results.get("results", []):
            output.append({
                "url": r["url"],
                "title": r.get("title", ""),
                "snippet": r.get("content", "")[:300],
            })
        return json.dumps(output, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _do_scrape_page(url: str) -> str:
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
        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
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
        return text[:6000] if text else "No readable content found."
    except Exception as e:
        return f"Scrape error: {e}"


# ─────────────────────────────────────────────
# LangChain tool schemas (for bind_tools)
# ─────────────────────────────────────────────

@tool
def search_jobs(query: str) -> str:
    """Search the web for job listings. Returns a JSON list of URLs and snippets."""
    return _do_search_jobs(query)

@tool
def scrape_page(url: str) -> str:
    """Scrape a webpage and return its cleaned text content."""
    return _do_scrape_page(url)

TOOL_SCHEMAS = [search_jobs, scrape_page]

# ─────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a job search agent. Your goal is to find relevant job listings and extract structured data from them.

WORKFLOW:
1. Call `search_jobs` with a good search query based on the user's profile.
2. Review the returned URLs and snippets. Pick 4-6 URLs that look like actual job listings or job boards.
3. Call `scrape_page` on each chosen URL to get the full content.
4. If a page returns very little content or an error, try another URL.
5. Once you have at least 3-5 real job listings, extract structured data and respond.

OUTPUT FORMAT:
When done, respond with ONLY a valid JSON object — no markdown, no explanation:
{
  "jobs": [
    {
      "title": "Job Title",
      "company": "Company Name",
      "location": "City, Country",
      "salary": "salary range or null",
      "technical_skills": ["skill1", "skill2"],
      "requirements": ["requirement1"],
      "responsibilities": ["duty1"],
      "years_of_experience": "X years or null",
      "seniority": "Junior/Mid/Senior or null",
      "employment_type": "Full-time/Part-time/Contract or null",
      "remote": true or false,
      "apply_url": "url or null"
    }
  ],
  "summary": "2-3 sentence summary in Vietnamese of what was found"
}

RULES:
- Only include real jobs with a real company name. Skip generic or duplicate listings.
- If a page has multiple jobs, extract all of them.
- Do not invent data. Use null for missing fields.
- Summary must be in Vietnamese.
"""


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b if isinstance(b, str)
            else b.get("text", "") if isinstance(b, dict) and b.get("type") == "text"
            else ""
            for b in content
        )
    return str(content)


def _parse_json_response(raw: str) -> dict:
    cleaned = re.sub(r"^```json\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    return json.loads(cleaned)


def _summarise_search(raw: str) -> str:
    try:
        items = json.loads(raw)
        if isinstance(items, list) and items:
            titles = [i.get("title", "") for i in items[:5] if i.get("title")]
            return f"{len(items)} results — " + " · ".join(t[:55] for t in titles[:3])
    except Exception:
        pass
    return raw[:120]


def _summarise_scrape(url: str, raw: str) -> str:
    domain = re.sub(r"https?://(www\.)?", "", url).split("/")[0]
    if "Scrape error" in raw:
        return f"{domain} — {raw[:100]}"
    lines = [l.strip() for l in raw.splitlines() if len(l.strip()) > 20][:3]
    preview = " · ".join(l[:55] for l in lines)
    return f"{domain} — {len(raw):,} chars — \"{preview[:120]}...\""


# ─────────────────────────────────────────────
# Manual async agent loop
# ─────────────────────────────────────────────

async def _run_agent_loop(messages: List, emit: Emit, max_iterations: int = 14) -> List:
    """
    Runs the LLM → tools → LLM loop manually.

    Key design: each tool call is individually awaited so we can emit
    events *as each one starts and finishes*, not after the whole batch.
    The LLM invocation is blocking so it runs in asyncio.to_thread().
    Tool calls (HTTP) also run in threads so they don't block the event loop.
    """
    llm_with_tools = get_llm().bind_tools(TOOL_SCHEMAS)

    for _iteration in range(max_iterations):
        # ── Ask the LLM what to do next ───────────────────────
        full_messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        response: AIMessage = await asyncio.to_thread(
            llm_with_tools.invoke, full_messages
        )
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None) or []

        # No tool calls → LLM has produced the final JSON answer
        if not tool_calls:
            return messages

        # ── Execute each tool call, one at a time ─────────────
        # (The LLM may batch multiple calls in one turn; we run them serially
        #  so the emit stream is ordered and readable.)
        for tc in tool_calls:
            name    = tc.get("name", "")
            args    = tc.get("args", {})
            call_id = tc.get("id", "")

            if name == "search_jobs":
                query = args.get("query", "")
                # Emit BEFORE the network call
                await emit("tool_call", f'Searching web: "{query}"')
                raw = await asyncio.to_thread(_do_search_jobs, query)
                await emit("tool_result", _summarise_search(raw))

            elif name == "scrape_page":
                url = args.get("url", "")
                short = (url[:88] + "…") if len(url) > 88 else url
                # Emit BEFORE the network call
                await emit("tool_call", f"Scraping: {short}")
                raw = await asyncio.to_thread(_do_scrape_page, url)
                await emit("tool_result", _summarise_scrape(url, raw))

            else:
                await emit("tool_call", f"Tool: {name}({json.dumps(args)[:80]})")
                raw = f"Unknown tool: {name}"
                await emit("tool_result", raw[:120])

            # Feed result back into the conversation
            messages.append(ToolMessage(content=str(raw), tool_call_id=call_id))

    return messages


# ─────────────────────────────────────────────
# Public run()
# ─────────────────────────────────────────────

async def run(cv_data: dict, preferences: str, background: str, emit: Emit) -> dict:
    """
    Search for jobs based on CV and preferences.
    Live-streams every search query and scrape as they happen.
    """
    await emit("prepare_query", "Building search query from your profile")

    education  = cv_data.get("education") or {}
    skills     = cv_data.get("technical_skills", [])
    major      = education.get("major", "")
    skills_str = ", ".join(skills[:8])

    parts = [p for p in [major, preferences, background] if p]
    topic = " ".join(parts) if parts else "entry level jobs"

    await emit("prepare_query",
               f"Target roles: {topic[:80]} | Key skills: {skills_str or 'none'}")

    user_message = (
        f"Find job listings for someone with this profile:\n"
        f"- Major / field: {major or 'Not specified'}\n"
        f"- Technical skills: {skills_str or 'Not specified'}\n"
        f"- Preferences: {preferences or 'None'}\n"
        f"- Background: {background or 'None'}\n\n"
        f"Search for: {topic} jobs Vietnam\n"
        f"Focus on Vietnam-based roles or remote roles open to Vietnam candidates."
    )

    messages: List = [HumanMessage(content=user_message)]

    await emit("run_agent", "Agent started — searching and scraping job listings live")

    # The loop emits events as each tool call fires
    messages = await _run_agent_loop(messages, emit)

    await emit("format_results", "Parsing extracted job listings…")

    # ── Find the final JSON answer in the message history ─────
    jobs: List[Dict[str, Any]] = []
    summary = ""

    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        raw = _extract_text(msg.content).strip()
        if not raw:
            continue
        try:
            data = _parse_json_response(raw)
            if "jobs" in data:
                jobs    = data.get("jobs", [])
                summary = data.get("summary", "")
                break
        except Exception:
            continue

    count = len(jobs)
    await emit("format_results",
               f"{count} job listing{'s' if count != 1 else ''} structured and ready")

    return {
        "jobs": jobs,
        "summary": summary,
        "top_job_titles": [j.get("title") for j in jobs[:10] if j.get("title")],
    }

get_llm()
