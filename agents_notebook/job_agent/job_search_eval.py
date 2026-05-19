"""
RAGAS Evaluation for Job Search Agent (Tool-based Implementation)
=================================================================
Evaluates the agent that uses search_jobs and scrape_page as LLM tools.

Metrics (no ground truth required):
  - answer_relevancy   : Is the final answer relevant to the query?
  - faithfulness       : Is the answer grounded in the scraped content?

Install:
    pip install ragas langchain-google-genai datasets python-dotenv tavily-python
    pip install langchain-core beautifulsoup4 requests tqdm
"""

# ══════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════

import os
import json
import re
import requests
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

from bs4 import BeautifulSoup
from tavily import TavilyClient
from tqdm import tqdm

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import AnswerRelevancy, Faithfulness

load_dotenv()


# ══════════════════════════════════════════════════════════════
# API KEYS
# ══════════════════════════════════════════════════════════════

def load_api_keys():
    google_key = os.getenv("GOOGLE_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not google_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    if not tavily_key:
        raise ValueError("TAVILY_API_KEY not found in environment variables")
    return google_key, tavily_key

GOOGLE_API_KEY, TAVILY_API_KEY = load_api_keys()
print("✓ API keys loaded successfully")


# ══════════════════════════════════════════════════════════════
# AGENT LLM
# ══════════════════════════════════════════════════════════════

_llm = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview",
            temperature=0,
            google_api_key=GOOGLE_API_KEY,
            max_retries=5,
        )
    return _llm


# ══════════════════════════════════════════════════════════════
# TOOL IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════

def _do_search_jobs(query: str) -> str:
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
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


@tool
def search_jobs(query: str) -> str:
    """Search the web for job listings. Returns a JSON list of URLs and snippets."""
    return _do_search_jobs(query)

@tool
def scrape_page(url: str) -> str:
    """Scrape a webpage and return its cleaned text content."""
    return _do_scrape_page(url)

TOOL_SCHEMAS = [search_jobs, scrape_page]


# ══════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════
# AGENT LOOP
# ══════════════════════════════════════════════════════════════

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


def run_agent_loop(messages: List, max_iterations: int = 14) -> tuple[List, List[str]]:
    """
    Runs the LLM → tools → LLM loop manually.
    
    Returns:
        (final_messages, contexts) where contexts is list of scraped content
    """
    llm_with_tools = get_llm().bind_tools(TOOL_SCHEMAS)
    contexts = []

    for _iteration in range(max_iterations):
        full_messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        response: AIMessage = llm_with_tools.invoke(full_messages)
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None) or []

        if not tool_calls:
            return messages, contexts

        for tc in tool_calls:
            name = tc.get("name", "")
            args = tc.get("args", {})
            call_id = tc.get("id", "")

            if name == "search_jobs":
                query = args.get("query", "")
                raw = _do_search_jobs(query)
            elif name == "scrape_page":
                url = args.get("url", "")
                raw = _do_scrape_page(url)
                # Collect scraped content as context
                if "Scrape error" not in raw and len(raw) > 100:
                    contexts.append(raw[:3000])
            else:
                raw = f"Unknown tool: {name}"

            messages.append(ToolMessage(content=str(raw), tool_call_id=call_id))

    return messages, contexts


# ══════════════════════════════════════════════════════════════
# TEST CASES
# ══════════════════════════════════════════════════════════════

TEST_CASES: List[Dict[str, Any]] = [
    # ── Software ─────────────────────────────────────────────
    {
        "query": "Find Software Engineer jobs in Da Nang",
        "user_skills": ["Python", "FastAPI", "PostgreSQL"],
    },
    # {
    #     "query": "Python developer remote jobs Vietnam",
    #     "user_skills": ["Python", "Django", "REST APIs"],
    # },
    # {
    #     "query": "Data Analyst jobs Ho Chi Minh City entry level",
    #     "user_skills": ["SQL", "Excel", "Power BI"],
    # },
    # {
    #     "query": "DevOps Engineer jobs Hanoi",
    #     "user_skills": ["Docker", "Kubernetes", "AWS", "CI/CD"],
    # },
    # {
    #     "query": "Frontend React developer part-time Vietnam",
    #     "user_skills": ["React", "TypeScript", "CSS"],
    # },

    # ── Finance & Accounting ─────────────────────────────────
    # {
    #     "query": "Accountant jobs Ho Chi Minh City",
    #     "user_skills": ["Excel", "QuickBooks", "Tax Reporting", "GAAP"],
    # },
    # {
    #     "query": "Financial Analyst jobs Hanoi senior level",
    #     "user_skills": ["Financial Modeling", "Excel", "Bloomberg", "CFA"],
    # },
    # {
    #     "query": "Auditor jobs Vietnam Big Four",
    #     "user_skills": ["IFRS", "Risk Assessment", "Excel", "Audit Planning"],
    # },
    # {
    #     "query": "Banking Relationship Manager jobs Ho Chi Minh City",
    #     "user_skills": ["Credit Analysis", "KYC", "Sales", "Customer Relationship"],
    # },

    # # ── Healthcare ───────────────────────────────────────────
    # {
    #     "query": "Nurse jobs Ho Chi Minh City hospital",
    #     "user_skills": ["Patient Care", "IV Therapy", "Medical Records", "CPR"],
    # },
    # {
    #     "query": "Pharmacist jobs Da Nang",
    #     "user_skills": ["Drug Dispensing", "Patient Counseling", "Inventory Management"],
    # },
    # {
    #     "query": "Medical Doctor jobs Hanoi private clinic",
    #     "user_skills": ["Diagnosis", "Patient Care", "Medical Imaging", "English"],
    # },

    # # ── Education ────────────────────────────────────────────
    # {
    #     "query": "English teacher jobs Vietnam international school",
    #     "user_skills": ["IELTS", "TEFL", "Curriculum Design", "Classroom Management"],
    # },
    # {
    #     "query": "Corporate Trainer jobs Ho Chi Minh City",
    #     "user_skills": ["Training Design", "Facilitation", "PowerPoint", "Leadership Coaching"],
    # },

    # # ── Hospitality ──────────────────────────────────────────
    # {
    #     "query": "Hotel Manager jobs Da Nang 5 star",
    #     "user_skills": ["Hotel Operations", "Revenue Management", "English", "Leadership"],
    # },
    # {
    #     "query": "Chef jobs Ho Chi Minh City restaurant",
    #     "user_skills": ["Menu Planning", "Food Safety", "Kitchen Management", "French Cuisine"],
    # },

    # # ── Logistics ────────────────────────────────────────────
    # {
    #     "query": "Supply Chain Manager jobs Ho Chi Minh City",
    #     "user_skills": ["SAP", "Procurement", "Inventory Management", "Negotiation"],
    # },
    # {
    #     "query": "Import Export Specialist jobs Vietnam",
    #     "user_skills": ["Customs Clearance", "Incoterms", "English", "ERP"],
    # },

    # # ── HR & Admin ───────────────────────────────────────────
    # {
    #     "query": "Human Resources Manager jobs Hanoi",
    #     "user_skills": ["Recruitment", "Labor Law", "Payroll", "Performance Management"],
    # },

    # # ── Construction ─────────────────────────────────────────
    # {
    #     "query": "Civil Engineer jobs Ho Chi Minh City construction",
    #     "user_skills": ["AutoCAD", "Project Management", "Structural Analysis", "MS Project"],
    # },
]


# ══════════════════════════════════════════════════════════════
# EVALUATION RUNNER
# ══════════════════════════════════════════════════════════════

def run_agent_for_eval(test_case: Dict[str, Any], outer_bar: tqdm) -> Dict[str, Any]:
    """Run one test case and collect contexts."""
    
    education = ""
    skills = test_case.get("user_skills", [])
    skills_str = ", ".join(skills[:8])
    
    user_message = (
        f"Find job listings for someone with this profile:\n"
        f"- Technical skills: {skills_str or 'Not specified'}\n\n"
        f"Search for: {test_case['query']}\n"
        f"Focus on Vietnam-based roles or remote roles open to Vietnam candidates."
    )
    
    messages = [HumanMessage(content=user_message)]
    
    try:
        messages, contexts = run_agent_loop(messages)
    except Exception as e:
        tqdm.write(f"  ⚠ Agent error: {e}")
        contexts = []
    
    # Extract final answer
    answer = ""
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        raw = _extract_text(msg.content).strip()
        if not raw:
            continue
        try:
            data = _parse_json_response(raw)
            if "jobs" in data:
                answer = data.get("summary", "")
                jobs_count = len(data.get("jobs", []))
                outer_bar.set_postfix_str(f"{jobs_count} jobs found")
                break
        except Exception:
            continue
    
    if not contexts:
        contexts = ["No context retrieved."]
    
    return {
        "question": test_case["query"],
        "answer": answer or "No answer generated.",
        "contexts": contexts,
    }


def build_dataset(test_cases: List[Dict[str, Any]]) -> Dataset:
    samples = {"question": [], "answer": [], "contexts": []}

    print(f"\n📦 Running agent on {len(test_cases)} test cases ...\n")

    with tqdm(
        total=len(test_cases),
        desc="Test cases",
        unit="query",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    ) as outer_bar:

        for tc in test_cases:
            outer_bar.set_description(f"Query: {tc['query'][:45]:<45}")
            try:
                s = run_agent_for_eval(tc, outer_bar)
                samples["question"].append(s["question"])
                samples["answer"].append(s["answer"])
                samples["contexts"].append(s["contexts"])
            except Exception as e:
                tqdm.write(f"  [ERROR] Failed on '{tc['query']}': {e}")
            outer_bar.update(1)

    return Dataset.from_dict(samples)


def run_evaluation(test_cases: List[Dict[str, Any]] = TEST_CASES) -> Dict[str, float]:

    print("\n🔧 Setting up RAGAS with Gemini ...")

    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    ragas_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview",
            temperature=0,
            google_api_key=GOOGLE_API_KEY,
        )
    )

    ragas_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=GOOGLE_API_KEY,
        )
    )

    # Instantiate metrics (required by newer RAGAS versions)
    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
    ]

    dataset = build_dataset(test_cases)

    print(f"\n📊 Running RAGAS on {len(dataset)} samples ...")
    print("   (this can take a few minutes)\n")

    with tqdm(total=2, desc="RAGAS metrics", unit="metric") as metric_bar:
        metric_bar.set_postfix_str("answer_relevancy, faithfulness")
        results = evaluate(dataset=dataset, metrics=metrics)
        metric_bar.update(2)

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("RAGAS EVALUATION RESULTS")
    print("="*60)

    # Handle both old dict-style and new EvaluationResult
    scores = {}
    
    # Try new API first
    try:
        df = results.to_pandas()
        for metric_name in ["answer_relevancy", "faithfulness"]:
            if metric_name in df.columns:
                score = df[metric_name].mean()
                emoji = "✅" if score >= 0.7 else ("⚠️" if score >= 0.5 else "❌")
                print(f"  {emoji}  {metric_name:<30} {score:.4f}")
                scores[metric_name] = round(score, 4)
    except:
        # Fallback to old dict API
        for metric_name, score in results.items():
            emoji = "✅" if score >= 0.7 else ("⚠️" if score >= 0.5 else "❌")
            print(f"  {emoji}  {metric_name:<30} {score:.4f}")
            scores[metric_name] = round(score, 4)
        df = results.to_pandas()

    overall = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"\n  {'Overall average':<32} {overall:.4f}")
    print("="*60)

    # ── Per-sample breakdown ──────────────────────────────────
    print("\nPer-sample breakdown:")
    display_cols = [col for col in ["question", "answer_relevancy", "faithfulness"] if col in df.columns]
    if display_cols:
        print(df[display_cols].to_string(index=False))

    # ── Save to JSON ──────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"ragas_results_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": ts,
            "scores": scores,
            "overall_average": round(overall, 4),
            "per_sample": df.to_dict(orient="records"),
        }, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Results saved to {out_path}")

    return scores


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_evaluation()