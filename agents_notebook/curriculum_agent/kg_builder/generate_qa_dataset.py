"""
DUT Curriculum — Text2Cypher QA Dataset Generator

Generates a structured QA dataset for evaluating a Text2Cypher agent.
Each record contains: context, question, cypher, answer, query_pattern, graph_element

Usage:
    python generate_qa_dataset.py
    python generate_qa_dataset.py --json dut_curriculum_ALL.json --output qa_dataset.json
"""

import json
import random
import re
import argparse
import time
from pathlib import Path

from tqdm import tqdm

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.graphs import Neo4jGraph


# ─────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────
NEO4J_URI      = "neo4j://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "12345678"

GOOGLE_API_KEY = "AIzaSyDVOtnnk-RDilFd49XnXad9V-UVh-swzVs"

QUESTIONS_PER_CATEGORY = 15   # target per graph_element category
RANDOM_SEED            = 42
DELAY_BETWEEN_CALLS    = 1.5  # seconds, to avoid rate limiting


# ─────────────────────────────────────────────────────────
# Initialise LLM + Graph
# ─────────────────────────────────────────────────────────
def initialize_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=5,
        google_api_key=GOOGLE_API_KEY,
    )


def initialize_graph():
    return Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        enhanced_schema=False,
    )


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────
def clean(text):
    return (text or "").strip()


def extract_names_from_rel_field(text):
    """
    Parse prerequisite / corequisite / hoc_truoc fields like:
      "3190111- Giải tích 1, 3050011- Vật lý 1"
    Returns list of subject names: ["Giải tích 1", "Vật lý 1"]
    """
    if not text:
        return []
    # Split on comma, then remove the leading code + dash
    parts = [p.strip() for p in text.split(",")]
    names = []
    for part in parts:
        # Remove leading 7-digit code and dash/space
        cleaned = re.sub(r'^\d{7}[-\s]*', '', part).strip()
        if cleaned:
            names.append(cleaned)
    return names


def run_cypher(graph, cypher):
    """Execute Cypher and return results. Returns error string on failure."""
    try:
        result = graph.query(cypher)
        return result
    except Exception as e:
        return f"ERROR: {e}"


def call_llm(llm, prompt):
    """Call LLM and return text response. Handles both string and list content."""
    try:
        response = llm.invoke(prompt)
        content = response.content
        # Gemini sometimes returns a list of content blocks
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return content.strip()
    except Exception as e:
        return f"ERROR: {e}"


def parse_llm_json(raw):
    """
    Extract JSON from LLM response.
    Handles markdown code fences and bare JSON.
    """
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        # Try to find the first {...} block
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return None


# ─────────────────────────────────────────────────────────
# Samplers — one per graph_element category
# ─────────────────────────────────────────────────────────
def sample_faculty_packets(data, n):
    """Sample (faculty_name, [program_names]) packets."""
    faculty_map = {}
    for prog in data:
        fac = clean(prog.get("khoa_quan_ly", ""))
        name = clean(prog.get("ten_chuong_trinh", ""))
        if fac and name:
            faculty_map.setdefault(fac, []).append(name)

    faculties = list(faculty_map.items())
    random.shuffle(faculties)
    packets = []
    for fac, programs in faculties[:n]:
        packets.append({
            "faculty_name": fac,
            "programs": programs[:5],   # show up to 5 program names as context
        })
    return packets


def sample_program_packets(data, n):
    """Sample individual program info packets."""
    pool = [p for p in data if clean(p.get("ten_chuong_trinh", ""))]
    random.shuffle(pool)
    packets = []
    for prog in pool[:n]:
        packets.append({
            "program_name":  clean(prog.get("ten_chuong_trinh", "")),
            "faculty":       clean(prog.get("khoa_quan_ly", "")),
            "major":         clean(prog.get("ten_nganh", "")),
            "total_credits": clean(prog.get("so_tin_chi", "")),
            "num_semesters": clean(prog.get("so_ky", "")),
            "language":      clean(prog.get("ngon_ngu", "")),
        })
    return packets


def sample_semester_packets(data, n):
    """Sample (program, semester_number, [subject_names]) packets."""
    packets = []
    pool = []
    for prog in data:
        prog_name = clean(prog.get("ten_chuong_trinh", ""))
        if not prog_name:
            continue
        semesters = {}
        for subj in prog.get("subjects", []):
            hk = clean(subj.get("hoc_ky", ""))
            name = clean(subj.get("ten_hoc_phan", ""))
            if hk and hk.isdigit() and name:
                semesters.setdefault(hk, []).append(name)
        for sem_num, subjects in semesters.items():
            pool.append({
                "program_name":   prog_name,
                "semester_number": int(sem_num),
                "subjects":       subjects,
            })
    random.shuffle(pool)
    return pool[:n]


def sample_subject_packets(data, n):
    """Sample individual subject info packets (with program context)."""
    pool = []
    for prog in data:
        prog_name = clean(prog.get("ten_chuong_trinh", ""))
        if not prog_name:
            continue
        for subj in prog.get("subjects", []):
            name = clean(subj.get("ten_hoc_phan", ""))
            hk   = clean(subj.get("hoc_ky", ""))
            if name and hk:
                pool.append({
                    "program_name":   prog_name,
                    "subject_name":   name,
                    "credits":        clean(subj.get("so_tin_chi", "")),
                    "elective":       clean(subj.get("tu_chon", "")),
                    "semester":       hk,
                })
    random.shuffle(pool)
    # Deduplicate by subject_name to avoid repetition
    seen = set()
    packets = []
    for p in pool:
        if p["subject_name"] not in seen:
            seen.add(p["subject_name"])
            packets.append(p)
        if len(packets) >= n:
            break
    return packets


def sample_has_subject_packets(data, n):
    """
    Sample (subject, semester, program) triples where the
    HAS_SUBJECT relationship is the focus.
    """
    return sample_subject_packets(data, n)   # same data, different prompt


def sample_prerequisite_packets(data, n):
    """Sample subjects that have non-empty tien_quyet."""
    pool = []
    for prog in data:
        prog_name = clean(prog.get("ten_chuong_trinh", ""))
        if not prog_name:
            continue
        for subj in prog.get("subjects", []):
            tq = clean(subj.get("tien_quyet", ""))
            name = clean(subj.get("ten_hoc_phan", ""))
            if tq and name:
                prereq_names = extract_names_from_rel_field(tq)
                if prereq_names:
                    pool.append({
                        "program_name":      prog_name,
                        "subject_name":      name,
                        "prerequisite_names": prereq_names,
                    })
    random.shuffle(pool)
    # Deduplicate by subject_name
    seen = set()
    packets = []
    for p in pool:
        if p["subject_name"] not in seen:
            seen.add(p["subject_name"])
            packets.append(p)
        if len(packets) >= n:
            break
    return packets


def sample_corequisite_packets(data, n):
    """Sample subjects that have non-empty song_hanh."""
    pool = []
    for prog in data:
        prog_name = clean(prog.get("ten_chuong_trinh", ""))
        if not prog_name:
            continue
        for subj in prog.get("subjects", []):
            sh = clean(subj.get("song_hanh", ""))
            name = clean(subj.get("ten_hoc_phan", ""))
            if sh and name:
                coreq_names = extract_names_from_rel_field(sh)
                if coreq_names:
                    pool.append({
                        "program_name":     prog_name,
                        "subject_name":     name,
                        "corequisite_names": coreq_names,
                    })
    random.shuffle(pool)
    seen = set()
    packets = []
    for p in pool:
        if p["subject_name"] not in seen:
            seen.add(p["subject_name"])
            packets.append(p)
        if len(packets) >= n:
            break
    return packets


# ─────────────────────────────────────────────────────────
# Prompt templates per category
# ─────────────────────────────────────────────────────────
# Question style pools — sampled randomly to force variety
FACULTY_STYLES = [
    "Ask what programs belong to this faculty.",
    "Ask how many programs this faculty manages.",
    "Ask which faculty a specific program belongs to (reverse lookup).",
    "Ask whether a specific program is under this faculty.",
    "Ask to list only the program names under this faculty.",
]

PROGRAM_STYLES = [
    "Ask about the total number of credits required.",
    "Ask how many semesters the program spans.",
    "Ask what language the program is taught in.",
    "Ask which faculty manages this program.",
    "Ask for the major/specialisation name of this program.",
    "Ask whether this program is full-time or part-time based on its properties.",
]

SEMESTER_STYLES = [
    "Ask what subjects are studied in a specific semester.",
    "Ask how many subjects are in a specific semester.",
    "Ask to list only the subject names for a semester.",
    "Ask which semester has the most subjects.",
    "Ask whether a specific subject appears in a given semester.",
]

SUBJECT_STYLES = [
    "Ask how many credits this subject carries.",
    "Ask whether this subject is elective or compulsory.",
    "Ask which semester this subject is taught in.",
    "Ask for all details about this subject.",
    "Ask whether this subject exists in the program.",
]

HAS_SUBJECT_STYLES = [
    "Ask which semester contains this subject in the program.",
    "Ask whether the program includes this subject at all.",
    "Ask how many semesters in the program contain any subject.",
    "Ask to find which semester a student would first encounter this subject.",
    "Ask which subjects are linked to a particular semester by the HAS_SUBJECT relationship.",
]

PREREQUISITE_STYLES = [
    "Ask what subjects must be completed before taking this subject.",
    "Ask what this subject unlocks (i.e. what subjects have it as a prerequisite).",
    "Ask whether subject A is a prerequisite for subject B.",
    "Ask how many prerequisites this subject has.",
    "Ask for the full prerequisite chain leading to this subject.",
]

COREQUISITE_STYLES = [
    "Ask what subjects must be taken at the same time as this subject.",
    "Ask whether two specific subjects are corequisites of each other.",
    "Ask how many corequisites this subject has.",
    "Ask which subjects in the program have corequisites.",
    "Ask to list all corequisite pairs in the program.",
]

STYLE_POOLS = {
    "Faculty":          FACULTY_STYLES,
    "Program":          PROGRAM_STYLES,
    "Semester":         SEMESTER_STYLES,
    "Subject":          SUBJECT_STYLES,
    "HAS_SUBJECT":      HAS_SUBJECT_STYLES,
    "PREREQUISITE_OF":  PREREQUISITE_STYLES,
    "COREQUISITE_WITH": COREQUISITE_STYLES,
}

SYSTEM_PROMPT = """You are an assistant that generates evaluation data for a Text2Cypher agent.
The knowledge graph has this structure:
  (Faculty)-[:HAS_PROGRAM]->(Program)-[:HAS_SEMESTER]->(Semester)-[:HAS_SUBJECT]->(Subject)
  (Subject)-[:PREREQUISITE_OF]->(Subject)
  (Subject)-[:COREQUISITE_WITH]->(Subject)
  (Subject)-[:RECOMMENDED_BEFORE]->(Subject)

Node properties:
  Faculty:  name
  Program:  ten_chuong_trinh, ma_nganh, so_tin_chi, ngon_ngu, so_ky
            (NOTE: faculty name is NOT stored on Program — to get faculty, traverse HAS_PROGRAM)
  Semester: number, program_code
  Subject:  ten_hoc_phan, so_tin_chi, tu_chon, ma_hp

To find which faculty manages a program:
  MATCH (f:Faculty)-[:HAS_PROGRAM]->(p:Program {ten_chuong_trinh: "..."}) RETURN f.name

To find programs under a faculty:
  MATCH (f:Faculty {name: "..."})-[:HAS_PROGRAM]->(p:Program) RETURN p.ten_chuong_trinh

Important rules:
- The user always provides their program name in the conversation context.
- Always filter by program using: MATCH (p:Program {ten_chuong_trinh: "<program_name>"})
- Use EXACT string matches only (=), never CONTAINS or fuzzy matching.
- Use subject name for matching: MATCH (s:Subject {ten_hoc_phan: "<name>"})
- Return only a valid JSON object with keys: question, cypher, query_pattern
- query_pattern must be one of: simple_lookup, traversal_1hop, traversal_multihop, aggregation, path_finding, cross_program
- Do NOT include any explanation or markdown outside the JSON.

Variety rules — you MUST follow these:
- Write the question as a natural, conversational sentence a real student would ask.
- NEVER start two questions with the same opening word or phrase.
- Vary sentence structure: use statements-as-questions, indirect questions, yes/no questions,
  and wh-questions (what/which/how many/does/is/can/could).
- Rephrase entity names naturally — do NOT wrap them in quotes inside the question.
- The question must sound like it came from a different person each time.
"""


def make_faculty_prompt(packet):
    style = random.choice(STYLE_POOLS["Faculty"])
    return f"""{SYSTEM_PROMPT}

Context:
  Faculty: "{packet['faculty_name']}"
  Programs under this faculty: {packet['programs']}

Question style to use: {style}

Generate one natural English question following that style, about this faculty and its programs.
The question should focus on the Faculty node or the HAS_PROGRAM relationship.
Return JSON: {{"question": "...", "cypher": "...", "query_pattern": "..."}}
"""


def make_program_prompt(packet):
    style = random.choice(STYLE_POOLS["Program"])
    return f"""{SYSTEM_PROMPT}

Context:
  Program name: "{packet['program_name']}"
  Faculty: "{packet['faculty']}"
  Major: "{packet['major']}"
  Total credits: {packet['total_credits']}
  Number of semesters: {packet['num_semesters']}
  Language: "{packet['language']}"

Question style to use: {style}

The user is enrolled in this program. Generate one natural English question following
that style, about this program's general information (credits, semesters, language, faculty, etc.).
Return JSON: {{"question": "...", "cypher": "...", "query_pattern": "..."}}
"""


def make_semester_prompt(packet):
    style = random.choice(STYLE_POOLS["Semester"])
    return f"""{SYSTEM_PROMPT}

Context:
  Program name: "{packet['program_name']}"
  Semester number: {packet['semester_number']}
  Subjects in this semester: {packet['subjects']}

Question style to use: {style}

The user is enrolled in "{packet['program_name']}".
Generate one natural English question following that style, about semester {packet['semester_number']} of this program.
Return JSON: {{"question": "...", "cypher": "...", "query_pattern": "..."}}
"""


def make_subject_prompt(packet):
    style = random.choice(STYLE_POOLS["Subject"])
    return f"""{SYSTEM_PROMPT}

Context:
  Program name: "{packet['program_name']}"
  Subject name: "{packet['subject_name']}"
  Credits: {packet['credits']}
  Elective: "{packet['elective']}"
  Semester: {packet['semester']}

Question style to use: {style}

The user is enrolled in "{packet['program_name']}".
Generate one natural English question following that style, about this specific subject.
Return JSON: {{"question": "...", "cypher": "...", "query_pattern": "..."}}
"""


def make_has_subject_prompt(packet):
    style = random.choice(STYLE_POOLS["HAS_SUBJECT"])
    return f"""{SYSTEM_PROMPT}

Context:
  Program name: "{packet['program_name']}"
  Subject name: "{packet['subject_name']}"
  Semester: {packet['semester']}

Question style to use: {style}

The user is enrolled in "{packet['program_name']}".
Generate one natural English question following that style. The focus is the HAS_SUBJECT relationship.
Return JSON: {{"question": "...", "cypher": "...", "query_pattern": "..."}}
"""


def make_prerequisite_prompt(packet):
    style = random.choice(STYLE_POOLS["PREREQUISITE_OF"])
    return f"""{SYSTEM_PROMPT}

Context:
  Program name: "{packet['program_name']}"
  Subject name: "{packet['subject_name']}"
  Prerequisites (must complete before): {packet['prerequisite_names']}

Question style to use: {style}

The user is enrolled in "{packet['program_name']}".
Generate one natural English question following that style, about prerequisites related to this subject.
Return JSON: {{"question": "...", "cypher": "...", "query_pattern": "..."}}
"""


def make_corequisite_prompt(packet):
    style = random.choice(STYLE_POOLS["COREQUISITE_WITH"])
    return f"""{SYSTEM_PROMPT}

Context:
  Program name: "{packet['program_name']}"
  Subject name: "{packet['subject_name']}"
  Corequisites (must take at same time): {packet['corequisite_names']}

Question style to use: {style}

The user is enrolled in "{packet['program_name']}".
Generate one natural English question following that style, about corequisites related to this subject.
Return JSON: {{"question": "...", "cypher": "...", "query_pattern": "..."}}
"""


PROMPT_BUILDERS = {
    "Faculty":          (sample_faculty_packets,      make_faculty_prompt),
    "Program":          (sample_program_packets,      make_program_prompt),
    "Semester":         (sample_semester_packets,     make_semester_prompt),
    "Subject":          (sample_subject_packets,      make_subject_prompt),
    "HAS_SUBJECT":      (sample_has_subject_packets,  make_has_subject_prompt),
    "PREREQUISITE_OF":  (sample_prerequisite_packets, make_prerequisite_prompt),
    "COREQUISITE_WITH": (sample_corequisite_packets,  make_corequisite_prompt),
}


# ─────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────
def generate_dataset(data, llm, graph, questions_per_category=QUESTIONS_PER_CATEGORY):
    random.seed(RANDOM_SEED)
    dataset = []

    categories = list(PROMPT_BUILDERS.items())
    total_categories = len(categories)

    # Outer bar — tracks category-level progress
    outer_bar = tqdm(
        categories,
        desc="Overall",
        unit="category",
        position=0,
        colour="green",
    )

    for graph_element, (sampler_fn, prompt_fn) in outer_bar:
        outer_bar.set_description(f"Overall  [{graph_element:<20}]")

        packets = sampler_fn(data, questions_per_category)
        if not packets:
            tqdm.write(f"  [!] No packets found for {graph_element}, skipping.")
            continue

        success = 0
        errors  = 0

        # Inner bar — tracks question-level progress within each category
        inner_bar = tqdm(
            packets,
            desc=f"  {graph_element:<20}",
            unit="q",
            position=1,
            leave=False,
            colour="cyan",
        )

        for packet in inner_bar:
            inner_bar.set_postfix(ok=success, err=errors)

            prompt = prompt_fn(packet)
            raw    = call_llm(llm, prompt)

            if raw.startswith("ERROR"):
                errors += 1
                inner_bar.set_postfix(ok=success, err=errors)
                tqdm.write(f"  [LLM error]   {graph_element}: {raw[:80]}")
                continue

            parsed = parse_llm_json(raw)
            if not parsed:
                errors += 1
                inner_bar.set_postfix(ok=success, err=errors)
                tqdm.write(f"  [Parse fail]  {graph_element}: {raw[:80]}")
                continue

            question      = parsed.get("question", "").strip()
            cypher        = parsed.get("cypher", "").strip()
            query_pattern = parsed.get("query_pattern", "unknown").strip()

            if not question or not cypher:
                errors += 1
                inner_bar.set_postfix(ok=success, err=errors)
                tqdm.write(f"  [Empty field] {graph_element}: question or cypher missing")
                continue

            # Run Cypher against Neo4j to get ground truth answer
            answer = run_cypher(graph, cypher)

            record = {
                "graph_element": graph_element,
                "query_pattern": query_pattern,
                "context":       packet,
                "question":      question,
                "cypher":        cypher,
                "answer":        answer,
            }
            dataset.append(record)
            success += 1
            inner_bar.set_postfix(ok=success, err=errors)

            # Print each successful question above the bars
            tqdm.write(f"  ✓ [{graph_element:<20}] [{query_pattern:<22}] {question[:65]}")

            time.sleep(DELAY_BETWEEN_CALLS)

        inner_bar.close()
        tqdm.write(f"\n  ── {graph_element}: {success} ok, {errors} failed ──\n")

    outer_bar.close()
    return dataset


# ─────────────────────────────────────────────────────────
# Summary report
# ─────────────────────────────────────────────────────────
def print_summary(dataset):
    from collections import Counter
    print("\n" + "="*60)
    print("  DATASET SUMMARY")
    print("="*60)
    print(f"  Total questions: {len(dataset)}")

    print("\n  By graph_element:")
    for elem, count in Counter(r["graph_element"] for r in dataset).most_common():
        print(f"    {elem:<20} {count}")

    print("\n  By query_pattern:")
    for pat, count in Counter(r["query_pattern"] for r in dataset).most_common():
        print(f"    {pat:<25} {count}")

    errors = sum(1 for r in dataset if isinstance(r["answer"], str) and r["answer"].startswith("ERROR"))
    print(f"\n  Cypher execution errors: {errors}/{len(dataset)}")
    print("="*60)


# ─────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Generate Text2Cypher QA dataset for DUT curriculum")
    p.add_argument("--json",   default="dut_curriculum_ALL.json", help="Input JSON file")
    p.add_argument("--output", default="qa_dataset.json",         help="Output JSON file")
    p.add_argument("--n",      type=int, default=QUESTIONS_PER_CATEGORY,
                   help="Questions per graph element category")
    return p.parse_args()


def main():
    args = parse_args()

    # Load JSON
    json_path = Path(args.json)
    if not json_path.exists():
        print(f"✗ File not found: {json_path}")
        return

    print(f"[1] Loading {json_path} …")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"    ✓ {len(data)} programs loaded")

    # Init LLM + Graph
    print("[2] Initialising LLM and Neo4j …")
    llm   = initialize_llm()
    graph = initialize_graph()
    print("    ✓ Ready")

    # Generate
    print(f"[3] Generating dataset ({args.n} questions per category × 7 categories) …")
    dataset = generate_dataset(data, llm, graph, questions_per_category=args.n)

    # Save
    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"\n[✓] Dataset saved → {out_path}")

    print_summary(dataset)


if __name__ == "__main__":
    main()