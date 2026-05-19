# %%
# !pip install neo4j langchain langchain-community langchain-google-genai langchain-neo4j langgraph python-dotenv pandas

# %%
from typing_extensions import Annotated, TypedDict, List
from typing import Literal
from operator import add

from langchain_community.graphs import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import os

# %% [markdown]
# ## Initialize LLM and Neo4j Connection

# %%
# Initialize LLM
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def initialize_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=5,
        google_api_key=GOOGLE_API_KEY
    )

llm = initialize_llm()

# Neo4j connection
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "12345678"

graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    enhanced_schema=True
)

os.environ["NEO4J_URI"] = NEO4J_URI
os.environ["NEO4J_USERNAME"] = NEO4J_USERNAME
os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD

print("✓ LLM and Neo4j connected")
print(f"✓ Graph schema loaded with {len(graph.structured_schema['node_props'])} node types")

# %% [markdown]
# ## State Definitions

# %%
class InputState(TypedDict):
    question: str

class OverallState(TypedDict):
    question: str
    cypher_statement: str
    database_records: List[dict]
    steps: Annotated[List[str], add]

class OutputState(TypedDict):
    cypher_statement: str
    database_records: List[dict]

# %% [markdown]
# ## Text-to-Cypher Chain

# %%
text2cypher_system = """
You are a Neo4j Cypher expert specialized in educational curriculum databases.

TASK: Generate a Cypher statement to query the graph database based on the user's question.

GRAPH SCHEMA:
{schema}

RULES:
1. Use ONLY node labels, relationships, and properties from the schema
2. Return human-readable property names (e.g., ten_hoc_phan AS CourseName)
3. Use CONTAINS for partial text matching (e.g., WHERE f.name CONTAINS "Cơ khí")
4. For Vietnamese text, use exact matches or CONTAINS (case-sensitive)
5. Always specify relationship direction clearly
6. Use OPTIONAL MATCH for prerequisites that may not exist
7. Return ALL relevant fields the user asks for

Generate ONLY the Cypher query, no explanation.
"""

text2cypher_human = """
USER QUESTION:
{question}
"""

text2cypher_prompt = ChatPromptTemplate.from_messages([
    ("system", text2cypher_system),
    ("human", text2cypher_human),
])

text2cypher_chain = text2cypher_prompt | llm | StrOutputParser()

print("✓ Text-to-Cypher chain ready")

# %% [markdown]
# ## Node Functions

# %%
def generate_cypher(state: OverallState) -> OverallState:
    """Generate Cypher query directly from question"""
    generated_cypher = text2cypher_chain.invoke({
        "question": state.get("question"),
        "schema": graph.schema,
    })
    return {
        "cypher_statement": generated_cypher,
        "steps": ["generate_cypher"],
    }

def execute_cypher(state: OverallState) -> OverallState:
    """Execute the Cypher query"""
    try:
        records = graph.query(state.get("cypher_statement"))
    except Exception as e:
        print(f"⚠ Cypher execution error: {e}")
        records = []
    return {
        "database_records": records if records else [],
        "steps": ["execute_cypher"],
    }

print("✓ Node functions defined")

# %% [markdown]
# ## Build LangGraph

# %%
from langgraph.graph import END, START, StateGraph

langgraph = StateGraph(OverallState, input=InputState, output=OutputState)

langgraph.add_node(generate_cypher)
langgraph.add_node(execute_cypher)

langgraph.add_edge(START, "generate_cypher")
langgraph.add_edge("generate_cypher", "execute_cypher")
langgraph.add_edge("execute_cypher", END)

langgraph = langgraph.compile()

print("✓ LangGraph compiled successfully!")

# %% [markdown]
# ## Evaluate

# %%
import json
import re
from pathlib import Path
from collections import defaultdict
import pandas as pd
from IPython.display import display

DATASET_PATH = r"C:\Users\Admin\Desktop\School_Projects\git repositories\SPARK-v2\curriculum_agent\kg_builder\qa_dataset.json"

with open(DATASET_PATH, encoding="utf-8") as f:
    dataset = json.load(f)

print(f"Total questions loaded: {len(dataset)}")

# All known patterns — both RS-F1 and VMA are computed for every pattern.
# RS-F1 measures set completeness (did the agent find all the right records?).
# VMA measures value presence (do the agent's records contain the GT values?).
# The two metrics are complementary, not mutually exclusive.
ALL_PATTERNS = {"traversal_1hop", "traversal_multihop", "path_finding",
                "cross_program", "simple_lookup", "aggregation", "corequisite_with"}

# %%
from collections import Counter

elem_counts    = Counter(r["graph_element"] for r in dataset)
pattern_counts = Counter(r["query_pattern"]  for r in dataset)

df_elem = pd.DataFrame([
    {"Graph Element": k, "# Questions": v}
    for k, v in sorted(elem_counts.items())
])
df_elem.loc[len(df_elem)] = ["Total", sum(elem_counts.values())]

df_pattern = pd.DataFrame([
    {"Query Pattern": k, "# Questions": v}
    for k, v in sorted(pattern_counts.items())
])
df_pattern.loc[len(df_pattern)] = ["Total", sum(pattern_counts.values())]

print("=== Table: Dataset distribution by Graph Element ===")
display(df_elem)
print("\n=== Table: Dataset distribution by Query Pattern ===")
display(df_pattern)

# %%
CHECKPOINT_PATH = "eval_checkpoint.json"

def save_checkpoint(results):
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def load_checkpoint():
    path = Path(CHECKPOINT_PATH)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        print(f"Checkpoint found — {len(data)} questions already done.")
        return data
    print("No checkpoint found — starting fresh.")
    return []

# %%
# Load any existing progress
results = load_checkpoint()
done_questions = {r["question"] for r in results}

remaining = [r for r in dataset if r["question"] not in done_questions]
print(f"Remaining: {len(remaining)} / {len(dataset)} questions")
print(f"Daily budget tip: {len(remaining)} questions × ~2 LLM calls = ~{len(remaining)*2} requests\n")

for i, record in enumerate(remaining):
    question = record["question"]
    global_i = len(results) + 1
    print(f"[{global_i:>3}/{len(dataset)}] {question[:70]}", end=" ... ", flush=True)

    try:
        agent_output  = langgraph.invoke({"question": question})
        agent_records = agent_output.get("database_records", [])
        agent_cypher  = agent_output.get("cypher_statement", "")
        agent_errors  = []

        if isinstance(agent_records, str):
            agent_records = []

        print("OK")
    except Exception as e:
        agent_records = []
        agent_cypher  = ""
        agent_errors  = [str(e)]
        print(f"ERROR: {e}")

    results.append({
        "question":      question,
        "graph_element": record["graph_element"],
        "query_pattern": record["query_pattern"],
        "gt_answer":     record["answer"],
        "gt_cypher":     record["cypher"],
        "agent_records": agent_records,
        "agent_cypher":  agent_cypher,
        "agent_errors":  agent_errors,
    })

    # Save after every question
    save_checkpoint(results)

print(f"\nDone — {len(results)} questions evaluated.")
print(f"Checkpoint saved → {CHECKPOINT_PATH}")

# %%
import re

# ----------------------------
# Normalization
# ----------------------------
def normalize_text(x):
    if x is None:
        return ""
    return str(x).strip().lower()


def extract_values(obj):
    values = []

    if isinstance(obj, dict):
        for v in obj.values():
            values.extend(extract_values(v))

    elif isinstance(obj, list):
        for item in obj:
            values.extend(extract_values(item))

    else:
        # keep raw but normalized string
        values.append(normalize_text(obj))

    return values


def normalize_record(record):
    return set(extract_values(record))


def result_set_f1(agent_records, gt_records, query_pattern=None, threshold=0.8):
    """
    Compute set-overlap Precision / Recall / F1.

    When query_pattern is provided the per-pattern threshold from
    get_threshold() is used instead of the default 0.8, so that e.g.
    aggregation uses 1.0 and traversal uses 0.7.
    """
    if not isinstance(gt_records, list):
        gt_records = []
    if not isinstance(agent_records, list):
        agent_records = []

    gt_norm   = [normalize_record(r) for r in gt_records   if isinstance(r, dict)]
    pred_norm = [normalize_record(r) for r in agent_records if isinstance(r, dict)]

    if len(gt_norm) == 0 and len(pred_norm) == 0:
        return 1.0, 1.0, 1.0

    if len(gt_norm) == 0 or len(pred_norm) == 0:
        return 0.0, 0.0, 0.0

    thr = get_threshold(query_pattern) if query_pattern else threshold

    def overlap(gt, pred):
        return len(gt & pred) / len(gt) if len(gt) > 0 else 0.0

    matched_gt = 0
    used = set()

    for gt in gt_norm:
        best   = 0.0
        best_i = -1

        for i, pred in enumerate(pred_norm):
            if i in used:
                continue
            score = overlap(gt, pred)
            if score > best:
                best   = score
                best_i = i

        if best >= thr and best_i != -1:
            matched_gt += 1
            used.add(best_i)

    precision = matched_gt / len(pred_norm) if pred_norm else 0.0
    recall    = matched_gt / len(gt_norm)   if gt_norm   else 0.0

    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    return round(precision, 4), round(recall, 4), round(f1, 4)

# ----------------------------
# Task-aware threshold
# ----------------------------
def get_threshold(query_pattern):
    """Match threshold for result_set_f1 — stricter for exact/scalar patterns."""
    if query_pattern == "aggregation":
        return 1.0   # count must be exact
    if query_pattern == "path_finding":
        return 0.9   # single record, near-exact
    if query_pattern == "simple_lookup":
        return 0.8   # may return extra fields
    if query_pattern in ("traversal_1hop", "traversal_multihop", "cross_program"):
        return 0.7   # multi-record lists — allow partial key matches
    return 0.8       # default


# ----------------------------
# Execution Accuracy
# ----------------------------
def execution_accuracy(agent_records):
    return int(
        isinstance(agent_records, list)
        and len(agent_records) > 0
        and all(isinstance(r, dict) for r in agent_records)
    )


# ----------------------------
# Scalar Value Match Accuracy
# ----------------------------
def value_match_accuracy(agent_records, gt_records):
    if not isinstance(gt_records, list) or len(gt_records) == 0:
        return 0
    if not isinstance(agent_records, list) or len(agent_records) == 0:
        return 0

    gt_values = normalize_record(gt_records[0])

    for agent in agent_records:
        if not isinstance(agent, dict):
            continue

        agent_values = normalize_record(agent)

        if gt_values.issubset(agent_values):
            return 1

    return 0

# %%
scored = []

for r in results:
    pattern  = r["query_pattern"]
    gt       = r["gt_answer"]
    agent    = r["agent_records"]

    ea  = execution_accuracy(agent)

    # Both metrics apply to every pattern:
    # RS-F1 — how completely did the agent retrieve the right set of records?
    # VMA   — do the agent's records contain the ground-truth values (subset check)?
    prec, rec, f1 = result_set_f1(agent, gt, query_pattern=pattern)
    vma = value_match_accuracy(agent, gt)

    scored.append({
        **r,
        "ea":        ea,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "vma":       vma,
    })

print(f"Scoring complete — {len(scored)} questions scored.")

# %%
def safe_mean(values):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None

ea_all   = safe_mean([s["ea"]  for s in scored])
f1_all   = safe_mean([s["f1"]  for s in scored if s["f1"]  is not None])
vma_all  = safe_mean([s["vma"] for s in scored if s["vma"] is not None])

df_overall = pd.DataFrame([
    {
        "Metric":      "Execution Accuracy (EA)",
        "Description": "Agent returned non-empty result",
        "Score":       f"{ea_all:.2%}",
        "Applied to":  "All questions",
    },
    {
        "Metric":      "Result Set F1 (RS-F1)",
        "Description": "Set overlap F1 with per-pattern match threshold",
        "Score":       f"{f1_all:.2%}" if f1_all is not None else "—",
        "Applied to":  "All questions",
    },
    {
        "Metric":      "Value Match Accuracy (VMA)",
        "Description": "GT values present in agent records (subset check)",
        "Score":       f"{vma_all:.2%}" if vma_all is not None else "—",
        "Applied to":  "All questions",
    },
])

print("=== Table 1: Overall Performance ===")
display(df_overall)

# %%
rows = []
for elem in sorted(set(s["graph_element"] for s in scored)):
    subset = [s for s in scored if s["graph_element"] == elem]

    ea  = safe_mean([s["ea"]  for s in subset])
    f1  = safe_mean([s["f1"]  for s in subset if s["f1"]  is not None])
    vma = safe_mean([s["vma"] for s in subset if s["vma"] is not None])
    n   = len(subset)

    rows.append({
        "Graph Element": elem,
        "N":             n,
        "EA":            f"{ea:.2%}"  if ea  is not None else "—",
        "RS-F1":         f"{f1:.2%}"  if f1  is not None else "—",
        "VMA":           f"{vma:.2%}" if vma is not None else "—",
    })

df_by_elem = pd.DataFrame(rows)
print("=== Table 2: Performance by Graph Element ===")
display(df_by_elem)

# %%
rows = []
for pat in sorted(set(s["query_pattern"] for s in scored)):
    subset = [s for s in scored if s["query_pattern"] == pat]

    ea  = safe_mean([s["ea"]  for s in subset])
    f1  = safe_mean([s["f1"]  for s in subset if s["f1"]  is not None])
    vma = safe_mean([s["vma"] for s in subset if s["vma"] is not None])
    n   = len(subset)

    rows.append({
        "Query Pattern": pat,
        "N":             n,
        "EA":            f"{ea:.2%}"  if ea  is not None else "—",
        "RS-F1":         f"{f1:.2%}"  if f1  is not None else "—",
        "VMA":           f"{vma:.2%}" if vma is not None else "—",
    })

df_by_pattern = pd.DataFrame(rows)
print("=== Table 3: Performance by Query Pattern ===")
display(df_by_pattern)

# %%
rows = []
for s in scored:
    rows.append({
        "question":      s["question"],
        "graph_element": s["graph_element"],
        "query_pattern": s["query_pattern"],
        "EA":            s["ea"],
        "RS-F1":         s["f1"]  if s["f1"]  is not None else "—",
        "VMA":           s["vma"] if s["vma"] is not None else "—",
        "gt_answer":     str(s["gt_answer"])[:80],
        "agent_records": str(s["agent_records"])[:80],
        "agent_cypher":  s["agent_cypher"],
        "agent_errors":  s["agent_errors"],
    })

df_detail = pd.DataFrame(rows)

# Show only failed questions for error analysis
failed = df_detail[
    (df_detail["EA"] == 0) |
    (df_detail["RS-F1"].apply(lambda x: float(x) < 1.0 if x not in ("—", None) else False)) |
    (df_detail["VMA"].apply(lambda x: x == 0 if x not in ("—", None) else False))
]

print(f"=== Per-question detail: {len(failed)} failed / {len(df_detail)} total ===")
display(df_detail)

# Save full detail to CSV for manual error analysis
df_detail.to_csv("evaluation_detail.csv", index=False, encoding="utf-8-sig")
print("Saved → evaluation_detail.csv")

# %%
df_overall.to_csv("eval_overall.csv",     index=False, encoding="utf-8-sig")
df_by_elem.to_csv("eval_by_element.csv",  index=False, encoding="utf-8-sig")
df_by_pattern.to_csv("eval_by_pattern.csv", index=False, encoding="utf-8-sig")

print("Saved:")
print("  eval_overall.csv")
print("  eval_by_element.csv")
print("  eval_by_pattern.csv")
print("  evaluation_detail.csv")