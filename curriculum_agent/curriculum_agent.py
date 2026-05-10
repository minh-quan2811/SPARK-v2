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
def initialize_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=5,
        google_api_key="AIzaSyB5H5BRqzVlDG45yMBXS1ZzVMe1otV6o7U"
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

# Query patterns that return lists vs single values
LIST_PATTERNS   = {"traversal_1hop", "traversal_multihop", "path_finding", "cross_program"}
SCALAR_PATTERNS = {"simple_lookup", "aggregation"}

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
def normalise_record(record):
    """
    Convert a dict record to a frozenset of its values,
    ignoring column alias names entirely.
    Handles nested lists by flattening them to strings.
    """
    values = []
    for v in record.values():
        if isinstance(v, list):
            values.append(tuple(sorted(str(x) for x in v)))
        else:
            values.append(str(v).strip().lower())
    return frozenset(values)


def execution_accuracy(agent_records):
    """
    EA: 1 if agent returned at least one record, 0 otherwise.
    """
    return 1 if (isinstance(agent_records, list) and len(agent_records) > 0) else 0


def result_set_f1(agent_records, gt_records):
    """
    RS-F1: compare agent vs ground truth as sets of normalised records.
    Returns (precision, recall, f1).
    If both are empty → perfect match (1,1,1).
    If one is empty and the other is not → (0,0,0).
    """
    if not isinstance(gt_records, list):
        gt_records = []
    if not isinstance(agent_records, list):
        agent_records = []

    agent_set = set(normalise_record(r) for r in agent_records if isinstance(r, dict))
    gt_set    = set(normalise_record(r) for r in gt_records    if isinstance(r, dict))

    if len(agent_set) == 0 and len(gt_set) == 0:
        return 1.0, 1.0, 1.0
    if len(agent_set) == 0 or len(gt_set) == 0:
        return 0.0, 0.0, 0.0

    intersection = agent_set & gt_set
    precision = len(intersection) / len(agent_set)
    recall    = len(intersection) / len(gt_set)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


def value_match_accuracy(agent_records, gt_records):
    """
    VMA: for single-scalar answers, compare the one returned value.
    Returns 1 if match, 0 otherwise.
    """
    if not isinstance(gt_records, list) or len(gt_records) == 0:
        return 0
    if not isinstance(agent_records, list) or len(agent_records) == 0:
        return 0

    # Extract single value from first record
    def extract_scalar(records):
        first = records[0]
        if isinstance(first, dict) and len(first) == 1:
            return str(list(first.values())[0]).strip().lower()
        # Multiple keys — join all values as a string for comparison
        return str(sorted(str(v) for v in first.values())).lower()

    return 1 if extract_scalar(agent_records) == extract_scalar(gt_records) else 0

# %%
scored = []

for r in results:
    pattern  = r["query_pattern"]
    gt       = r["gt_answer"]
    agent    = r["agent_records"]

    ea  = execution_accuracy(agent)

    if pattern in LIST_PATTERNS:
        prec, rec, f1 = result_set_f1(agent, gt)
        vma = None
    elif pattern in SCALAR_PATTERNS:
        prec, rec, f1 = None, None, None
        vma = value_match_accuracy(agent, gt)
    else:
        # Unknown pattern — apply both conservatively
        prec, rec, f1 = result_set_f1(agent, gt)
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
        "Description": "Set overlap between agent and ground truth records",
        "Score":       f"{f1_all:.2%}",
        "Applied to":  "traversal_1hop, traversal_multihop, path_finding, cross_program",
    },
    {
        "Metric":      "Value Match Accuracy (VMA)",
        "Description": "Exact scalar value match",
        "Score":       f"{vma_all:.2%}",
        "Applied to":  "simple_lookup, aggregation",
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
        "RS-F1":         f"{f1:.2%}"  if f1  is not None else "N/A",
        "VMA":           f"{vma:.2%}" if vma is not None else "N/A",
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