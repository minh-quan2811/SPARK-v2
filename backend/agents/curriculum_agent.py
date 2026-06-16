import os
import re
import asyncio
from typing import Annotated, Literal, List, Optional
from typing_extensions import TypedDict
from operator import add
from pydantic import BaseModel, Field

from neo4j.exceptions import CypherSyntaxError
from langchain_community.graphs import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_neo4j import Neo4jVector
from langchain_neo4j.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema
from langgraph.graph import END, START, StateGraph
from dotenv import load_dotenv

load_dotenv()


# Singletons

_llm: Optional[ChatGoogleGenerativeAI] = None
_graph: Optional[Neo4jGraph] = None
_example_selector = None
_cypher_corrector = None


def get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview",
            temperature=0,
            max_retries=5,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    return _llm


def get_graph() -> Neo4jGraph:
    global _graph
    if _graph is None:
        _graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            enhanced_schema=True,
        )
    return _graph


import difflib
import unicodedata

def resolve_program_name(program: str) -> str:
    """
    Find the exact program name stored in Neo4j closest to the input.
    Handles ALL Unicode lookalike issues without any manual character map.
    Works by comparing against what is actually in the database.
    """
    # NFC first — merges decomposed diacritics (ă stored as a + combining breve, etc.)
    program = unicodedata.normalize('NFC', program)

    rows = get_graph().query(
        "MATCH (p:Program) RETURN p.ten_chuong_trinh AS name"
    )
    db_names = [r["name"] for r in rows if r.get("name")]

    if not db_names:
        return program

    # Exact match — fastest path, no fuzzy needed
    if program in db_names:
        return program

    # Fuzzy match — finds the closest real name in the DB
    matches = difflib.get_close_matches(program, db_names, n=1, cutoff=0.8)
    if matches:
        print(f"Resolved: '{program}' → '{matches[0]}'")
        return matches[0]

    # No close match found — return as-is, let validation handle it
    return program

def get_example_selector():
    global _example_selector
    if _example_selector is None:
        examples = [
            {
                "question": "Retrieve all subjects in semester 4 only for program Cơ khí hàng không K2020_Kỹ sư",
                "query": (
                    "MATCH (p:Program)-[:HAS_SEMESTER]->(s:Semester)-[:HAS_SUBJECT]->(sub:Subject) "
                    "WHERE p.ten_chuong_trinh = 'Cơ khí hàng không K2020_Kỹ sư' "
                    "AND toInteger(s.number) >= 4 AND toInteger(s.number) <= 4 "
                    "RETURN p.ten_chuong_trinh AS program, toInteger(s.number) AS semester, "
                    "sub.ma_hp AS code, sub.ten_hoc_phan AS name, sub.so_tin_chi AS credits "
                    "ORDER BY toInteger(s.number), sub.ma_hp"
                ),
            },
            {
                "question": "Retrieve all subjects in semesters 3 through 5 for program Công nghệ Thông tin K2021CLC Đặc thù_CNPM",
                "query": (
                    "MATCH (p:Program)-[:HAS_SEMESTER]->(s:Semester)-[:HAS_SUBJECT]->(sub:Subject) "
                    "WHERE p.ten_chuong_trinh = 'Công nghệ Thông tin K2021CLC Đặc thù_CNPM' "
                    "AND toInteger(s.number) >= 3 AND toInteger(s.number) <= 5 "
                    "RETURN p.ten_chuong_trinh AS program, toInteger(s.number) AS semester, "
                    "sub.ma_hp AS code, sub.ten_hoc_phan AS name, sub.so_tin_chi AS credits "
                    "ORDER BY toInteger(s.number), sub.ma_hp"
                ),
            },
            {
                "question": "Retrieve all subjects in semesters 6 and 7 for program Kỹ thuật Điện K2022_kỹ sư",
                "query": (
                    "MATCH (p:Program)-[:HAS_SEMESTER]->(s:Semester)-[:HAS_SUBJECT]->(sub:Subject) "
                    "WHERE p.ten_chuong_trinh = 'Kỹ thuật Điện K2022_kỹ sư' "
                    "AND toInteger(s.number) >= 6 AND toInteger(s.number) <= 7 "
                    "RETURN p.ten_chuong_trinh AS program, toInteger(s.number) AS semester, "
                    "sub.ma_hp AS code, sub.ten_hoc_phan AS name, sub.so_tin_chi AS credits "
                    "ORDER BY toInteger(s.number), sub.ma_hp"
                ),
            },
        ]
        _example_selector = SemanticSimilarityExampleSelector.from_examples(
            examples,
            GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            ),
            Neo4jVector,
            k=2,
            input_keys=["question"],
        )
    return _example_selector


def get_cypher_corrector():
    global _cypher_corrector
    if _cypher_corrector is None:
        corrector_schema = [
            Schema(el["start"], el["type"], el["end"])
            for el in get_graph().structured_schema.get("relationships", [])
        ]
        _cypher_corrector = CypherQueryCorrector(corrector_schema)
    return _cypher_corrector


# State

class InputState(TypedDict):
    program: str
    current_semester: int
    plan_preferences: str

class OverallState(TypedDict):
    program: str
    current_semester: int
    plan_preferences: str
    time_scope: str
    task_target: str
    task_allocation: str
    next_action: str
    cypher_statement: str
    cypher_errors: List[str]
    database_records: List[dict]
    steps: Annotated[List[str], add]
    correction_attempts: int

class OutputState(TypedDict):
    cypher_statement: str
    database_records: List[dict]
    dependencies: dict
    errors: List[str]

class FilterCondition(BaseModel):
    node_label: str      = Field(description="Node label from the Cypher statement")
    property_key: str    = Field(description="Property key being filtered")
    property_value: str  = Field(description="Value used in the filter")

class CypherEvaluation(BaseModel):
    errors:  List[str]             = Field(description="List of errors in the Cypher statement")
    filters: List[FilterCondition] = Field(description="Filters used in the Cypher statement")


# Chains

_extract_time_scope_chain = ChatPromptTemplate.from_messages([
    ("system", """You are a study planning assistant. Read the student's plan preferences and extract the time duration they want to plan for.

Describe the time scope in natural language based on what they say. Examples:
- "I want to plan for next 2 semesters" → "the next 2 semesters starting from semester {current_semester}, covering semesters {current_semester} and {current_semester_plus_1}"
- "I want to plan for this semester only" → "the current semester only (semester {current_semester})"
- "plan for 1 year ahead" → "the next 2 semesters starting from semester {current_semester}"
- No time mentioned → "the current semester only (semester {current_semester})"

Always include the actual semester numbers in your response. Current semester is {current_semester}.
Return only the natural language time scope description, nothing else."""),
    ("human", "Plan preferences: {plan_preferences}"),
]) | get_llm() | StrOutputParser()


_task_target_chain = ChatPromptTemplate.from_messages([
    ("system", """You are a curriculum planning assistant. Based on the program name, current semester, and the time scope the student wants, state clearly and specifically what needs to be retrieved from the curriculum database.

Your output should be a single clear goal statement that includes:
- The exact program name
- The exact semester numbers to retrieve
- What information is needed (subjects, their codes, names, credits)

Be specific with the actual numbers and program name. Do not use placeholders."""),
    ("human", """Program: {program}
Current semester: {current_semester}
Time scope requested: {time_scope}

State the retrieval target."""),
]) | get_llm() | StrOutputParser()


_task_allocation_chain = ChatPromptTemplate.from_messages([
    ("system", """You are an expert in Neo4j graph database queries for educational curriculum systems.

Given a retrieval target, write a step-by-step plan for how to construct the Cypher query.

GRAPH SCHEMA:
{schema}

Key facts:
- Traversal path: (Program)-[:HAS_SEMESTER]->(Semester)-[:HAS_SUBJECT]->(Subject)
- Program is matched by exact equality on p.ten_chuong_trinh
- Semester number is stored as a Neo4j Integer — always use toInteger(s.number) when filtering or returning it
- Subject properties: sub.ma_hp (code), sub.ten_hoc_phan (name), sub.so_tin_chi (credits)
- The program name and semester numbers must be written as literal values in the query, not as parameters

Each step should describe one logical operation. Be specific."""),
    ("human", "Retrieval target: {task_target}"),
]) | get_llm() | StrOutputParser()


_text2cypher_chain = ChatPromptTemplate.from_messages([
    ("system", """You are a Neo4j Cypher expert specialized in educational curriculum databases.

TASK: Generate a Cypher query to retrieve all subjects for the given program and semester range.

GRAPH SCHEMA:
{schema}

FEW-SHOT EXAMPLES:
{fewshot_examples}

CONSTRUCTION PLAN:
{task_allocation}

RULES:
1. Traverse: (Program)-[:HAS_SEMESTER]->(Semester)-[:HAS_SUBJECT]->(Subject)
2. Match program with exact equality: p.ten_chuong_trinh = '<actual program name>'
3. Filter semesters with: toInteger(s.number) >= <start> AND toInteger(s.number) <= <end>
4. Return: p.ten_chuong_trinh AS program, toInteger(s.number) AS semester, sub.ma_hp AS code, sub.ten_hoc_phan AS name, sub.so_tin_chi AS credits
5. Order by: toInteger(s.number), sub.ma_hp
6. Write the actual program name and actual semester numbers as literals — no placeholders, no parameters

Generate ONLY the Cypher query, no explanation, no markdown fences."""),
    ("human", "Retrieval target: {task_target}"),
]) | get_llm() | StrOutputParser()


_validate_cypher_chain = ChatPromptTemplate.from_messages([
    ("system", """You are a Cypher query validator for a curriculum database.

SCHEMA:
{schema}

CYPHER STATEMENT:
{cypher}

RETRIEVAL TARGET:
{task_target}

Check for: incorrect node labels or relationships, misspelled property names, invalid syntax, wrong relationship directions, missing return fields, use of $parameters instead of literal values.

Key facts:
- Correct traversal: (Program)-[:HAS_SEMESTER]->(Semester)-[:HAS_SUBJECT]->(Subject)
- Program must be matched with exact string literal, not a parameter
- s.number must always be wrapped with toInteger()
- Correct properties: p.ten_chuong_trinh, sub.ma_hp, sub.ten_hoc_phan, sub.so_tin_chi

Return errors (empty list if valid) and any filters found on node properties."""),
    ("human", "Validate this Cypher query."),
]) | get_llm().with_structured_output(CypherEvaluation)


_correct_cypher_chain = ChatPromptTemplate.from_messages([
    ("system", """You are a Cypher query correction expert for curriculum databases.

SCHEMA:
{schema}

RETRIEVAL TARGET:
{task_target}

INCORRECT CYPHER:
{cypher}

ERRORS FOUND:
{errors}

Fix the Cypher statement. It must:
- Traverse: (Program)-[:HAS_SEMESTER]->(Semester)-[:HAS_SUBJECT]->(Subject)
- Use exact string literal for p.ten_chuong_trinh (no $parameters)
- Use toInteger(s.number) for semester filtering and returning
- Return: program, semester, code, name, credits
- Order by: toInteger(s.number), sub.ma_hp
- Contain no $parameters or placeholders of any kind

Return ONLY the corrected Cypher query, no explanation, no markdown fences."""),
    ("human", "Correct the Cypher query."),
]) | get_llm() | StrOutputParser()


# Helpers

def _strip_cypher(cypher: str) -> str:
    cypher = cypher.strip()
    cypher = re.sub(r"^```(?:cypher)?\s*", "", cypher)
    cypher = re.sub(r"\s*```$", "", cypher)
    return cypher.strip()


# Node functions

def extract_time_scope(state: OverallState) -> OverallState:
    time_scope = _extract_time_scope_chain.invoke({
        "plan_preferences": state["plan_preferences"] or "No preferences provided.",
        "current_semester": state["current_semester"],
        "current_semester_plus_1": state["current_semester"] + 1,
    })
    return {"time_scope": time_scope, "steps": ["extract_time_scope"]}


def task_target(state: OverallState) -> OverallState:
    target = _task_target_chain.invoke({
        "program": state["program"],
        "current_semester": state["current_semester"],
        "time_scope": state["time_scope"],
    })
    return {"task_target": target, "steps": ["task_target"]}


def task_allocation(state: OverallState) -> OverallState:
    allocation = _task_allocation_chain.invoke({
        "task_target": state["task_target"],
        "schema": get_graph().schema,
    })
    return {"task_allocation": allocation, "steps": ["task_allocation"]}


def generate_cypher(state: OverallState) -> OverallState:
    NL = "\n"
    fewshot = (NL * 2).join([
        f"Question: {ex['question']}{NL}Cypher: {ex['query']}"
        for ex in get_example_selector().select_examples({"question": state["task_target"]})
    ])
    cypher = _text2cypher_chain.invoke({
        "task_target": state["task_target"],
        "task_allocation": state["task_allocation"],
        "fewshot_examples": fewshot,
        "schema": get_graph().schema,
    })
    return {"cypher_statement": _strip_cypher(cypher), "steps": ["generate_cypher"]}


def validate_cypher(state: OverallState) -> OverallState:
    errors = []
    mapping_errors = []
    cypher = _strip_cypher(state["cypher_statement"])

    # Check for syntax errors
    try:
        get_graph().query(cypher)
    except CypherSyntaxError as e:
        errors.append(str(e.message))

    corrected = get_cypher_corrector()(cypher)
    if not corrected:
        errors.append("The generated Cypher statement doesn't fit the graph schema")
    if corrected and corrected != cypher:
        print("Relationship direction was corrected")
    cypher = corrected or cypher

    llm_eval = _validate_cypher_chain.invoke({
        "task_target": state["task_target"],
        "schema": get_graph().schema,
        "cypher": cypher,
    })
    if llm_eval.errors:
        errors.extend(llm_eval.errors)

    # Check property mappings
    if llm_eval.filters:
        for f in llm_eval.filters:
            node_props = get_graph().structured_schema["node_props"].get(f.node_label, [])
            matching_props = [p for p in node_props if p["property"] == f.property_key]
            if not matching_props:
                continue
            if matching_props[0]["type"] != "STRING":
                continue
            found = get_graph().query(
                f"MATCH (n:{f.node_label}) WHERE toLower(n.`{f.property_key}`) CONTAINS toLower($value) RETURN 'yes' LIMIT 1",
                {"value": f.property_value},
            )
            if not found:
                print(f"Missing value: {f.node_label}.{f.property_key} = {f.property_value}")
                mapping_errors.append(
                    f"Missing value mapping for {f.node_label}.{f.property_key} = {f.property_value}"
                )

    if mapping_errors:
        next_action = "end"
    elif errors:
        next_action = "correct_cypher"
    else:
        next_action = "execute_cypher"

    return {
        "next_action": next_action,
        "cypher_statement": cypher,
        "cypher_errors": errors + mapping_errors,
        "steps": ["validate_cypher"],
    }


def correct_cypher(state: OverallState) -> OverallState:
    corrected = _correct_cypher_chain.invoke({
        "errors": state.get("cypher_errors", []),
        "cypher": state["cypher_statement"],
        "task_target": state["task_target"],
        "schema": get_graph().schema,
    })
    return {
        "next_action": "validate_cypher",
        "cypher_statement": _strip_cypher(corrected),
        "correction_attempts": state.get("correction_attempts", 0) + 1,
        "steps": ["correct_cypher"],
    }


def execute_cypher(state: OverallState) -> OverallState:
    records = get_graph().query(state["cypher_statement"])
    return {
        "database_records": records or [],
        "next_action": "end",
        "steps": ["execute_cypher"],
    }


def build_output(state: OverallState) -> OutputState:
    records = state.get("database_records") or []
    subject_codes = [r["code"] for r in records if r.get("code")]
    dependencies = _retrieve_dependencies(subject_codes)
    return {
        "cypher_statement": state.get("cypher_statement", ""),
        "database_records": records,
        "dependencies": dependencies,
        "errors": state.get("cypher_errors", []),
    }


# Deterministic dependency traversal

_DEPENDENCY_CYPHER = """
MATCH (a:Subject)-[r:PREREQUISITE_OF|COREQUISITE_WITH]->(b:Subject)
WHERE a.ma_hp IN $codes AND b.ma_hp IN $codes
RETURN a.ma_hp AS from_code, type(r) AS rel_type, b.ma_hp AS to_code
"""


def _retrieve_dependencies(subject_codes: list[str]) -> dict:
    if not subject_codes:
        return {"prerequisites": [], "corequisites": []}

    rows = get_graph().query(_DEPENDENCY_CYPHER, {"codes": subject_codes})

    prerequisites = [
        {"from": r["from_code"], "to": r["to_code"]}
        for r in rows if r["rel_type"] == "PREREQUISITE_OF"
    ]

    seen: set[tuple] = set()
    corequisites = []
    for r in rows:
        if r["rel_type"] == "COREQUISITE_WITH":
            key = tuple(sorted([r["from_code"], r["to_code"]]))
            if key not in seen:
                seen.add(key)
                corequisites.append({"subjects": list(key)})

    return {"prerequisites": prerequisites, "corequisites": corequisites}


# Conditional edges

def _after_validate(state: OverallState) -> Literal["correct_cypher", "execute_cypher", "build_output"]:
    if state["next_action"] == "end":
        return "build_output"
    if state.get("correction_attempts", 0) >= 3:
        return "build_output"
    if state["next_action"] == "correct_cypher":
        return "correct_cypher"
    return "execute_cypher"


# LangGraph

def _build_graph():
    builder = StateGraph(OverallState, input=InputState, output=OutputState)

    builder.add_node(extract_time_scope)
    builder.add_node(task_target)
    builder.add_node(task_allocation)
    builder.add_node(generate_cypher)
    builder.add_node(validate_cypher)
    builder.add_node(correct_cypher)
    builder.add_node(execute_cypher)
    builder.add_node(build_output)

    builder.add_edge(START, "extract_time_scope")
    builder.add_edge("extract_time_scope", "task_target")
    builder.add_edge("task_target", "task_allocation")
    builder.add_edge("task_allocation", "generate_cypher")
    builder.add_edge("generate_cypher", "validate_cypher")
    builder.add_conditional_edges("validate_cypher", _after_validate)
    builder.add_edge("correct_cypher", "validate_cypher")
    builder.add_edge("execute_cypher", "build_output")
    builder.add_edge("build_output", END)

    return builder.compile()


_langgraph = _build_graph()


# Public run()

async def run(
    emit,
    program: str = "",
    current_semester: int = 1,
    plan_preferences: str = "",
) -> dict:
    """
    Retrieve all subjects within the planning scope and their dependency graph.
    """
    program = resolve_program_name(program)
    await emit("prepare_query", f"Program: {program} | Current semester: {current_semester}")
    await emit("run_agent", "Running curriculum pipeline…")

    result = await asyncio.to_thread(
        _langgraph.invoke,
        {
            "program":          program,
            "current_semester": current_semester,
            "plan_preferences": plan_preferences,
        },
    )

    records = result.get("database_records", [])
    deps = result.get("dependencies", {})
    await emit("format_results", f"{len(records)} subjects retrieved")
    await emit(
        "format_results",
        f"{len(deps.get('prerequisites', []))} prerequisite pairs, "
        f"{len(deps.get('corequisites', []))} corequisite pairs",
    )

    return result


get_llm()