import os
import asyncio
import json
from typing import Optional
from pydantic import BaseModel

from dotenv import load_dotenv

from langchain_community.graphs import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


# =========================
# Pydantic Schemas
# =========================

class CurriculumResult(BaseModel):
    query: str
    cypher_statement: str
    records: list[dict]
    error: Optional[str] = None


# =========================
# Lazy Initialization
# =========================

_graph = None
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

def get_graph():
    global _graph
    if _graph is None:
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_username = os.getenv("NEO4J_USERNAME")
        neo4j_password = os.getenv("NEO4J_PASSWORD")

        _graph = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
            enhanced_schema=True
        )
    return _graph


# =========================
# Text-to-Cypher Chain
# =========================

TEXT_TO_CYPHER_SYSTEM = """
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

TEXT_TO_CYPHER_HUMAN = """
USER QUESTION:
{question}
"""


def get_text_to_cypher_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", TEXT_TO_CYPHER_SYSTEM),
        ("human", TEXT_TO_CYPHER_HUMAN),
    ])
    return prompt | get_llm() | StrOutputParser()


# =========================
# Main Agent Function
# =========================

async def run(question: str, emit):
    """
    Query curriculum data from Neo4j graph database.

    Args:
        question: Natural language question about curriculum
        emit: Async callback function to emit progress updates

    Returns:
        dict with query, cypher_statement, records, and optional error
    """
    error_msg = None
    try:
        await emit("generate_cypher", "Generating Cypher query from question")

        graph = get_graph()
        chain = get_text_to_cypher_chain()

        # Generate Cypher statement
        cypher_statement = await asyncio.to_thread(
            chain.invoke,
            {
                "question": question,
                "schema": graph.schema,
            }
        )

        await emit("execute_query", "Executing query on Neo4j")

        # Execute Cypher query
        try:
            records = await asyncio.to_thread(graph.query, cypher_statement)
            if not isinstance(records, list):
                records = []
        except Exception as e:
            records = []
            error_msg = str(e)
            await emit("query_error", f"Query execution failed: {error_msg}")

        await emit("format_results", "Formatting results")

        return {
            "query": question,
            "cypher_statement": cypher_statement,
            "records": records,
            "error": error_msg,
        }

    except Exception as e:
        await emit("agent_error", f"Curriculum agent error: {str(e)}")
        return {
            "query": question,
            "cypher_statement": "",
            "records": [],
            "error": str(e),
        }

get_llm()

# =========================
# Test Case
# =========================
# if __name__ == "__main__":
#     import sys

#     async def test_emit(event: str, message: str):
#         print(f"  [{event}] {message}")

#     async def main():
#         # Example test question
#         test_question = "What are all the courses in the Công nghệ Thực phẩm K2020CLC program?"
        
#         try:
#             print("\n=== Starting Curriculum Query Test ===\n")
#             result = await run(test_question, test_emit)
            
#             print("\n=== Curriculum Query Complete ===\n")
#             print("FULL RESULT:")
#             print(json.dumps({
#                 "query": result['query'],
#                 "error": result['error'],
#                 "records_count": len(result['records'])
#             }, indent=2, ensure_ascii=False))
            
#             print("\n=== GENERATED CYPHER QUERY ===\n")
#             print(result['cypher_statement'])
            
#             print("\n=== QUERY EXECUTION STATUS ===\n")
#             if result['error']:
#                 print(f"ERROR: {result['error']}")
#             else:
#                 print("Status: Query executed successfully")
            
#             print(f"\n=== RETRIEVED RECORDS ({len(result['records'])} total) ===\n")
#             if result['records']:
#                 for i, record in enumerate(result['records'][:20], 1):
#                     print(f"[{i}] Record:")
#                     for key, value in record.items():
#                         # Truncate long values
#                         if isinstance(value, str) and len(value) > 100:
#                             print(f"    {key}: {value[:100]}...")
#                         else:
#                             print(f"    {key}: {value}")
#                     print()
#             else:
#                 print("No records retrieved from query")
            
#             print("\n=== RAW JSON RESULT ===\n")
#             print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
            
#         except Exception as e:
#             print(f"Test error: {e}")
#             import traceback
#             traceback.print_exc()

#     asyncio.run(main())