import os
import time
import uuid
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
)

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
QDRANT_URL      = os.getenv("QDRANT_URL")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "Job_Description"
EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"
BATCH_SIZE      = 50
MAX_RETRIES     = 3
RETRY_DELAY     = 5
CSV_PATH        = "VietJobs.csv"
CHECKPOINT_FILE = "checkpoint.txt"

SEMANTIC_FIELDS = [
    "job_title",
    "description",
    "requirements_text",
    "technical_skills",
    "soft_skills",
    "category",
]

METADATA_FIELDS = [
    "location",
    "country",
    "qualifications",
    "languages_required",
    "experience_required",
    "salary",
    "contract_type",
    "working_hours",
    "benefits",
    "salary_min",
    "salary_max",
    "salary_avg",
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def build_semantic_text(row: pd.Series) -> str:
    parts = []
    for field in SEMANTIC_FIELDS:
        value = row.get(field, "")
        if pd.notna(value) and str(value).strip():
            label = field.replace("_", " ").title()
            parts.append(f"{label}: {str(value).strip()}")
    return "\n".join(parts)


def build_payload(row: pd.Series) -> dict:
    payload = {}
    for field in SEMANTIC_FIELDS + METADATA_FIELDS:
        value = row.get(field, None)
        payload[field] = None if pd.isna(value) else str(value).strip()

    # numeric salary fields
    for num_field in ["salary_min", "salary_max", "salary_avg"]:
        try:
            payload[num_field] = float(row[num_field]) if pd.notna(row.get(num_field)) else None
        except (ValueError, TypeError):
            payload[num_field] = None

    return payload


def load_checkpoint() -> int:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return int(f.read().strip())
    return 0


def save_checkpoint(index: int):
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(str(index))


def ensure_collection(client: QdrantClient, vector_size: int):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        for field in ["location", "country", "category", "contract_type",
                      "experience_required", "salary_min", "salary_max"]:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD
                if field not in ("salary_min", "salary_max")
                else PayloadSchemaType.FLOAT,
            )
        print(f"Collection '{COLLECTION_NAME}' created.")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists. Resuming.")


def upsert_with_retry(client: QdrantClient, points: list, batch_start: int, batch_end: int) -> bool:
    for attempt in range(MAX_RETRIES):
        try:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=False,
            )
            save_checkpoint(batch_end)
            return True
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"\nRetrying batch {batch_start}–{batch_end} "
                      f"(attempt {attempt + 2}/{MAX_RETRIES}): {e}")
                time.sleep(RETRY_DELAY)
            else:
                print(f"\nFailed after {MAX_RETRIES} attempts at batch "
                      f"{batch_start}–{batch_end}: {e}")
                print("Progress saved. Re-run the script to resume.")
                return False


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    vector_size = model.get_embedding_dimension()

    print("Connecting to Qdrant Cloud...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)
    ensure_collection(client, vector_size)

    print(f"Reading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df = df.reset_index(drop=True)
    total = len(df)

    start_index = load_checkpoint()
    if start_index > 0:
        print(f"Resuming from row {start_index}/{total}")

    batches = range(start_index, total, BATCH_SIZE)
    for batch_start in tqdm(batches, desc="Embedding batches"):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch     = df.iloc[batch_start:batch_end]

        texts      = [build_semantic_text(row) for _, row in batch.iterrows()]
        embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i].tolist(),
                payload=build_payload(row),
            )
            for i, (_, row) in enumerate(batch.iterrows())
        ]

        success = upsert_with_retry(client, points, batch_start, batch_end)
        if not success:
            break

    print(f"\nDone. Total points in collection: "
          f"{client.count(collection_name=COLLECTION_NAME).count}")

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)


if __name__ == "__main__":
    main()