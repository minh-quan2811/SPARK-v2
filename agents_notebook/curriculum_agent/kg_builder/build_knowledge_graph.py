"""
DUT Curriculum → Neo4j Knowledge Graph Builder (Fixed Structure)

python build_knowledge_graph_fixed.py                          # default file
python build_knowledge_graph_fixed.py dut_curriculum_108.json
python build_knowledge_graph_fixed.py --uri neo4j://127.0.0.1:7687 --password 12345678 --wipe data.json
"""

import json
import sys
import argparse
import re
from pathlib import Path

from neo4j import GraphDatabase

# CLI
def parse_args():
    p = argparse.ArgumentParser(description="Load DUT curriculum JSON into Neo4j")
    p.add_argument("json_file", nargs="?", default="dut_curriculum_ALL.json")
    p.add_argument("--uri",      default="neo4j://127.0.0.1:7687")
    p.add_argument("--user",     default="neo4j")
    p.add_argument("--password", default="password")
    p.add_argument("--database", default="neo4j")
    p.add_argument("--wipe",     action="store_true",
                   help="Delete all nodes before importing")
    return p.parse_args()


# Helpers
def clean(text):
    """Remove extra whitespace from text"""
    return (text or "").strip()


def extract_subject_codes(text):
    """
    Extract 7-digit subject codes from prerequisite/corequisite fields.
    Example: "3190111- Giải tích 1, 3050011- Vật lý 1" → ["3190111", "3050011"]
    """
    if not text:
        return []
    codes = re.findall(r'\b(\d{7})\b', text)
    return list(dict.fromkeys(codes))  # Remove duplicates while keeping order


# Schema setup
def create_constraints(session):
    """Create uniqueness constraints for each node type"""
    
    # First, drop any old wrong constraints
    print("  Checking for old constraints …")
    try:
        # Drop the wrong semester_num constraint if it exists
        session.run("DROP CONSTRAINT semester_num IF EXISTS")
        print("    ✓ Dropped old semester_num constraint")
    except:
        pass
    
    constraints = [
        ("faculty_name",  "Faculty",  "name"),
        ("program_code",  "Program",  "ma_ctdt"),
        ("semester_id",   "Semester", "id"),  # Must be 'id' not 'number' - multiple programs share same number
        ("subject_code",  "Subject",  "ma_hp"),
    ]
    
    print("  Creating constraints …")
    for cname, label, prop in constraints:
        stmt = (f"CREATE CONSTRAINT {cname} IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE")
        try:
            session.run(stmt)
            print(f"    ✓ {label}.{prop}")
        except Exception as e:
            print(f"    ! {cname} already exists or error: {e}")


def wipe_database(session):
    """Delete all nodes and relationships"""
    print("  Wiping existing graph …")
    session.run("MATCH (n) DETACH DELETE n")
    print("    ✓ Database cleared")



# Load data into Neo4j
def load_program(tx, prog):
    """
    Load one program and all its structure:
      Faculty → Program → Semester → Subject
    """
    
    # Extract fields
    faculty_name = clean(prog.get("khoa_quan_ly", ""))
    ma_ctdt = clean(prog.get("ma_ctdt", ""))
    
    if not faculty_name or not ma_ctdt:
        return
    
    # 1. Create Faculty node
    tx.run("""
        MERGE (f:Faculty {name: $name})
    """, name=faculty_name)
    
    # 2. Create Program node with properties
    tx.run("""
        MERGE (p:Program {ma_ctdt: $ma_ctdt})
        SET p.stt = $stt,
            p.ma_nganh = $ma_nganh,
            p.ten_chuong_trinh = $ten_chuong_trinh,
            p.ten_nganh = $ten_nganh,
            p.chuyen_nganh = $chuyen_nganh,
            p.so_tin_chi = $so_tin_chi,
            p.ngon_ngu = $ngon_ngu,
            p.so_ky = $so_ky,
            p.tu = $tu,
            p.den = $den,
            p.status = $status
    """, 
        ma_ctdt=ma_ctdt,
        stt=clean(prog.get("stt", "")),
        ma_nganh=clean(prog.get("ma_nganh", "")),
        ten_chuong_trinh=clean(prog.get("ten_chuong_trinh", "")),
        ten_nganh=clean(prog.get("ten_nganh", "")),
        chuyen_nganh=clean(prog.get("chuyen_nganh", "")),
        so_tin_chi=clean(prog.get("so_tin_chi", "")),
        ngon_ngu=clean(prog.get("ngon_ngu", "")),
        so_ky=clean(prog.get("so_ky", "")),
        tu=clean(prog.get("tu", "")),
        den=clean(prog.get("den", "")),
        status=clean(prog.get("status", ""))
    )
    
    # 3. Create Faculty → Program relationship
    tx.run("""
        MATCH (f:Faculty {name: $faculty})
        MATCH (p:Program {ma_ctdt: $ma_ctdt})
        MERGE (f)-[:HAS_PROGRAM]->(p)
    """, faculty=faculty_name, ma_ctdt=ma_ctdt)
    
    # 4. Process subjects and create Semester nodes
    subjects = prog.get("subjects", [])
    
    # Group subjects by semester
    semesters = {}
    for subj in subjects:
        hoc_ky = clean(subj.get("hoc_ky", ""))
        if hoc_ky and hoc_ky.isdigit():
            sem_num = int(hoc_ky)
            if sem_num not in semesters:
                semesters[sem_num] = []
            semesters[sem_num].append(subj)
    
    # Create Semester nodes and relationships
    for sem_num in sorted(semesters.keys()):
        semester_id = f"{ma_ctdt}_sem_{sem_num}"
        
        # Create Semester node
        tx.run("""
            MERGE (s:Semester {id: $id})
            SET s.number = $number,
                s.program_code = $program_code
        """, id=semester_id, number=sem_num, program_code=ma_ctdt)
        
        # Create Program → Semester relationship
        tx.run("""
            MATCH (p:Program {ma_ctdt: $ma_ctdt})
            MATCH (s:Semester {id: $semester_id})
            MERGE (p)-[:HAS_SEMESTER]->(s)
        """, ma_ctdt=ma_ctdt, semester_id=semester_id)
        
        # Create Subject nodes for this semester
        for subj in semesters[sem_num]:
            ma_hp = clean(subj.get("ma_hp", ""))
            if not ma_hp:
                continue
            
            # Create Subject node
            tx.run("""
                MERGE (subj:Subject {ma_hp: $ma_hp})
                SET subj.ten_hoc_phan = $ten_hoc_phan,
                    subj.so_tin_chi = $so_tin_chi,
                    subj.tu_chon = $tu_chon,
                    subj.ht_da = $ht_da,
                    subj.tq_da = $tq_da,
                    subj.tt = $tt,
                    subj.ky_hieu = $ky_hieu
            """,
                ma_hp=ma_hp,
                ten_hoc_phan=clean(subj.get("ten_hoc_phan", "")),
                so_tin_chi=clean(subj.get("so_tin_chi", "")),
                tu_chon=clean(subj.get("tu_chon", "")),
                ht_da=clean(subj.get("ht_da", "")),
                tq_da=clean(subj.get("tq_da", "")),
                tt=clean(subj.get("tt", "")),
                ky_hieu=clean(subj.get("ky_hieu", ""))
            )
            
            # Create Semester → Subject relationship
            tx.run("""
                MATCH (sem:Semester {id: $semester_id})
                MATCH (subj:Subject {ma_hp: $ma_hp})
                MERGE (sem)-[:HAS_SUBJECT]->(subj)
            """, semester_id=semester_id, ma_hp=ma_hp)

# Second pass: Create subject relationship (prerequisites, corequisites, etc.)

def link_subject_relationships(session, all_programs):
    """
    Create relationships between subjects based on:
      - tien_quyet: Prerequisite (must complete before)
      - song_hanh: Corequisite (must take at same time)
      - hoc_truoc: Recommended prerequisite (should take before)
    """
    
    print("  Creating subject relationships …")
    
    # Collect all valid subject codes
    all_subject_codes = set()
    for prog in all_programs:
        for subj in prog.get("subjects", []):
            ma_hp = clean(subj.get("ma_hp", ""))
            if ma_hp:
                all_subject_codes.add(ma_hp)
    
    counts = {
        "prerequisite": 0,
        "corequisite": 0,
        "recommended": 0,
        "skipped": 0
    }
    
    with session.begin_transaction() as tx:
        for prog in all_programs:
            for subj in prog.get("subjects", []):
                ma_hp = clean(subj.get("ma_hp", ""))
                if not ma_hp:
                    continue
                
                # 1. Prerequisite (tien_quyet): Required before this subject
                #    Direction: prerequisite → this_subject
                for prereq_code in extract_subject_codes(subj.get("tien_quyet", "")):
                    if prereq_code not in all_subject_codes:
                        counts["skipped"] += 1
                        continue
                    
                    tx.run("""
                        MATCH (prereq:Subject {ma_hp: $prereq_code})
                        MATCH (current:Subject {ma_hp: $current_code})
                        MERGE (prereq)-[:PREREQUISITE_OF]->(current)
                    """, prereq_code=prereq_code, current_code=ma_hp)
                    counts["prerequisite"] += 1
                
                # 2. Corequisite (song_hanh): Must take at same time
                #    Direction: bidirectional
                for coreq_code in extract_subject_codes(subj.get("song_hanh", "")):
                    if coreq_code == ma_hp or coreq_code not in all_subject_codes:
                        counts["skipped"] += 1
                        continue
                    
                    tx.run("""
                        MATCH (a:Subject {ma_hp: $code_a})
                        MATCH (b:Subject {ma_hp: $code_b})
                        MERGE (a)-[:COREQUISITE_WITH]->(b)
                        MERGE (b)-[:COREQUISITE_WITH]->(a)
                    """, code_a=ma_hp, code_b=coreq_code)
                    counts["corequisite"] += 1
                
                # 3. Recommended prerequisite (hoc_truoc): Should take before
                #    Direction: recommended → this_subject
                for rec_code in extract_subject_codes(subj.get("hoc_truoc", "")):
                    if rec_code not in all_subject_codes:
                        counts["skipped"] += 1
                        continue
                    
                    tx.run("""
                        MATCH (rec:Subject {ma_hp: $rec_code})
                        MATCH (current:Subject {ma_hp: $current_code})
                        MERGE (rec)-[:RECOMMENDED_BEFORE]->(current)
                    """, rec_code=rec_code, current_code=ma_hp)
                    counts["recommended"] += 1
        
        tx.commit()
    
    print(f"    ✓ {counts['prerequisite']} PREREQUISITE_OF")
    print(f"    ✓ {counts['corequisite']} COREQUISITE_WITH")
    print(f"    ✓ {counts['recommended']} RECOMMENDED_BEFORE")
    print(f"    ! {counts['skipped']} skipped (subject not found)")


# Main
def main():
    args = parse_args()
    
    # Check if file exists
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"✗ File not found: {json_path}")
        sys.exit(1)
    
    # Load JSON data
    print(f"[1] Loading {json_path} …")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"    ✓ {len(data)} program(s) loaded")
    
    # Connect to Neo4j
    print(f"\n[2] Connecting to Neo4j at {args.uri} …")
    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    
    try:
        driver.verify_connectivity()
        print("    ✓ Connected")
    except Exception as e:
        print(f"    ✗ Connection failed: {e}")
        sys.exit(1)
    
    # Create database structure
    with driver.session(database=args.database) as session:
        
        print("\n[3] Setting up schema …")
        create_constraints(session)
        
        if args.wipe:
            print("\n[4] Clearing database …")
            wipe_database(session)
        
        print("\n[5] Loading data …")
        success = 0
        errors = 0
        
        for i, prog in enumerate(data, 1):
            ma_ctdt = prog.get("ma_ctdt", f"unknown_{i}")
            ten_ct = prog.get("ten_chuong_trinh", "")
            subject_count = len(prog.get("subjects", []))
            
            try:
                with session.begin_transaction() as tx:
                    load_program(tx, prog)
                    tx.commit()
                
                print(f"  [{i:>4}/{len(data)}] ✓ {ma_ctdt}  ({subject_count} subjects)  {ten_ct[:50]}")
                success += 1
                
            except Exception as e:
                print(f"  [{i:>4}/{len(data)}] ✗ {ma_ctdt}  ERROR: {e}")
                errors += 1
        
        print(f"\n    ✓ {success} programs loaded")
        if errors > 0:
            print(f"    ✗ {errors} programs failed")
        
        print("\n[6] Creating subject relationships …")
        link_subject_relationships(session, data)
    
    driver.close()
    
    print("\n" + "="*70)
    print("✓ Knowledge graph built successfully!")
    print("="*70)
    print("\nGraph structure:")
    print("  Faculty → Program → Semester → Subject")
    print("\nSubject relationships:")
    print("  PREREQUISITE_OF    (required before)")
    print("  COREQUISITE_WITH   (take at same time)")
    print("  RECOMMENDED_BEFORE (should take before)")
    print("\nNext steps:")
    print("  1. Open Neo4j Browser")
    print("  2. Run: MATCH (n) RETURN n LIMIT 100")
    print("  3. Or try: MATCH path = (f:Faculty)-[:HAS_PROGRAM]->(p:Program)")
    print("             -[:HAS_SEMESTER]->(s:Semester)-[:HAS_SUBJECT]->(subj:Subject)")
    print("             RETURN path LIMIT 25")
    print("="*70)


if __name__ == "__main__":
    main()