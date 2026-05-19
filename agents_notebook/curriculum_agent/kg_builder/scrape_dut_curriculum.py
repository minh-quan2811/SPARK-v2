"""
DUT Curriculum Scraper - https://sv.dut.udn.vn/G_ListCTDT.aspx

python scrape_dut_curriculum.py          # all faculties
python scrape_dut_curriculum.py 108      # one faculty
"""

import requests
from bs4 import BeautifulSoup
import json, csv, time, sys

BASE_URL  = "https://sv.dut.udn.vn"
LIST_URL  = f"{BASE_URL}/G_ListCTDT.aspx"
FRAME_URL = f"{BASE_URL}/WebAjax/evLopHP_Load.aspx"

BROWSER_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Connection":      "keep-alive",
}


def make_session():
    s = requests.Session()
    s.headers.update(BROWSER_HEADERS)
    return s


# ─────────────────────────────────────────────────────────
# Step 1 – GET landing page → session cookie + VIEWSTATE
# ─────────────────────────────────────────────────────────
def get_initial_state(session):
    resp = session.get(LIST_URL, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    def hidden(id_):
        tag = soup.find("input", {"id": id_})
        return tag["value"] if tag else ""

    faculties = {o["value"]: o.text.strip()
                 for o in soup.select("#GListCTDT_cboKhoa option")}

    return {
        "viewstate":    hidden("__VIEWSTATE"),
        "viewstategen": hidden("__VIEWSTATEGENERATOR"),
        "faculties":    faculties,
    }


# ─────────────────────────────────────────────────────────
# Step 2 – POST with selected faculty → program rows
# Returns (programs, updated_soup) so we can re-use tokens
# ─────────────────────────────────────────────────────────
def get_program_list(session, state, faculty_code):
    payload = {
        "__VIEWSTATE":          state["viewstate"],
        "__VIEWSTATEGENERATOR": state["viewstategen"],
        "_ctl0:MainContent:GListCTDT_cboTrDo":   "Đại học",
        "_ctl0:MainContent:GListCTDT_cboKhoa":   faculty_code,
        "_ctl0:MainContent:GListCTDT_cboOrder":  "Xếp theo Tên ngành, Tên CTĐT",
        "_ctl0:MainContent:GListCTDT_btnDuLieu": "Dữ liệu",
    }
    resp = session.post(LIST_URL, data=payload, headers={"Referer": LIST_URL}, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Capture refreshed VIEWSTATE from the POST response
    def hidden(id_):
        tag = soup.find("input", {"id": id_})
        return tag["value"] if tag else state[id_.lower().replace("__","")]

    state["viewstate"]    = hidden("__VIEWSTATE")
    state["viewstategen"] = hidden("__VIEWSTATEGENERATOR")

    programs = []
    table = soup.find("table", {"id": "CTDTGridInfo"})
    if not table:
        print("  [!] No program table in response.")
        return programs

    for row in table.select("tr.GridRow"):
        cells = row.find_all("td")
        if len(cells) < 13:
            continue
        # onclick="javascript:CTDT_LoadKhung('1081004','SPKT Công nghiệp K2013');return false;"
        onclick = row.get("onclick", "")
        parts   = onclick.split("'")
        ma_ctdt = parts[1] if len(parts) > 1 else cells[5].text.strip()
        ten_ct  = parts[3] if len(parts) > 3 else cells[3].text.strip()

        programs.append({
            "stt":              cells[0].text.strip(),
            "ma_nganh":         cells[1].text.strip(),
            "ten_nganh":        cells[2].text.strip(),
            "ten_chuong_trinh": ten_ct,
            "chuyen_nganh":     cells[4].text.strip(),
            "ma_ctdt":          ma_ctdt,
            "so_tin_chi":       cells[6].text.strip(),
            "ngon_ngu":         cells[7].text.strip(),
            "so_ky":            cells[8].text.strip(),
            "tu":               cells[9].text.strip(),
            "den":              cells[10].text.strip(),
            "khoa_quan_ly":     cells[12].text.strip(),
        })

    return programs


# ─────────────────────────────────────────────────────────
# Step 3 – "Click" a program row → load its curriculum.
#
# From Public.js: $.ajaxSetup({
#   url: "WebAjax/evLopHP_Load.aspx?E=G_KhungCTDT&MaNganh=" + MaCTDT,
#   type: "POST"
# })
# So: POST /WebAjax/evLopHP_Load.aspx?E=G_KhungCTDT&MaNganh=1081004
# ─────────────────────────────────────────────────────────
def get_curriculum(session, ma_ctdt):
    url = FRAME_URL
    params = {"E": "G_KhungCTDT", "MaNganh": ma_ctdt}
    headers = {
        "Referer": LIST_URL,
        "Accept":  "*/*",
        "X-Requested-With": "XMLHttpRequest",
    }

    # JS uses $.ajax POST (type: "POST" from ajaxSetup)
    resp = session.post(url, params=params, data={}, headers=headers, timeout=15)

    if resp.status_code == 404:
        return {"summary": {}, "subjects": [], "status": "not_published"}

    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    if not soup.find("table", {"id": "G_KhungCTDT_Grid"}):
        return {"summary": {}, "subjects": [], "status": "empty"}

    # ── Summary (top small table) ──
    summary = {}
    grid0 = soup.find("table", {"id": "G_KhungCTDT_Grid0"})
    if grid0:
        r = grid0.find("tr", class_="GridRow")
        if r:
            c = r.find_all("td")
            summary = {
                "ma_ten_nganh": c[0].text.strip() if len(c) > 0 else "",
                "ma_ten_ctdt":  c[1].text.strip() if len(c) > 1 else "",
                "so_hoc_ky":    c[2].text.strip() if len(c) > 2 else "",
                "tong_tin_chi": c[3].text.strip() if len(c) > 3 else "",
                "tin_chi_bb":   c[4].text.strip() if len(c) > 4 else "",
                "tin_chi_tc":   c[5].text.strip() if len(c) > 5 else "",
            }

    # ── Subject rows ──
    subjects = []
    for row in soup.select("#G_KhungCTDT_Grid tr.GridRow"):
        c = row.find_all("td")
        if len(c) < 12:
            continue
        subjects.append({
            "tt":           c[0].text.strip(),
            "hoc_ky":       c[1].text.strip(),
            "ten_hoc_phan": c[2].text.strip(),
            "ky_hieu":      c[3].text.strip(),
            "ma_hp":        c[4].text.strip(),
            "so_tin_chi":   c[5].text.strip(),
            "tu_chon":      c[6].text.strip(),
            "ht_da":        "x" if "GridCheck" in c[7].get("class", []) else "",
            "tq_da":        "x" if "GridCheck" in c[8].get("class", []) else "",
            "hoc_truoc":    c[9].text.strip(),
            "song_hanh":    c[10].text.strip(),
            "tien_quyet":   c[11].text.strip(),
        })

    return {"summary": summary, "subjects": subjects, "status": "ok"}


# ─────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────
def scrape(faculty_code="ALL", delay=1.0):
    print("[1] Starting session and loading main page …")
    session = make_session()
    state   = get_initial_state(session)

    print("    Available faculties:")
    for code, name in state["faculties"].items():
        print(f"      {code:5s}  {name}")

    target_codes = (
        [c for c in state["faculties"] if c != "ALL"]
        if faculty_code == "ALL"
        else [faculty_code]
    )

    if faculty_code != "ALL" and faculty_code not in state["faculties"]:
        print(f"\n[!] Faculty code '{faculty_code}' not found.")
        return

    all_results = []

    for fac in target_codes:
        print(f"\n[2] Faculty {fac} – {state['faculties'][fac]}")
        programs = get_program_list(session, state, fac)
        print(f"    {len(programs)} program(s) found.")

        for i, prog in enumerate(programs, 1):
            ma  = prog["ma_ctdt"]
            ten = prog["ten_chuong_trinh"]
            print(f"  [3] ({i}/{len(programs)}) {ma} – {ten} … ", end="", flush=True)
            try:
                cur = get_curriculum(session, ma)
                prog.update(cur)
                if cur["status"] == "ok":
                    print(f"{len(cur['subjects'])} subjects")
                else:
                    print(cur["status"])
            except Exception as e:
                print(f"ERROR: {e}")
                prog.update({"summary": {}, "subjects": [], "status": "error"})

            all_results.append(prog)
            time.sleep(delay)

    # ── JSON ──
    out_json = f"dut_curriculum_{faculty_code}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n[✓] JSON → {out_json}")

    # ── CSV (flat, one row per subject) ──
    out_csv = f"dut_curriculum_{faculty_code}.csv"
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "status", "ma_ctdt", "ten_chuong_trinh", "ten_nganh", "chuyen_nganh",
            "so_tin_chi_ct", "ngon_ngu", "so_ky", "tu", "den", "khoa_quan_ly",
            "tt", "hoc_ky", "ten_hoc_phan", "ma_hp", "so_tin_chi_hp",
            "tu_chon", "ht_da", "tq_da", "hoc_truoc", "song_hanh", "tien_quyet",
        ])
        for p in all_results:
            base = [
                p.get("status",""), p["ma_ctdt"], p["ten_chuong_trinh"],
                p["ten_nganh"], p["chuyen_nganh"], p["so_tin_chi"],
                p["ngon_ngu"], p["so_ky"], p["tu"], p["den"], p["khoa_quan_ly"],
            ]
            for s in (p.get("subjects") or [{}]):
                w.writerow(base + [
                    s.get("tt",""), s.get("hoc_ky",""), s.get("ten_hoc_phan",""),
                    s.get("ma_hp",""), s.get("so_tin_chi",""), s.get("tu_chon",""),
                    s.get("ht_da",""), s.get("tq_da",""), s.get("hoc_truoc",""),
                    s.get("song_hanh",""), s.get("tien_quyet",""),
                ])

    print(f"[✓] CSV  → {out_csv}")
    return all_results


if __name__ == "__main__":
    fac = sys.argv[1] if len(sys.argv) > 1 else "ALL"
    scrape(faculty_code=fac)