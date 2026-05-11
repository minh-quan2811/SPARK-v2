def _extract_skill_gaps(cv_data, job_data):
    current = set(s.lower() for s in cv_data.get("technical_skills", []))
    required = set()
    for job in job_data.get("jobs", [])[:5]:
        for skill in job.get("technical_skills") or []:
            required.add(str(skill).lower())
    gaps = sorted(s for s in required if s not in current)
    return gaps[:20]

def _extract_course_names(curriculum_data):
    names = []
    for record in curriculum_data.get("courses", [])[:20]:
        if isinstance(record, dict):
            for value in record.values():
                if isinstance(value, str) and len(value) < 120:
                    names.append(value)
                    break
    seen = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out[:10]

def run(cv_data, job_data, curriculum_data, user_data):
    skill_gaps = _extract_skill_gaps(cv_data, job_data)
    courses = _extract_course_names(curriculum_data)

    md = []
    md.append("# Personalized Career Roadmap\n")
    md.append("## Student Profile")
    md.append(f"- Faculty: {user_data.get('faculty', '')}")
    md.append(f"- Year: {user_data.get('year', '')}")
    md.append("")

    md.append("## Current Technical Skills")
    for skill in cv_data.get("technical_skills", [])[:20]:
        md.append(f"- {skill}")
    md.append("")

    md.append("## Recommended Target Jobs")
    for title in job_data.get("top_job_titles", [])[:5]:
        md.append(f"- {title}")
    md.append("")

    md.append("## Skill Gaps")
    if skill_gaps:
        for gap in skill_gaps:
            md.append(f"- {gap.title()}")
    else:
        md.append("- No major skill gaps identified.")
    md.append("")

    md.append("## Recommended Courses")
    for course in courses:
        md.append(f"- {course}")
    md.append("")

    md.append("## 6-Month Action Plan")
    md.append("### Months 1-2")
    md.append("- Complete 2-3 foundational courses.")
    md.append("- Strengthen missing technical skills.")
    md.append("")
    md.append("### Months 3-4")
    md.append("- Build one portfolio project aligned with target jobs.")
    md.append("- Update CV and GitHub.")
    md.append("")
    md.append("### Months 5-6")
    md.append("- Apply for internships and entry-level positions.")
    md.append("- Practice interviews and coding tests.")
    md.append("")

    md.append("## Personalized Preferences")
    prefs = user_data.get("plan_preferences") or user_data.get("preferences")
    if prefs:
        md.append(prefs)
    else:
        md.append("No additional preferences provided.")

    return "\n".join(md)