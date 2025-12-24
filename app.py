CORE_SKILLS = ["java", "spring boot", "sql"]
IMPORTANT_SKILLS = ["git", "aws", "docker"]
OPTIONAL_SKILLS = ["kubernetes", "microservices", "ci cd"]

import os
from flask import Flask, render_template, request

from utils.resume_parser import extract_text_from_pdf
from utils.skill_extractor import extract_skills
from utils.ats_scorer import calculate_ats_score
from utils.text_cleaner import clean_text

app = Flask(__name__)

# ---------- Upload folder (Render safe) ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------- Canonical Skill List ----------
SKILLS = [
    "python", "java", "c", "c++", "sql",
    "spring boot", "microservices", "rest api",
    "docker", "kubernetes",
    "aws", "azure", "gcp",
    "git",
    "machine learning", "deep learning", "ai",
    "data structures", "algorithms"
]


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        resume_file = request.files.get("resume")
        jd_text = request.form.get("jd", "")

        if not resume_file or resume_file.filename == "":
            return render_template("index.html", error="Please upload a resume")

        # Save resume
        resume_path = os.path.join(app.config["UPLOAD_FOLDER"], resume_file.filename)
        resume_file.save(resume_path)

        # Clean text
        resume_text = clean_text(extract_text_from_pdf(resume_path))
        jd_clean = clean_text(jd_text)

        # Skill extraction
        resume_skills = extract_skills(resume_text, SKILLS)
        jd_skills = extract_skills(jd_clean, SKILLS)

        matched = resume_skills & jd_skills
        missing = jd_skills - resume_skills

        # ATS score
        ats = calculate_ats_score(
            resume_text,
            jd_clean,
            matched,
            jd_skills
        )

        return render_template(
            "index.html",
            score=ats["final_score"],
            skill_score=ats["skill_score"],
            similarity_score=ats["similarity_score"],
            keyword_score=ats["keyword_score"],
            matched=matched,
            missing=missing
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
