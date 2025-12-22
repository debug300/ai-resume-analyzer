import os
from flask import Flask, render_template, request

from utils.resume_parser import extract_text_from_pdf, clean_text
from utils.skill_extractor import extract_skills
from utils.ats_scorer import calculate_ats_score

app = Flask(__name__)

# ---------- UPLOADS (RENDER SAFE) ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------- SKILLS DICTIONARY ----------
SKILLS = [
    "python", "java", "c", "c++", "sql", "flask", "django",
    "machine learning", "git", "docker", "aws", "react",
    "spring", "spring boot", "rest api", "microservices"
]


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        # --------- INPUTS ----------
        resume_file = request.files.get("resume")
        jd_text = request.form.get("jd", "")

        if not resume_file or resume_file.filename == "":
            return render_template(
                "index.html",
                score=None,
                error="Please upload a resume"
            )

        # --------- SAVE RESUME ----------
        resume_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            resume_file.filename
        )
        resume_file.save(resume_path)

        # --------- TEXT EXTRACTION ----------
        raw_resume_text = extract_text_from_pdf(resume_path)
        resume_text = clean_text(raw_resume_text)
        jd_clean = clean_text(jd_text)

        # --------- SKILL EXTRACTION ----------
        resume_skills = extract_skills(resume_text, SKILLS)
        jd_skills = extract_skills(jd_clean, SKILLS)

        matched = resume_skills & jd_skills
        missing = jd_skills - resume_skills

        # --------- ATS SCORE ----------
        score = calculate_ats_score(
            resume_text,
            jd_clean,
            matched,
            jd_skills
        )

        return render_template(
            "index.html",
            score=score,
            matched=sorted(matched),
            missing=sorted(missing)
        )

    return render_template("index.html", score=None)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
