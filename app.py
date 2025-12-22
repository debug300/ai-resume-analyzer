import os
from flask import Flask, render_template, request

from utils.resume_parser import extract_text_from_pdf, clean_text
from utils.skill_extractor import extract_skills
from utils.ats_scorer import calculate_ats_score

app = Flask(__name__)

# ✅ Absolute uploads path      (FIXES RENDER)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
SKILLS = [
    "python", "java", "c", "c++", "sql", "flask", "django",
    "machine learning", "git", "docker", "aws", "react"
]


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        resume_file = request.files["resume"]
        jd_text = request.form["jd"]

        if resume_file.filename == "":
            return render_template("index.html", score=None, error="Please upload a resume")

        resume_path = os.path.join(app.config["UPLOAD_FOLDER"], resume_file.filename)
        resume_file.save(resume_path)

        resume_text = extract_text_from_pdf(resume_path)
        jd_clean = clean_text(jd_text)

        resume_skills = extract_skills(resume_text, SKILLS)
        jd_skills = extract_skills(jd_clean, SKILLS)


        matched = resume_skills & jd_skills
        missing = jd_skills - resume_skills

        score = calculate_ats_score(
    resume_text,
    jd_clean,
    matched,
    jd_skills
)


        return render_template(
            "index.html",
            score=score,
            matched=matched,
            missing=missing
        )

    return render_template("index.html", score=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
# redeploy trigger
