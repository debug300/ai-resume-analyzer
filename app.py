from flask import Flask, render_template, request
import os

from utils.resume_parser import extract_text_from_pdf
from utils.jd_parser import parse_job_description
from utils.skill_extractor import extract_skills
from utils.ats_scorer import calculate_ats_score

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

SKILLS = [
    "python", "java", "c", "c++", "sql", "flask", "django",
    "machine learning", "git", "docker", "aws", "react"
]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # 1. Get user inputs
        resume_file = request.files["resume"]
        jd_text = request.form["jd"]

        # 2. Save resume
        resume_path = os.path.join("uploads", resume_file.filename)
        resume_file.save(resume_path)

        # 3. Extract & clean text
        resume_text = extract_text_from_pdf(resume_path)
        jd_clean = clean_text(jd_text)

        # 4. Skill extraction
        resume_skills = extract_skills(resume_text)
        jd_skills = extract_skills(jd_clean)

        matched = resume_skills.intersection(jd_skills)
        missing = jd_skills.difference(resume_skills)

        # 5. ATS Score
        score = calculate_ats_score(resume_text, jd_clean)

        # 6. Send data to UI
        return render_template(
            "index.html",
            score=score,
            matched=matched,
            missing=missing
        )

    # For first page load
    return render_template("index.html", score=None)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
