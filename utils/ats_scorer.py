from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_ats_score(resume_text, jd_text, matched_skills, jd_skills):
    # --- TF-IDF Similarity ---
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    tfidf_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    # --- Skill Match Ratio ---
    if len(jd_skills) == 0:
        skill_score = 0
    else:
        skill_score = len(matched_skills) / len(jd_skills)

    # --- Final Weighted ATS Score ---
    final_score = (
        0.45 * skill_score +     # skills matter most
        0.35 * tfidf_score +     # semantic similarity
        0.20 * min(skill_score * 1.2, 1.0)  # skill boost
    )

    return round(final_score * 100, 2)
