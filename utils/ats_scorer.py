from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_ats_score(
    resume_text: str,
    jd_text: str,
    matched_skills: set,
    jd_skills: set
) -> float:
    """
    Hybrid ATS score:
    60% Skill Match Score
    40% Text Similarity Score (TF-IDF)
    """

    # ---------- 1. SKILL MATCH SCORE ----------
    if len(jd_skills) == 0:
        skill_score = 0
    else:
        skill_score = (len(matched_skills) / len(jd_skills)) * 100

    # ---------- 2. TEXT SIMILARITY SCORE ----------
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.85,
        min_df=1
    )

    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    similarity_score = similarity * 100

    # ---------- 3. FINAL HYBRID SCORE ----------
    final_score = (0.6 * skill_score) + (0.4 * similarity_score)

    # Clamp to 100
    final_score = min(round(final_score, 2), 100.0)

    return final_score
