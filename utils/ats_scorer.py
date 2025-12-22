from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_ats_score(resume_text, jd_text):
    documents = [resume_text, jd_text]

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    score = round(similarity[0][0] * 100, 2)
    return score


if __name__ == "__main__":
    resume_text = """
    python developer with experience in flask sql and machine learning
    worked on git based projects
    """

    jd_text = """
    looking for a software engineer with python django sql and docker skills
    git experience required
    """

    score = calculate_ats_score(
    resume_text,
    jd_clean,
    resume_skills,
    jd_skills
)

    print(f"ATS Match Score: {ats_score}%")
