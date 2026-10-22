import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from similarity.similarity import Similarity

class TextCosineSimilarity(Similarity):

    def compute_matrix(self, series):
        cleaned_texts = series.fillna('').astype(str).str.strip().tolist()
        n = len(cleaned_texts)

        if not any(cleaned_texts):
            return np.eye(n, dtype=np.float32)

        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        tfidf_matrix = vectorizer.fit_transform(cleaned_texts)

        sim_matrix = cosine_similarity(tfidf_matrix).astype(np.float32)
        np.fill_diagonal(sim_matrix, 1.0)
        return sim_matrix