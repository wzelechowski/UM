from similarity.numeric_similarity import NumericSimilarity
from similarity.set_similarity import SetSimilarity
from similarity.text_cosine_similarity import TextCosineSimilarity

class ProcessingService:
    def __init__(self, component_similarities):
        self._component_similarities = component_similarities

    @classmethod
    def create_default(cls):
        set_similarity = SetSimilarity()
        text_similarity = TextCosineSimilarity()

        default_similarities = {
            'genres': set_similarity,
            'cast_list': set_similarity,
            'keywords': set_similarity,
            'production_companies': set_similarity,
            'release_year': NumericSimilarity(default_max_diff=50.0),
            'vote_average': NumericSimilarity(default_max_diff=5.0),
            'runtime': NumericSimilarity(default_max_diff=120.0),
            'title': text_similarity,
            'director': text_similarity,
            'original_language': text_similarity,
            'collection_name': text_similarity,
        }
        return cls(component_similarities=default_similarities)

    def precompute_component_matrices(self, df_raw):
        df = df_raw.set_index('movie_id') if 'movie_id' in df_raw.columns else df_raw.copy()
        movie_ids = df.index.tolist()

        print(f"Obliczanie macierzy podobieństw dla {len(movie_ids)} filmów i {len(self._component_similarities)} cech...")

        matrices = {
            comp: similarity.compute_matrix(df[comp])
            for comp, similarity in self._component_similarities.items()
            if comp in df.columns
        }

        return movie_ids, matrices