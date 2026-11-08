class MovieService:
    def __init__(self, df_train, df_task, movie_repository, tmdb_client):
        self.df_train = df_train
        self.df_task = df_task
        self.movie_repository = movie_repository
        self.tmdb_client = tmdb_client

    def get_all_required_ids(self):
        return set(self.df_train['movie_id']).union(set(self.df_task['movie_id']))

    def sync_missing_movies(self):
        all_required_ids = self.get_all_required_ids()
        missing_ids = self.movie_repository.get_missing_movie_ids(all_required_ids)
        if missing_ids:
            for movie_id in missing_ids:
                movie_data = self.tmdb_client.get_movie_features(movie_id)
                if movie_data:
                    self.movie_repository.upsert_movie(movie_data)

    def get_all_features_as_dataframe(self):
        return self.movie_repository.get_all_features_as_dataframe()