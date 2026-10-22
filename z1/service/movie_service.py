
class MovieService:
    def __init__(self, df_train, df_task, movie_repository, tmdb_client):
        self.df_train = df_train
        self.df_task = df_task
        self.movie_repository = movie_repository
        self.tmdb_client = tmdb_client


    def get_all_required_ids(self):
        return set(self.df_train['movie_id']).union(set(self.df_task['movie_id']))

    def get_missing_ids(self):
        all_required_ids = self.get_all_required_ids()
        missing_ids =  self.movie_repository.get_missing_movie_ids(all_required_ids)
        if missing_ids:
            for i, movie_id in enumerate(missing_ids, 1):
                movie_data = self.tmdb_client.get_movie_features(movie_id)
                if movie_data:
                    self.movie_repository.upsert_movie(movie_data)

