class FeatureService:
    def __init__(self, movie_service):
        self.movie_service = movie_service

    def get_raw_features(self):
        self.movie_service.sync_missing_movies()
        all_required_ids = self.movie_service.get_all_required_ids()

        df_raw_features = self.movie_service.get_all_features_as_dataframe()
        return df_raw_features[df_raw_features['movie_id'].isin(all_required_ids)]