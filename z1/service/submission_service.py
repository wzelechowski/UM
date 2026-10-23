class SubmissionService:
    def __init__(self, processing_service, feature_service, prediction_service):
        self.processing_service = processing_service
        self.feature_service = feature_service
        self.prediction_service = prediction_service

    def create_submission(self, df_train, df_task):
        df_raw_features = self.feature_service.get_raw_features()
        movie_ids, matrices = self.processing_service.precompute_component_matrices(df_raw_features)
        return self.prediction_service.predict_for_all_users(df_train, df_task, movie_ids, matrices)

    def submission_to_csv(self, df_train, df_task, output_path='./files/submission.csv'):
        df_submission = self.create_submission(df_train, df_task)
        df_submission.to_csv(output_path, sep=';', header=False, index=False)

