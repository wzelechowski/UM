class SubmissionService:
    def __init__(self, processing_service, prediction_service):
        self.processing_service = processing_service
        self.prediction_service = prediction_service

    def create_submission(self, df_train, df_task):
        user_to_idx, movie_to_idx, u_indices, m_indices, ratings = (
            self.processing_service.prepare_interaction_data(df_train, df_task)
        )
        return self.prediction_service.predict_for_all_users(
            df_train, df_task, user_to_idx, movie_to_idx, u_indices, m_indices, ratings
        )

    def submission_to_csv(self, df_train, df_task, output_path='./files/submission.csv'):
        df_submission = self.create_submission(df_train, df_task)
        df_submission.to_csv(output_path, sep=';', header=False, index=False)
