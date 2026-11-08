class SubmissionService:
    def __init__(self, processing_service, feature_service, prediction_service):
        self.processing_service = processing_service
        self.feature_service = feature_service
        self.prediction_service = prediction_service

    def create_submissions(self, df_train, df_task, tree_image_path=None):
        df_raw_features = self.feature_service.get_raw_features()
        df_features, condition_factories = self.processing_service.prepare_features_and_factories(df_raw_features)
        return self.prediction_service.predict_for_all_users(
            df_train,
            df_task,
            df_features,
            condition_factories,
            tree_image_path=tree_image_path,
        )

    def submissions_to_csv(
        self,
        df_train,
        df_task,
        tree_csv_path='./files/submission_tree.csv',
        forest_csv_path='./files/submission_forest.csv',
        tree_image_path='./files/tree.png',
    ):
        df_tree, df_forest = self.create_submissions(df_train, df_task, tree_image_path=tree_image_path)
        df_tree.to_csv(tree_csv_path, sep=';', header=False, index=False)
        df_forest.to_csv(forest_csv_path, sep=';', header=False, index=False)
