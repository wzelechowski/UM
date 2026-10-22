import os
from dotenv import load_dotenv
from db.movie_db import MovieDB
from client.tmdb_client import TMDBClient
from util.data_loader import DataLoader
from repository.movie_repository import MovieRepository
from service.movie_service import MovieService
from service.processing_service import ProcessingService
from service.prediction_service import PredictionService


def main():
    load_dotenv()
    api_key = os.getenv('API_KEY')
    if not api_key:
        raise ValueError("Brak klucza API_KEY w pliku .env!")

    tmdb_client = TMDBClient(api_key)
    movie_db = MovieDB()
    data_loader = DataLoader()

    df_train, df_task = data_loader.load_data('./files/train.csv', './files/task.csv')

    movie_repository = MovieRepository(movie_db)
    movie_service = MovieService(df_train, df_task, movie_repository, tmdb_client)

    movie_service.sync_missing_movies()
    all_required_ids = movie_service.get_all_required_ids()

    df_raw_features = movie_repository.get_all_features_as_dataframe()
    df_raw_features = df_raw_features[df_raw_features['movie_id'].isin(all_required_ids)]

    tmdb_client.close()
    movie_db.close()

    processing_service = ProcessingService.create_default()
    movie_ids, matrices = processing_service.precompute_component_matrices(df_raw_features)

    prediction_service = PredictionService.create_default()
    df_submission = prediction_service.predict_for_all_users(df_train, df_task, movie_ids, matrices)

    df_submission.to_csv('./files/submission.csv', sep=';', header=False, index=False)
    print("Sukces! Plik submission.csv został wygenerowany.")


if __name__ == '__main__':
    main()