import os
from pathlib import Path
from dotenv import load_dotenv
from db.movie_db import MovieDB
from client.tmdb_client import TMDBClient
from util.data_loader import DataLoader
from repository.movie_repository import MovieRepository
from service.movie_service import MovieService
from service.processing_service import ProcessingService
from service.prediction_service import PredictionService
from service.feature_service import FeatureService
from service.submission_service import SubmissionService


def main():
    base_dir = Path(__file__).resolve().parent
    load_dotenv(base_dir / '.env')

    api_key = os.getenv('API_KEY')
    if not api_key:
        raise ValueError("Brak klucza API_KEY w pliku .env!")

    tmdb_client = TMDBClient(api_key)
    movie_db = MovieDB(str(base_dir / 'movies.sqlite'))
    data_loader = DataLoader()

    df_train, df_task = data_loader.load_data(
        str(base_dir / 'files' / 'train.csv'),
        str(base_dir / 'files' / 'task.csv')
    )

    movie_repository = MovieRepository(movie_db)
    movie_service = MovieService(df_train, df_task, movie_repository, tmdb_client)
    feature_service = FeatureService(movie_service)
    prediction_service = PredictionService.create_default()
    processing_service = ProcessingService.create_default()

    submission_service = SubmissionService(processing_service, feature_service, prediction_service)
    submission_service.submission_to_csv(
        df_train,
        df_task,
        str(base_dir / 'files' / 'submission.csv')
    )
    print("Sukces! Plik submission.csv został wygenerowany.")

    tmdb_client.close()
    movie_db.close()


if __name__ == '__main__':
    main()