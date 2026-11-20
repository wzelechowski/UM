from pathlib import Path
from db.movie_db import MovieDB
from repository.movie_repository import MovieRepository
from util.data_loader import DataLoader
from service.processing_service import ProcessingService
from service.prediction_service import PredictionService
from service.submission_service import SubmissionService


def main():
    base_dir = Path(__file__).resolve().parent
    db_path = base_dir / 'movies.sqlite'
    movie_db = MovieDB(str(db_path)) if db_path.exists() else None

    data_loader = DataLoader()
    df_train, df_task = data_loader.load_data(
        str(base_dir / 'files' / 'train.csv'),
        str(base_dir / 'files' / 'task.csv')
    )

    if movie_db is not None:
        MovieRepository(movie_db)

    processing_service = ProcessingService.create_default()
    prediction_service = PredictionService.create_default()

    submission_service = SubmissionService(processing_service, prediction_service)
    submission_service.submission_to_csv(
        df_train,
        df_task,
        str(base_dir / 'files' / 'submission.csv')
    )
    print("Sukces! Plik submission.csv został wygenerowany.")

    if movie_db is not None:
        movie_db.close()


if __name__ == '__main__':
    main()
