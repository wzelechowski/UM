import os
from dotenv import load_dotenv
from db.movie_db import MovieDB
from client.tmdb_client import TMDBClient
from util.data_loader import DataLoader
from similarity_engine import precompute_component_matrices
from general_predictor import predict_for_all_users
from service.movie_service import MovieService
from repository.movie_repository import MovieRepository

def main():
    load_dotenv()
    api_key = os.getenv('API_KEY')
    if not api_key:
        raise ValueError("Brak klucza API_KEY w pliku .env!")

    tmdb_client = TMDBClient(api_key)
    movie_db = MovieDB()

    data_loader = DataLoader()

    df_train, df_task = data_loader.load_data('files/train.csv', 'files/task.csv')

    movie_repository = MovieRepository(movie_db)
    movie_service = MovieService(df_train, df_task, movie_repository, tmdb_client)

    df_raw_features = movie_db.get_all_features_as_dataframe()

    df_raw_features = df_raw_features[df_raw_features['movie_id'].isin(all_required_ids)]

    movie_ids, matrices = precompute_component_matrices(df_raw_features)

    df_submission = predict_for_all_users(df_train, df_task, movie_ids, matrices)
    df_submission.to_csv('files/submission.csv', sep=';', header=False, index=False)
    print("Sukces! Plik submission.csv został wygenerowany.")

    tmdb_client.close()
    movie_db.close()

if __name__ == '__main__':
    main()