import json
import pandas as pd

class MovieRepository:
    def __init__(self, movie_db):
        self.movie_db = movie_db
        self.create_table()

    def create_table(self):
        cursor = self.movie_db.connection.cursor()
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS movies
                       (
                           movie_id
                           INTEGER
                           PRIMARY
                           KEY,
                           title
                           TEXT,
                           release_year
                           INTEGER,
                           runtime
                           INTEGER,
                           vote_average
                           REAL,
                           vote_count
                           INTEGER,
                           popularity
                           REAL,
                           budget
                           INTEGER,
                           revenue
                           INTEGER,
                           original_language
                           TEXT,
                           collection_name
                           TEXT,
                           genres
                           TEXT,
                           production_companies
                           TEXT,
                           production_countries
                           TEXT,
                           cast_list
                           TEXT,
                           director
                           TEXT,
                           keywords
                           TEXT
                       )
                       """)
        self.movie_db.connection.commit()

    def get_missing_movie_ids(self, required_ids):
        cursor = self.movie_db.connection.cursor()
        cursor.execute("SELECT movie_id FROM movies")
        existing_ids = {row[0] for row in cursor.fetchall()}
        return required_ids - existing_ids

    def upsert_movie(self, movie_data):
        cursor = self.movie_db.connection.cursor()

        release_date = movie_data.get('release_date', '')
        release_year = int(release_date[:4]) if len(release_date) >= 4 and release_date[:4].isdigit() else None

        collection = movie_data.get('belongs_to_collection')
        collection_name = collection.get('name') if isinstance(collection, dict) else None

        genres = [g.get('name') for g in movie_data.get('genres', [])]
        companies = [c.get('name') for c in movie_data.get('production_companies', [])]
        countries = [c.get('iso_3166_1') for c in movie_data.get('production_countries', [])]

        genres_json = json.dumps(genres)
        companies_json = json.dumps(companies)
        countries_json = json.dumps(countries)

        cast_json = json.dumps(movie_data.get('cast', []))
        keywords_json = json.dumps(movie_data.get('keywords', []))

        cursor.execute("""
                       INSERT INTO movies (movie_id, title, release_year, runtime, vote_average, vote_count,
                                           popularity, budget, revenue, original_language, collection_name,
                                           genres, production_companies, production_countries,
                                           cast_list, director, keywords)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(movie_id) DO
                       UPDATE SET
                           title=excluded.title,
                           release_year=excluded.release_year,
                           runtime=excluded.runtime,
                           vote_average=excluded.vote_average,
                           vote_count=excluded.vote_count,
                           popularity=excluded.popularity,
                           budget=excluded.budget,
                           revenue=excluded.revenue,
                           original_language=excluded.original_language,
                           collection_name=excluded.collection_name,
                           genres=excluded.genres,
                           production_companies=excluded.production_companies,
                           production_countries=excluded.production_countries,
                           cast_list=excluded.cast_list,
                           director=excluded.director,
                           keywords=excluded.keywords
                       """, (
                           movie_data.get('id'),
                           movie_data.get('title'),
                           release_year,
                           movie_data.get('runtime'),
                           movie_data.get('vote_average'),
                           movie_data.get('vote_count'),
                           movie_data.get('popularity'),
                           movie_data.get('budget'),
                           movie_data.get('revenue'),
                           movie_data.get('original_language'),
                           collection_name,
                           genres_json,
                           companies_json,
                           countries_json,
                           cast_json,
                           movie_data.get('director'),
                           keywords_json
                       ))
        self.movie_db.connection.commit()

    def get_all_features_as_dataframe(self):
        query = "SELECT * FROM movies"
        df = pd.read_sql_query(query, self.movie_db.connection)

        list_columns = ['genres', 'production_companies', 'production_countries', 'cast_list', 'keywords']
        for col in list_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: json.loads(x) if pd.notnull(x) and x != "" else [])

        return df
