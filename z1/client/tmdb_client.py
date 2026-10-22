import requests
import time

class TMDBClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.themoviedb.org/3"
        self.session = requests.Session()

    def get_movie_features(self, movie_id):
        url = f"{self.base_url}/movie/{movie_id}"

        params = {
        'append_to_response' : 'genres,credits,keywords'
        }

        headers = {
            'Authorization': f'Bearer {self.api_key}'
        }

        try:
            response = self.session.get(url, params=params, headers=headers)

            if response.status_code == 429:
                time.sleep(5)
                return self.get_movie_features(movie_id)

            response.raise_for_status()
            data = response.json()

            crew = data.get('credits', {}).get('crew', [])
            directors = [member['name'] for member in crew if member.get('job') == 'Director']
            data['director'] = directors[0] if directors else None

            cast = data.get('credits', {}).get('cast', [])
            data['cast'] = [actor['name'] for actor in cast[:5]]

            keywords_list = data.get('keywords', {}).get('keywords', [])
            data['keywords'] = [kw['name'] for kw in keywords_list]

            return data

        except requests.exceptions.RequestException as e:
            print(f"Błąd podczas pobierania danych dla filmu {movie_id}: {e}")
            return None

    def close(self):
        self.session.close()