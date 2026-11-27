import sqlite3

class MovieDB:
    def __init__(self, db_path='./db/movies.sqlite'):
        self.connection = sqlite3.connect(db_path)

    def close(self):
        self.connection.close()