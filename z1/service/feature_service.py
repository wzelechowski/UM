import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler


class FeatureService:
    def __init__(self, num_cols=None):
        self.num_cols = num_cols or ['release_year', 'vote_average', 'runtime']
        self.mlb_genres = MultiLabelBinarizer()
        self.mlb_cast = MultiLabelBinarizer()
        self.scaler = MinMaxScaler()

    def prepare_feature_matrix(self, df_features):
        df = df_features.copy()

        if 'movie_id' in df.columns:
            df.set_index('movie_id', inplace=True)

        genres_matrix = self.mlb_genres.fit_transform(df['genres'])
        genres_df = pd.DataFrame(
            genres_matrix,
            columns=[f"genre_{c}" for c in self.mlb_genres.classes_],
            index=df.index
        )

        cast_matrix = self.mlb_cast.fit_transform(df['cast_list'])
        cast_df = pd.DataFrame(
            cast_matrix,
            columns=[f"cast_{c}" for c in self.mlb_cast.classes_],
            index=df.index
        )

        df[self.num_cols] = df[self.num_cols].fillna(0)

        scaled_nums = self.scaler.fit_transform(df[self.num_cols])
        nums_df = pd.DataFrame(scaled_nums, columns=self.num_cols, index=df.index)

        return pd.concat([nums_df, genres_df, cast_df], axis=1)