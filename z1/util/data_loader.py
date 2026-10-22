import pandas as pd

class DataLoader:
    def __init__(self, columns=None):
        if columns is None:
            columns = ['id', 'user_id', 'movie_id', 'evaluation']
        self.columns = columns

    def load_data(self, train_path, task_path):
        df_train = pd.read_csv(train_path, sep=';', names=self.columns, header=None)
        df_task = pd.read_csv(task_path, sep=';', names=self.columns, header=None)

        return df_train, df_task