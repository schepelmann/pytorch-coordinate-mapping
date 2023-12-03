import pandas as pd
import torch
from torch.utils.data import Dataset

from utils import normalize


class CoordinateDataset(Dataset):
    def __init__(self, csv_file: str) -> None:
        super().__init__()

        # load data as pandas dataframe
        self.dataframe = pd.read_csv(
            csv_file,
            header=0,
            names=['x1', 'x2', 'y1', 'y2']
        )

        # scale each column to values between [0, 1]
        self.dataframe = self.normalize(self.dataframe)


    def normalize(self, df: pd.DataFrame) -> pd.DateOffset:
        x1, x2, y1, y2 = normalize(df['x1'], df['x2'], df['y1'], df['y2'])

        return pd.concat([x1, x2, y1, y2], axis=1)    


    def __len__(self):
        return len(self.dataframe)
    

    def __getitem__(self, index) -> tuple:
        row = self.dataframe.iloc[[index]]
        X = torch.tensor(row[['x1', 'x2']].to_numpy()[0], dtype=torch.float32)
        Y = torch.tensor(row[['y1', 'y2']].to_numpy()[0], dtype=torch.float32)

        return X, Y