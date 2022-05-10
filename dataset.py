import torch
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path


class MovielenDataset(Dataset):
    def __init__(self, path: str, sep: str, subset: str):
        super().__init__()
        assert subset in ["train", "test"], "Unknown subset"

        rating_df = pd.read_csv(Path(path) / "ratings.csv", sep=sep, encoding="latin-1")
        user_df = pd.read_csv(Path(path) / "users.csv", sep=sep, encoding="latin-1")
        item_df = pd.read_csv(Path(path) / "movies.csv", sep=sep, encoding="latin-1")

        user_size = user_df.userId.max() + 1
        item_size = item_df.movieId.max() + 1

        # split by list users
        train_ratio = 0.9
        data_ratio = train_ratio if subset == "train" else 1 - train_ratio

        subset_rating = rating_df.sample(frac=data_ratio)

        rating = torch.tensor(subset_rating.rating.values)
        user_id = subset_rating.userId
        item_id = subset_rating.movieId
        indices = torch.tensor(list(zip(user_id, item_id))).t()

        self.user_item = (
            torch.sparse_coo_tensor(indices, rating, (user_size, item_size))
            .to_dense()
            .to(dtype=torch.float)
        )

        ones = torch.ones_like(rating)
        self.mask = (
            torch.sparse_coo_tensor(indices, ones, (user_size, item_size))
            .to_dense()
            .to(dtype=torch.float)
        )

    def __len__(self):
        return self.user_item.size(0)

    def __getitem__(self, idx):
        user_item = self.user_item[idx]
        mask = self.mask[idx]
        return user_item, mask
