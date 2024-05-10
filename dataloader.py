from torch.utils.data import Dataset
import torch


class ScoresDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (list of tuples): A list where each tuple contains (score1, score2).
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        score1, score2 = self.data[idx]
        # Convert to a tensor before returning, if necessary
        return (
            torch.tensor(score1, dtype=torch.float16),
            torch.tensor(score2, dtype=torch.float16),
        )
