import json
import torch

from torch.utils.data import Dataset


class ScoresDataset(Dataset):
    def __init__(self, scores1, scores2):
        """
        Args:
            data (list of tuples): A list where each tuple contains (score1, score2).
        """
        self.data = [(score1, score2) for score1, score2 in zip(scores1, scores2)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        score1, score2 = self.data[idx]
        # Convert to a tensor of dtype float32 before feeding into neural network
        return (
            torch.tensor(score1, dtype=torch.float32),
            torch.tensor(score2, dtype=torch.float32),
        )


def collate_fn(batch):
    """
    Collate fn for ScoresDataset
    """
    # batch is a list of tuples: [(tensor1, tensor2), (tensor1, tensor2), ...]
    # We need to stack tensors of the same position together.

    # Unpack the tuples in the batch into two separate lists
    scores1, scores2 = zip(*batch)

    # Convert lists of tensors into a single tensor for each list
    scores1 = torch.stack(scores1)
    scores2 = torch.stack(scores2)

    # Combine the individual tensors into a single tensor of shape (batch_size, 2)
    batch_tensor = torch.cat((scores1.unsqueeze(1), scores2.unsqueeze(1)), dim=1)

    return batch_tensor


def load_into_scores_ds(
    model_name1: str, seed1: str, model_name2: str, seed2: str, metric, fold_num=None
):
    """ """

    try:
        file_path = (
            f"model_outputs/{model_name1}_{seed1}_{model_name2}_{seed2}/{metric}_scores_{fold_num}.json"
            if fold_num
            else f"model_outputs/{model_name1}_{seed1}_{model_name2}_{seed2}/{metric}_scores.json"
        )
        with open(file_path, "r") as file:
            data = json.load(file)

        scores_ds = ScoresDataset(data[f"{metric}_scores1"], data[f"{metric}_scores2"])

        return scores_ds

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return None
