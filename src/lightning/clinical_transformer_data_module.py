import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from typing import Callable, List, Dict, Any, Optional

class ClinicalDataset(Dataset):
    def __init__(self, tokenized_data: List[Dict[str, Any]]):
        """
        Args:
            tokenized_data (List[dict]): List of pre-tokenized examples.
        """
        self.data = tokenized_data  # Store tokenized data directly

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]  # Directly return the tokenized sample

