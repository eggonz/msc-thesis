import pickle

import torch
   

class ExpertDataSaver:
    """interface to save expert data using AutoProxy"""
    def __init__(self):
        self._data = []

    def append(self, idx, init_pose, f2f_delta, f2m_delta, gt_delta):
        """
        Args:
            idx: int
            init_pose: torch.Tensor [7]
            f2f_delta: torch.Tensor [7]
            f2m_delta: torch.Tensor [7]
            gt_delta: torch.Tensor [7]
        """
        self._data.append((
            int(idx),
            init_pose.detach().cpu().clone(),
            f2f_delta.detach().cpu().clone(),
            f2m_delta.detach().cpu().clone(),
            gt_delta.detach().cpu().clone()))
    
    def get_data(self):
        return self._data

    def __len__(self):
        return len(self._data)


class ExpertDataset(torch.utils.data.Dataset):
    """Saved data as a torch Dataset"""
    def __init__(self, data):
        self._data = data

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self._data[i] for i in idx]
        return self._data[idx]
    
    def __len__(self):
        return len(self._data)
    
    def save_pkl(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self._data, f)
        print(f"ExpertDataset saved to {path}")

    @classmethod
    def load_pkl(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print(f"ExpertDataset loaded from {path}")
        return cls(data)
