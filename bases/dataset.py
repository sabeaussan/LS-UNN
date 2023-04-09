import numpy as np
from torch.utils.data import Dataset


class TrajectoriesTrainingDataset(Dataset):
    def __init__(self, trajectories_r1,trajectories_r2):
        self.trajectories_r1 = np.loadtxt(trajectories_r1,dtype=np.float32)
        self.trajectories_r2 = np.loadtxt(trajectories_r2,dtype=np.float32)

    def __len__(self):
        return len(self.trajectories_r1)

    def __getitem__(self, idx):
        sample = (self.trajectories_r1[idx],self.trajectories_r2[idx])
        return sample

class TrajectoriesTestingDataset(Dataset):
    def __init__(self, trajectories_r1,trajectories_r2):
        self.trajectories_r1 = np.loadtxt(trajectories_r1,dtype=np.float32)
        self.trajectories_r2 = np.loadtxt(trajectories_r2,dtype=np.float32)


    def __len__(self):
        return len(self.trajectories_r1)

    def __getitem__(self, idx):
        sample = (self.trajectories_r1[idx],self.trajectories_r2[idx])
        return sample



