"""
XRD Dataset for Diffusion Model Training

Dataset that returns (synth_xrd, real_xrd, temp) triplets.
"""

import torch
from torch.utils.data import Dataset


class XRDTransformDataset(Dataset):
    """
    Dataset that returns (synth_xrd, real_xrd, temp) triplets.
    Both synth_xrd and real_xrd are expected to be [N, L] arrays/tensors.
    They are reshaped to (N, 1, L). The temp is a scalar tensor for each pair.
    """
    def __init__(self, synth_xrd, real_xrd, temperature):
        assert len(synth_xrd) == len(real_xrd) == len(temperature), "Mismatched data lengths!"

        if torch.is_tensor(synth_xrd):
            self.synth_xrd = synth_xrd.clone().detach().float().unsqueeze(1)
        else:
            self.synth_xrd = torch.tensor(synth_xrd, dtype=torch.float32).unsqueeze(1)

        if torch.is_tensor(real_xrd):
            self.real_xrd = real_xrd.clone().detach().float().unsqueeze(1)
        else:
            self.real_xrd = torch.tensor(real_xrd, dtype=torch.float32).unsqueeze(1)

        if torch.is_tensor(temperature):
            self.temperature = temperature.clone().detach().float().unsqueeze(1)
        else:
            self.temperature = torch.tensor(temperature, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.synth_xrd)

    def __getitem__(self, idx):
        return self.synth_xrd[idx], self.real_xrd[idx], self.temperature[idx]