import torch
from torch.utils.data import Dataset

class TaheriAdderDataset(Dataset):
    """
    Dataset para tiempos de división (Taheri-Araghi 2015).
    Fase 1: Implementación vacía.
    Fase 2: Cargará y procesará las secuencias temporales reales.
    """
    def __init__(self, data_path="data/taheri_2015/data.csv", seq_length=60):
        self.data_path = data_path
        self.seq_length = seq_length
        self.samples = []
        
        # TODO (Fase 2): Cargar y parsear archivo CSV real

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Devuelve secuencias de shape (seq_length, features)
        return torch.zeros((self.seq_length, 3)), torch.tensor([0.0])
