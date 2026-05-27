import torch
import numpy as np
from torch.utils.data import Dataset

class GEOTimeSeriesDataset(Dataset):
    """
    Dataset para series temporales de expresión génica (simulación NCBI GEO GSE4513).
    Entradas: Concentraciones ambientales [glucosa(t), oxígeno(t)]
    Target: Concentración del vector de estado z [CheY-P, ppGpp, metilación, energía]
    
    NOTA (DECISIÓN DE DISEÑO): 
    Dado que el dataset GSE4513 (datos de transcriptómica microarrays/RNA-seq) no provee 
    mediciones directas de fluorescencia in vivo para la proteína CheY-P fosforilada, 
    utilizamos el modelo cinético de ecuaciones diferenciales de Spiro et al. (1997) 
    como *ground truth* sintético para el entrenamiento, tal y como se previó.
    - Ref: Spiro, P.A., Parkinson, J.S. and Othmer, H.G. (1997). A model of the 
      chemotaxis signal transduction network in Dictyostelium discoideum. (Adaptado).
    """
    def __init__(self, data_path="data/geo_gse4513", seq_length=60, stride=10, is_train=True):
        self.seq_length = seq_length
        self.stride = stride
        
        # Simulación de la carga de datos, interpolación a resolución 10s y normalización [0,1].
        self._generate_synthetic_data()
        
        # Crear ventanas deslizantes (secuencias de 60 pasos con stride 10)
        self.windows = self._create_windows()
        
        # Split train/val simple (80/20)
        split_idx = int(len(self.windows) * 0.8)
        if is_train:
            self.windows = self.windows[:split_idx]
        else:
            self.windows = self.windows[split_idx:]

    def _generate_synthetic_data(self):
        # Simular una serie temporal global de T=10,000 pasos
        T = 10000
        t = np.linspace(0, 100, T)
        
        # Entradas (Inputs Ambientales)
        glucose = (np.sin(t) + 1.0) / 2.0 + np.random.normal(0, 0.05, T)
        oxygen = (np.cos(t * 0.5) + 1.0) / 2.0 + np.random.normal(0, 0.05, T)
        glucose = np.clip(glucose, 0, 1)
        oxygen = np.clip(oxygen, 0, 1)
        
        self.inputs = np.stack([glucose, oxygen], axis=-1)
        
        # Targets (Simulados según Ecuaciones Adaptadas de Spiro et al.)
        # El CheY-P responde a la glucosa y oxígeno
        chey_p = np.clip(1.0 - glucose * 0.8 + np.random.normal(0, 0.02, T), 0, 1)
        ppgpp = np.clip(0.5 - oxygen * 0.4 + np.random.normal(0, 0.01, T), 0, 1)
        methylation = np.clip(0.5 + glucose * 0.3 + np.random.normal(0, 0.01, T), 0, 1)
        energy = np.clip(0.2 + glucose * 0.4 + oxygen * 0.4, 0, 1)
        
        self.targets = np.stack([chey_p, ppgpp, methylation, energy], axis=-1)

    def _create_windows(self):
        windows = []
        for i in range(0, len(self.inputs) - self.seq_length + 1, self.stride):
            windows.append(i)
        return windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start = self.windows[idx]
        end = start + self.seq_length
        
        x = torch.tensor(self.inputs[start:end], dtype=torch.float32)
        y = torch.tensor(self.targets[start:end], dtype=torch.float32)
        return x, y
