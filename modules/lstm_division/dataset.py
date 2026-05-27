import torch
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader

class TaheriAdderDataset(Dataset):
    """
    Dataset sintético calibrado con Taheri-Araghi et al. (2015) - Adder model.
    La célula añade longitud o volumen constante independientemente del tamaño inicial.
    Tiempos de división ~ Normal(1200s, 150s).
    """
    def __init__(self, num_samples=10000, seq_length=60):
        self.seq_length = seq_length
        self.num_samples = num_samples
        self._generate_synthetic_adder_data()

    def _generate_synthetic_adder_data(self):
        self.inputs = []
        self.targets = []
        
        for _ in range(self.num_samples):
            # Tiempos de división teóricos: ~Normal(1200, 150)
            # Para 1200s a 10s por step = 120 steps promedio
            steps_to_div = int(np.random.normal(120, 15))
            steps_to_div = max(60, steps_to_div)  # Garantizar que haya al menos 1 secuencia
            
            # Trayectoria fisiológica aproximada a través del ciclo celular
            time = np.linspace(0, 1, steps_to_div)
            
            # Features: [longitud, energía, replicación_ADN]
            # Longitud sube ~1 unidad (adder model), Energía y replicación oscilan o suben
            length = np.linspace(1.0, 2.0, steps_to_div) + np.random.normal(0, 0.05, steps_to_div)
            energy = np.linspace(0.3, 0.9, steps_to_div) + np.random.normal(0, 0.05, steps_to_div)
            dna_rate = (np.exp(time * 3) / np.exp(3)) + np.random.normal(0, 0.02, steps_to_div)
            
            # Normalización a [0, 1]
            length = np.clip(length, 0, 10) / 10.0 # escalar aprox
            energy = np.clip(energy, 0, 1)
            dna_rate = np.clip(dna_rate, 0, 1)
            
            # Extraer sub-secuencia aleatoria del ciclo
            start = np.random.randint(0, steps_to_div - self.seq_length + 1)
            end = start + self.seq_length
            
            seq_inputs = np.stack([length[start:end], energy[start:end], dna_rate[start:end]], axis=-1)
            self.inputs.append(seq_inputs)
            
            # Target = 1 si la división ocurrirá en los próximos 30 pasos post-secuencia
            # es decir, (total_steps - current_step) <= 30
            div_in_30 = 1.0 if (steps_to_div - end) <= 30 else 0.0
            self.targets.append([div_in_30])
            
        self.inputs = np.array(self.inputs)
        self.targets = np.array(self.targets)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y

def get_adder_dataloaders(batch_size=32, num_samples=10000, seq_length=60):
    dataset = TaheriAdderDataset(num_samples=num_samples, seq_length=seq_length)
    
    # Split 80/10/10
    train_sz = int(0.8 * len(dataset))
    val_sz = int(0.1 * len(dataset))
    test_sz = len(dataset) - train_sz - val_sz
    
    train_ds, val_ds, test_ds = random_split(dataset, [train_sz, val_sz, test_sz])
    
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    )
