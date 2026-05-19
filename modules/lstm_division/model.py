import torch
import torch.nn as nn

class DivisionLSTM(nn.Module):
    """
    Clasificador LSTM para predicción de división celular (Adder Model).
    Fase 1 (MVP): Usa un stub que simula división cada 1200 pasos.
    Fase 2: Predicción temporal en base a historial de longitud y energía.
    """
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=2):
        super().__init__()
        # Arquitectura LSTM para Fase 2
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: [batch_size, sequence_length, features]
        out, _ = self.lstm(x)
        logits = self.classifier(out[:, -1, :])
        return torch.sigmoid(logits)
        
    def predict_division_stub(self, steps_lived):
        """
        Stub de Fase 1: devuelve True si los pasos de vida superan un límite (ej. 1200).
        """
        if steps_lived >= 1200:
            return True
        return False
