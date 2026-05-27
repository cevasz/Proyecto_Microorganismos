import torch
import torch.nn as nn

class DivisionLSTM(nn.Module):
    """
    Clasificador LSTM para predicción de división celular (Adder Model).
    Ref: Taheri-Araghi et al., Current Biology 25(3), 2015.
    """
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        # Arquitectura central LSTM con dropout
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        
        # Head de clasificación
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: Tensor de tamaño (batch_size, sequence_length, input_dim)
        Retorna P(división en los próximos 30 pasos) ∈ [0,1]
        """
        out, _ = self.lstm(x)
        # Se toma la salida del último estado de la secuencia
        p_div = self.classifier(out[:, -1, :])
        return p_div

    def predict_division_stub(self, steps_lived):
        """Mantiene compatibilidad con módulos de Fase 1 durante la transición"""
        return steps_lived >= 1200
