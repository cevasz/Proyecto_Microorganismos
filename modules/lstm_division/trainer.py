import os
import torch
import torch.nn as nn
from scipy.stats import chisquare
import numpy as np
from .model import DivisionLSTM
from .dataset import get_adder_dataloaders

def train_lstm_adder():
    print("Iniciando entrenamiento de LSTM de División Celular (Fase 2)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hiperparámetros
    epochs = 100
    patience = 10
    
    # 80/10/10 split
    train_loader, val_loader, test_loader = get_adder_dataloaders(batch_size=64)
    model = DivisionLSTM(input_dim=3, hidden_dim=128, num_layers=2, dropout=0.2).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    
    os.makedirs("models", exist_ok=True)
    best_model_path = "models/lstm_division_best.pt"
    
    best_val_loss = float("inf")
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            p_div = model(x)
            loss = criterion(p_div, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                p_div = model(x)
                val_loss += criterion(p_div, y).item()
                
        t_loss = train_loss / len(train_loader)
        v_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1:03d}/{epochs} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")
        
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[Early Stopping] Detenido en epoch {epoch+1}")
                break
                
    # VALIDACIÓN FINAL EN TEST SET: Métricas reales e inferencia
    print("\n--- VALIDACIÓN FINAL EN TEST SET ---")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    test_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            p_div = model(x)
            test_loss += criterion(p_div, y).item()
            
            all_preds.extend(p_div.cpu().numpy().flatten())
            all_targets.extend(y.cpu().numpy().flatten())
            
    test_loss /= len(test_loader)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Evaluar métricas de clasificación reales (umbral estándar > 0.85)
    preds_binary = (all_preds > 0.85).astype(np.float32)
    accuracy = np.mean(preds_binary == all_targets)
    
    tp = np.sum((preds_binary == 1) & (all_targets == 1))
    fp = np.sum((preds_binary == 1) & (all_targets == 0))
    fn = np.sum((preds_binary == 0) & (all_targets == 1))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy (umbral > 0.85): {accuracy:.2%}")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")
    
    # Test Chi-cuadrado real sobre frecuencias de eventos de división
    # Compara la proporción de divisiones predichas vs ground truth en el test set
    obs_div = int(np.sum(preds_binary))
    exp_div = int(np.sum(all_targets))
    
    f_obs = np.array([obs_div, len(preds_binary) - obs_div])
    f_exp = np.array([exp_div, len(all_targets) - exp_div])
    
    # Suavizado Laplace para evitar ceros
    f_obs = f_obs + 1
    f_exp = f_exp + 1
    f_exp = f_exp * (f_obs.sum() / f_exp.sum())
    
    chi2, p_val = chisquare(f_obs=f_obs, f_exp=f_exp)
    print(f"Resultado Chi2 (Distribución de eventos de división) -> p-value: {p_val:.4f}")
    if p_val > 0.05:
        print("[EXITO] La tasa de eventos de división predicha no difiere significativamente del test set (p > 0.05)")
    else:
        print("[AVISO] Divergencia en la distribución de división. Revisar calibración de umbrales LSTM.")

if __name__ == "__main__":
    train_lstm_adder()
