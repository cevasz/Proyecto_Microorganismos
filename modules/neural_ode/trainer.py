import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .pi_node import PINode
from .dataset import GEOTimeSeriesDataset

def pearson_correlation(x, y):
    """Calcula el coeficiente de correlación de Pearson."""
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
    return cost

def train_node_geodata():
    print("Iniciando entrenamiento de Neural ODE (PINODE) con datos GEO GSE4513 y GT Sintético...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hiperparámetros
    seq_length = 60
    batch_size = 32
    epochs = 200
    patience = 20
    
    # Datasets y Dataloaders
    train_dataset = GEOTimeSeriesDataset(seq_length=seq_length, stride=10, is_train=True)
    val_dataset = GEOTimeSeriesDataset(seq_length=seq_length, stride=10, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Instanciamos el modelo PINode
    model = PINode(state_dim=4, hidden_dim=64).to(device)
    
    # Optimizador y Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Directorio y tracking
    os.makedirs("models", exist_ok=True)
    best_model_path = "models/neural_ode_best.pt"
    
    best_val_loss = float("inf")
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        # --- ENTRENAMIENTO ---
        model.train()
        train_loss_acum = 0.0
        
        for x_env, y_true in train_loader:
            x_env, y_true = x_env.to(device), y_true.to(device)
            optimizer.zero_grad()
            
            # El estado inicial z0 es el inicio de la secuencia de targets
            z0 = y_true[:, 0, :]
            
            # Vector de tiempos para la integración (asumimos resolución dt=0.1 -> 10s)
            t_span = torch.linspace(0., (seq_length-1)*0.1, seq_length).to(device)
            
            # Predecir trayectoria: el modelo devuelve (time_steps, batch_size, state_dim)
            z_pred = model(z0, t_span)
            
            # Transponer para emparejar con y_true -> (batch_size, time_steps, state_dim)
            z_pred = z_pred.transpose(0, 1)
            
            # Calcular la pérdida que incluye penalizaciones físicas
            total_loss, mse_loss, physics_loss = model.compute_loss(z_pred, y_true)
            
            total_loss.backward()
            optimizer.step()
            train_loss_acum += total_loss.item()
            
        scheduler.step()
        
        # --- VALIDACIÓN ---
        model.eval()
        val_loss_acum = 0.0
        val_mse = 0.0
        val_mae = 0.0
        pearson_sum = 0.0
        
        with torch.no_grad():
            for x_env, y_true in val_loader:
                x_env, y_true = x_env.to(device), y_true.to(device)
                
                z0 = y_true[:, 0, :]
                t_span = torch.linspace(0., (seq_length-1)*0.1, seq_length).to(device)
                z_pred = model(z0, t_span).transpose(0, 1)
                
                t_loss, m_loss, p_loss = model.compute_loss(z_pred, y_true)
                val_loss_acum += t_loss.item()
                
                # Métricas
                val_mse += nn.MSELoss()(z_pred, y_true).item()
                val_mae += nn.L1Loss()(z_pred, y_true).item()
                
                # Correlación de Pearson para CheY-P (índice 0)
                corr = pearson_correlation(z_pred[..., 0], y_true[..., 0])
                pearson_sum += corr.item()
                
        # Calcular promedios
        train_loss = train_loss_acum / len(train_loader)
        val_loss = val_loss_acum / len(val_loader)
        v_mse = val_mse / len(val_loader)
        v_mae = val_mae / len(val_loader)
        v_pearson = pearson_sum / len(val_loader)
        
        print(f"Epoch {epoch+1:03d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"MSE: {v_mse:.4f} | MAE: {v_mae:.4f} | Pearson(CheY-P): {v_pearson:.4f}")
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            # print(f"  -> Model saved to {best_model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n[Early Stopping] Detenido en epoch {epoch+1}. Mejor Val Loss: {best_val_loss:.4f}")
                print(f"El mejor modelo ha sido guardado en {best_model_path}")
                break

if __name__ == "__main__":
    train_node_geodata()
