import os
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import config
from model import AFPDeepPred, FocalLoss
from data_loader import get_data_loader

def main():
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    
    train_loader = get_data_loader(config.TRAIN_CSV_PATH, config.BATCH_SIZE, shuffle=True)
    val_loader = get_data_loader(config.VAL_CSV_PATH, config.BATCH_SIZE, shuffle=False)

    model = AFPDeepPred(
        v_dim=config.V_DIM, 
        q_dim=config.Q_DIM, 
        h_dim=config.H_DIM, 
        h_out=config.H_OUT,
        dropout=config.DROPOUT
    ).to(config.DEVICE)
    
    criterion = FocalLoss(alpha=1, gamma=2)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_val_auc = 0.0

    print("Starting model training...")
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_loss = 0
        for figures, embeddings, labels in train_loader:
            figures = figures.to(config.DEVICE)
            embeddings = embeddings.to(config.DEVICE)
            labels = labels.to(config.DEVICE).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(figures, embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_probs = []
        val_labels = []
        with torch.no_grad():
            for figures, embeddings, labels in val_loader:
                figures = figures.to(config.DEVICE)
                embeddings = embeddings.to(config.DEVICE)
                
                outputs = model(figures, embeddings)
                val_probs.extend(outputs.cpu().numpy().flatten())
                val_labels.extend(labels.cpu().numpy().flatten())
        
        val_auc = roc_auc_score(val_labels, val_probs)
        avg_train_loss = total_loss / len(train_loader)
        
        print(f'Epoch [{epoch+1}/{config.NUM_EPOCHS}], Loss: {avg_train_loss:.4f}, Val AUC: {val_auc:.4f}')

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"New best model saved! Val AUC: {best_val_auc:.4f}")

    print("Training finished.")
    print(f'Best Validation AUC: {best_val_auc:.4f}')

if __name__ == '__main__':
    main()