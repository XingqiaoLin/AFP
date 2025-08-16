import torch
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score
import config
from model import AFPDeepPred
from data_loader import get_data_loader

def main():
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Using CPU for evaluation.")

    val_loader = get_data_loader(config.VAL_CSV_PATH, config.BATCH_SIZE, shuffle=False)
    
    model = AFPDeepPred(
        v_dim=config.V_DIM, 
        q_dim=config.Q_DIM, 
        h_dim=config.H_DIM, 
        h_out=config.H_OUT,
        dropout=config.DROPOUT
    ).to(config.DEVICE)
    
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE))
    model.eval()

    print("Evaluating model on the validation set...")
    
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for figures, embeddings, labels in val_loader:
            figures = figures.to(config.DEVICE)
            embeddings = embeddings.to(config.DEVICE)
            
            outputs = model(figures, embeddings)
            
            probs = outputs.cpu().numpy().flatten()
            predicted = (probs > 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(predicted)
            all_labels.extend(labels.cpu().numpy().flatten())

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    mcc = matthews_corrcoef(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    print(f'Accuracy: {accuracy*100:.2f}%')
    print(f'Sensitivity (Recall): {sensitivity*100:.2f}%')
    print(f'Specificity: {specificity*100:.2f}%')
    print(f'MCC: {mcc:.4f}')
    print(f'AUC: {auc:.4f}')

if __name__ == '__main__':
    main()