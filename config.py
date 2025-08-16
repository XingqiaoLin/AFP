# config.py

import torch


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


TRAIN_CSV_PATH = 'data/train.csv'
VAL_CSV_PATH = 'data/val.csv'
MODEL_SAVE_DIR = 'models'
BEST_MODEL_PATH = f'{MODEL_SAVE_DIR}/best_model.pth'


NUM_EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001


V_DIM = 256      
Q_DIM = 128      
H_DIM = 256      
H_OUT = 1        
DROPOUT = 0.2    
