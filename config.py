# config.py

import torch

# --- 设备配置 ---
# 自动检测是否可用GPU，否则使用CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 文件路径 ---
# 请根据您的实际文件存放位置修改这些路径
TRAIN_CSV_PATH = 'data/train.csv'
VAL_CSV_PATH = 'data/val.csv'
MODEL_SAVE_DIR = 'models'
BEST_MODEL_PATH = f'{MODEL_SAVE_DIR}/best_model.pth'

# --- 训练超参数 ---
NUM_EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# --- 模型结构参数 ---
V_DIM = 256      
Q_DIM = 128      
H_DIM = 256      
H_OUT = 1        
DROPOUT = 0.2    