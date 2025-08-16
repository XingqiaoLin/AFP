import torch
import torch.nn as nn

class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, dropout=0.2):
        super(BANLayer, self).__init__()
        self.k = 3
        self.v_net = nn.Sequential(nn.Linear(v_dim, h_dim * self.k), nn.ReLU(), nn.Dropout(dropout))
        self.q_net = nn.Sequential(nn.Linear(q_dim, h_dim * self.k), nn.ReLU(), nn.Dropout(dropout))
        self.p_net = nn.AvgPool1d(self.k, stride=self.k)
        self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
        self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', v, att_map, q)
        fusion_logits = self.p_net(fusion_logits.unsqueeze(1)).squeeze(1) * self.k
        return fusion_logits

    def forward(self, v, q):
        v_ = self.v_net(v)
        q_ = self.q_net(q)
        att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', self.h_mat, v_, q_) + self.h_bias
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_mat.size(1)):
            logits += self.attention_pooling(v_, q_, att_maps[:, i, :, :])
        return self.bn(logits)

class AFPDeepPred(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, dropout):
        super(AFPDeepPred, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, v_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.mlp = nn.Sequential(
            nn.Linear(1280, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(dropout),
            nn.Linear(512, q_dim), nn.ReLU()
        )
        self.ban = BANLayer(v_dim, q_dim, h_dim, h_out, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(h_dim + q_dim, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, figure, embedding):
        figure = figure.unsqueeze(1)
        cnn_features = self.cnn(figure)
        mlp_features = self.mlp(embedding)
        ban_out = self.ban(cnn_features.unsqueeze(1), mlp_features.unsqueeze(1))
        combined_features = torch.cat((ban_out, mlp_features), dim=1)
        output = self.classifier(combined_features)
        return output

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss