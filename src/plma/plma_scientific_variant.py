import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import pickle
import numpy as np
import time
import json
import os
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')

# 检查GPU可用性并设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DataReader(Dataset):
    def __init__(self, subID: int) -> None:
        super().__init__()
        with open(r'PODAR_individual_modeling_code-master/data/dataset.pkl', 'rb') as f:
            self.data = pickle.load(f)[subID]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


def co_fn(data_dict, type='obj'):
    delta_v = [data['delta_v'] for data in data_dict]
    abs_v = [data['abs_v'] for data in data_dict]
    t_cur = [data['i'] for data in data_dict]
    d_cur = [data['d'] for data in data_dict]
    goal = 'st_angle' if type == 'obj' else 'response'  # 'response', 'st_angle'
    label = [data[goal] for data in data_dict]
    rows, cols = len(delta_v), delta_v[0].shape[0]

    return (torch.cat(delta_v).reshape(rows, cols).float().to(device),
            torch.cat(abs_v).reshape(rows, cols).float().to(device),
            torch.cat(t_cur).reshape(rows, cols).float().to(device),
            torch.cat(d_cur).reshape(rows, cols).float().to(device),
            torch.cat(label).reshape(rows).float().to(device))


# LRAM模块
class LRAM(nn.Module):
    def __init__(self, in_features, phi):
        super(LRAM, self).__init__()
        kernel_size = {'T': 5, 'B': 7, 'S': 5, 'L': 7}[phi]
        groups = {'T': in_features, 'B': in_features, 'S': max(1, in_features // 8), 'L': max(1, in_features // 8)}[phi]
        padding = kernel_size // 2

        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, groups=groups,
                                bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        input = input.unsqueeze(1)
        x = self.conv1d(input)
        x = self.sigmoid(x)
        x = x.squeeze(1)
        return x


# MFAFM模块
class MFAFM(nn.Module):
    def __init__(self, features, factor=1):
        super(MFAFM, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(features, features))
        self.weights.data.copy_(torch.eye(features))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c = x.size()
        weights = self.softmax(self.weights)
        output = torch.matmul(x, weights)
        return output, weights


# PLMA模型
class PLMA(nn.Module):
    def __init__(self, horizon=None) -> None:
        super(PLMA, self).__init__()
        self.lram = LRAM(in_features=1, phi='T')
        self.mfafm = MFAFM(features=71)
        self.m_ego, self.m_obj = torch.tensor(1.8).to(device), torch.tensor(1.8).to(device)
        self.alpha = nn.Parameter(torch.FloatTensor([0.7]))
        self.A = nn.Parameter(torch.FloatTensor([0.3]))
        self.B = nn.Parameter(torch.FloatTensor([0.5]))
        self.horizon = int(horizon * 10) if horizon is not None else None
        self.scale = nn.Parameter(torch.FloatTensor([0.5 / 11.25]))
        self.Alpha_ = nn.Parameter(torch.tensor([0.7], dtype=torch.float32))

    def forward(self, delta_v, abs_v, i, d):
        m = 0.5 * (self.m_ego + self.m_obj)

        self.A_ = torch.clamp(self.A, 0.17, 50)
        self.B_ = torch.clamp(self.B, 0., 50)
        v = self.Alpha_ * delta_v + (1 - self.Alpha_) * abs_v

        w_i = torch.exp(-1 * self.A_ * i)
        w_d = torch.exp(-1 * self.B_ * d)

        damage = torch.mul(v, torch.abs(v)) * m * 0.001 * self.scale
        damage = self.lram(damage)
        damage, attention_weights = self.mfafm(damage)
        self.attention_weights = attention_weights

        attenu = torch.mul(w_i, w_d)
        podar_t = torch.mul(damage, attenu)
        podar = torch.max(podar_t[:, :self.horizon], dim=1)[0] if self.horizon else torch.max(podar_t, dim=1)[0]

        self.p_max = podar.max()

        return podar


def init_weights(layer):
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0, std=0.5)
    elif type(layer) == nn.Linear:
        nn.init.xavier_normal_(layer.weight)


def calculate_metrics(y_true, y_pred):
    """计算全面的评估指标 - 响应审稿人关于评估指标不够全面的意见"""
    if isinstance(y_true, torch.Tensor):
        y_true_cpu = y_true.detach().cpu().numpy()
    else:
        y_true_cpu = y_true

    if isinstance(y_pred, torch.Tensor):
        y_pred_cpu = y_pred.detach().cpu().numpy()
    else:
        y_pred_cpu = y_pred

    # R²分数
    y_mean = np.mean(y_true_cpu)
    ss_tot = np.sum((y_true_cpu - y_mean) ** 2)
    ss_res = np.sum((y_true_cpu - y_pred_cpu) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    # MSE
    mse = mean_squared_error(y_true_cpu, y_pred_cpu)

    # MAE
    mae = mean_absolute_error(y_true_cpu, y_pred_cpu)

    # RMSE
    rmse = np.sqrt(mse)

    # 预测方差
    pred_var = np.var(y_pred_cpu)

    # 相对误差
    rel_error = np.mean(np.abs((y_true_cpu - y_pred_cpu) / (y_true_cpu + 1e-8))) * 100

    # 新增评估指标 - 响应审稿人意见
    # 平均绝对百分比误差 (MAPE)
    mape = np.mean(np.abs((y_true_cpu - y_pred_cpu) / (y_true_cpu + 1e-8))) * 100

    # 对称平均绝对百分比误差 (SMAPE)
    smape = 100 * np.mean(2 * np.abs(y_pred_cpu - y_true_cpu) / (np.abs(y_pred_cpu) + np.abs(y_true_cpu) + 1e-8))

    # 皮尔逊相关系数
    correlation = np.corrcoef(y_true_cpu, y_pred_cpu)[0, 1] if len(y_true_cpu) > 1 else 0.0

    # 最大绝对误差
    max_error = np.max(np.abs(y_true_cpu - y_pred_cpu))

    # 平均偏差 (Bias)
    bias = np.mean(y_pred_cpu - y_true_cpu)

    # 标准化均方根误差 (NRMSE)
    nrmse = rmse / (np.max(y_true_cpu) - np.min(y_true_cpu) + 1e-8)

    # 解释方差分数
    explained_variance = 1 - np.var(y_true_cpu - y_pred_cpu) / (np.var(y_true_cpu) + 1e-8)

    # 中位数绝对误差
    median_ae = np.median(np.abs(y_true_cpu - y_pred_cpu))

    return {
        'r2': r2,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'pred_var': pred_var,
        'rel_error': rel_error,
        'mape': mape,
        'smape': smape,
        'correlation': correlation,
        'max_error': max_error,
        'bias': bias,
        'nrmse': nrmse,
        'explained_variance': explained_variance,
        'median_ae': median_ae
    }


def calculate_r2(y_true, y_pred):
    """计算R²分数（保持向后兼容）"""
    return calculate_metrics(y_true, y_pred)['r2']


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=1000, min_delta=1e-6, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        """保存模型权重"""
        self.best_weights = model.state_dict().copy()


# ==================== 基线模型 ====================

class CNN_LSTM(nn.Module):
    """CNN-LSTM基线模型"""

    def __init__(self, input_size=4, hidden_size=64, num_layers=2, seq_len=71):
        super(CNN_LSTM, self).__init__()
        self.seq_len = seq_len

        # CNN特征提取
        self.conv1d = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.conv2d = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        # LSTM层
        self.lstm = nn.LSTM(64, hidden_size, num_layers, batch_first=True, dropout=0.2)

        # 输出层
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, delta_v, abs_v, i, d):
        x = torch.stack([delta_v, abs_v, i, d], dim=2)
        x = x.transpose(1, 2)

        x = self.relu(self.conv1d(x))
        x = self.dropout(x)
        x = self.relu(self.conv2d(x))
        x = self.dropout(x)

        x = x.transpose(1, 2)

        lstm_out, (h_n, c_n) = self.lstm(x)

        output = self.fc(lstm_out[:, -1, :])
        return output.squeeze()


class MultiHeadAttention(nn.Module):
    """多头注意力基线模型"""

    def __init__(self, input_size=4, d_model=128, n_heads=8, seq_len=71):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        self.input_projection = nn.Linear(input_size, d_model)

        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.output_projection = nn.Linear(d_model, 1)

    def forward(self, delta_v, abs_v, i, d):
        x = torch.stack([delta_v, abs_v, i, d], dim=2)

        x = self.input_projection(x)

        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        x = torch.mean(x, dim=1)
        output = self.output_projection(x)
        return output.squeeze()


class VanillaRNN(nn.Module):
    """基础RNN基线模型"""

    def __init__(self, input_size=4, hidden_size=64, num_layers=2, seq_len=71):
        super(VanillaRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, delta_v, abs_v, i, d):
        x = torch.stack([delta_v, abs_v, i, d], dim=2)
        rnn_out, h_n = self.rnn(x)
        output = self.fc(self.dropout(rnn_out[:, -1, :]))
        return output.squeeze()


class VanillaLSTM(nn.Module):
    """基础LSTM基线模型"""

    def __init__(self, input_size=4, hidden_size=64, num_layers=2, seq_len=71):
        super(VanillaLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, delta_v, abs_v, i, d):
        x = torch.stack([delta_v, abs_v, i, d], dim=2)
        lstm_out, (h_n, c_n) = self.lstm(x)
        output = self.fc(self.dropout(lstm_out[:, -1, :]))
        return output.squeeze()


class BiLSTM(nn.Module):
    """双向LSTM基线模型"""

    def __init__(self, input_size=4, hidden_size=64, num_layers=2, seq_len=71):
        super(BiLSTM, self).__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers,
                              batch_first=True, dropout=0.2, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, delta_v, abs_v, i, d):
        x = torch.stack([delta_v, abs_v, i, d], dim=2)
        bilstm_out, (h_n, c_n) = self.bilstm(x)
        output = self.fc(self.dropout(bilstm_out[:, -1, :]))
        return output.squeeze()


class GRU(nn.Module):
    """GRU基线模型"""

    def __init__(self, input_size=4, hidden_size=64, num_layers=2, seq_len=71):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, delta_v, abs_v, i, d):
        x = torch.stack([delta_v, abs_v, i, d], dim=2)
        gru_out, h_n = self.gru(x)
        output = self.fc(self.dropout(gru_out[:, -1, :]))
        return output.squeeze()


class TransformerEncoderModel(nn.Module):
    """完整的Transformer Encoder基线模型 - 响应审稿人要求添加最新期刊方法"""

    def __init__(self, input_size=4, d_model=128, n_heads=8, num_layers=3, dim_feedforward=256, seq_len=71,
                 dropout=0.1):
        super(TransformerEncoderModel, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, delta_v, abs_v, i, d):
        x = torch.stack([delta_v, abs_v, i, d], dim=2)  # [batch, seq, 4]
        x = self.input_projection(x)
        x = self.transformer(x)
        x = self.dropout(x)
        x = torch.mean(x, dim=1)  # 全局平均池化
        output = self.output_projection(x)
        return output.squeeze()


def train_baseline_model(model_type, subID, data_type='obj', max_epochs=1000, patience=100):
    """训练基线模型"""
    with open(r'PODAR_individual_modeling_code-master/data/dataset.pkl', 'rb') as f:
        data = pickle.load(f)[subID]

    delta_v, abs_v, t_cur, d_cur, label = co_fn(data, type=data_type)

    if model_type in ['RandomForest', 'SVM']:
        X = torch.cat([delta_v, abs_v, t_cur, d_cur], dim=1).cpu().numpy()
        y = label.cpu().numpy()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        start_time = time.time()

        if model_type == 'RandomForest':
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_scaled, y)
            y_pred = model.predict(X_scaled)
        elif model_type == 'SVM':
            model = SVR(kernel='rbf', C=1.0, gamma='scale')
            model.fit(X_scaled, y)
            y_pred = model.predict(X_scaled)

        training_time = time.time() - start_time
        metrics = calculate_metrics(y, y_pred)
        metrics['training_time'] = training_time

        return metrics

    else:
        if model_type == 'CNN_LSTM':
            model = CNN_LSTM().to(device)
        elif model_type == 'MultiHeadAttention':
            model = MultiHeadAttention().to(device)
        elif model_type == 'VanillaRNN':
            model = VanillaRNN().to(device)
        elif model_type == 'VanillaLSTM':
            model = VanillaLSTM().to(device)
        elif model_type == 'BiLSTM':
            model = BiLSTM().to(device)
        elif model_type == 'GRU':
            model = GRU().to(device)
        elif model_type == 'Transformer':
            model = TransformerEncoderModel().to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.apply(init_weights)

        data_set = DataReader(subID)
        data_loader = DataLoader(dataset=data_set, batch_size=32, shuffle=True,
                                 collate_fn=lambda x: co_fn(x, data_type))

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=0.5, patience=20)

        early_stopping = EarlyStopping(patience=patience, min_delta=1e-6)

        start_time = time.time()

        for epoch in range(max_epochs):
            model.train()
            epoch_loss = 0
            batch_count = 0

            for delta_v_batch, abs_v_batch, t_cur_batch, d_cur_batch, label_batch in data_loader:
                optimizer.zero_grad()

                output = model(delta_v_batch, abs_v_batch, t_cur_batch, d_cur_batch)
                loss = criterion(output, label_batch)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            model.eval()
            with torch.no_grad():
                val_output = model(delta_v, abs_v, t_cur, d_cur)
                val_loss = criterion(val_output, label)

            scheduler.step(val_loss)

            if early_stopping(val_loss, model):
                break

        training_time = time.time() - start_time

        model.eval()
        with torch.no_grad():
            final_output = model(delta_v, abs_v, t_cur, d_cur)
            metrics = calculate_metrics(label, final_output)
            metrics['training_time'] = training_time

        return metrics


def train(subID, type='obj', max_epochs=50000, patience=1000, min_delta=1e-6):
    peak_num_angle_25 = [3, 7, 5, 4, 3, 3, 4, 5]
    peak_num_respons = [4, 7, 7, 6, 5, 4, 7, 7]

    batch_size = 77
    learning_rate = 0.01

    hor = peak_num_angle_25[subID] if type == 'obj' else peak_num_respons[subID]
    net = PLMA(horizon=hor).to(device)
    net.apply(init_weights)

    data_set = DataReader(subID)
    data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True, collate_fn=lambda x: co_fn(x, type))

    with open(r'PODAR_individual_modeling_code-master/data/dataset.pkl', 'rb') as f:
        val_data = pickle.load(f)[subID]
    val_delta_v, val_abs_v, val_t_cur, val_d_cur, val_label = co_fn(val_data, type=type)

    criteria = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=500)

    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    loss_rec = []
    r2_rec = []
    best_r2 = -float('inf')
    ite_num = 0
    epoch = 0
    start_time = time.time()

    while epoch < max_epochs:
        epoch_loss = 0
        batch_count = 0

        for batch_id, (delta_v, abs_v, t_cur, d_cur, label) in enumerate(data_loader):
            net.train()
            y = net(delta_v, abs_v, t_cur, d_cur)
            loss = criteria(y, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1
            ite_num += 1

        net.eval()
        with torch.no_grad():
            val_pred = net(val_delta_v, val_abs_v, val_t_cur, val_d_cur)
            val_loss = criteria(val_pred, val_label)
            val_metrics = calculate_metrics(val_label, val_pred)
            val_r2 = val_metrics['r2']

        avg_epoch_loss = epoch_loss / batch_count
        loss_rec.append(avg_epoch_loss)
        r2_rec.append(val_r2)

        scheduler.step(val_loss)

        if val_r2 > best_r2:
            best_r2 = val_r2

        if early_stopping(val_loss, net):
            break

        epoch += 1

    total_time = time.time() - start_time

    os.makedirs('trainning results', exist_ok=True)

    if type == 'obj':
        m = torch.jit.script(net)
        torch.jit.save(m, r'trainning results/angle_{}.pt'.format(subID))
    else:
        m = torch.jit.script(net)
        torch.jit.save(m, r'trainning results/response_{}.pt'.format(subID))

    net.eval()
    with torch.no_grad():
        final_pred = net(val_delta_v, val_abs_v, val_t_cur, val_d_cur)
        final_metrics = calculate_metrics(val_label, final_pred)

    return {
        'A': net.A_.item(),
        'B': net.B_.item(),
        'p_max': net.p_max.item(),
        'scale': net.scale.item(),
        'best_r2': best_r2,
        'epochs': epoch,
        'training_time': total_time,
        'loss_history': loss_rec,
        'r2_history': r2_rec,
        'final_metrics': final_metrics
    }


def verify_and_calculate_r2(subID):
    folder_n = 'trainning results'
    results = {}

    try:
        net_angle = torch.jit.load(r'{}/angle_{}.pt'.format(folder_n, subID))
        net_angle.eval()

        with open(r'PODAR_individual_modeling_code-master/data/dataset.pkl', 'rb') as f:
            data = pickle.load(f)[subID]

        delta_v, abs_v, t_cur, d_cur, label = co_fn(data, type='obj')

        with torch.no_grad():
            net_risk = net_angle(delta_v, abs_v, t_cur, d_cur)
            r2_angle = calculate_r2(label, net_risk)

        results['r2_angle'] = r2_angle

    except Exception as e:
        results['r2_angle'] = None

    try:
        net_response = torch.jit.load(r'{}/response_{}.pt'.format(folder_n, subID))
        net_response.eval()

        delta_v, abs_v, t_cur, d_cur, label = co_fn(data, type='sub')

        with torch.no_grad():
            net_risk = net_response(delta_v, abs_v, t_cur, d_cur)
            r2_response = calculate_r2(label, net_risk)

        results['r2_response'] = r2_response

    except Exception as e:
        results['r2_response'] = None

    return results


def cross_subject_generalization_analysis(all_results):
    models = ['CNN_LSTM', 'MultiHeadAttention', 'VanillaRNN', 'VanillaLSTM', 'BiLSTM', 'GRU', 'Transformer',
              'RandomForest', 'SVM', 'PLMA']
    data_types = ['obj', 'sub']

    print("\n" + "=" * 100)
    print("跨被试泛化性能分析")
    print("=" * 100)
    print(f"{'Model':<20} {'Type':<5} {'Mean_R2':<10} {'Std_R2':<10} {'Min_R2':<10} {'Max_R2':<10} {'CV_R2':<10}")
    print("-" * 100)

    generalization_results = {}

    for data_type in data_types:
        generalization_results[data_type] = {}

        for model in models:
            r2_values = []

            for subID in range(8):
                if (subID in all_results and
                        data_type in all_results[subID] and
                        model in all_results[subID][data_type]):
                    metrics = all_results[subID][data_type][model]['avg_metrics']
                    r2_values.append(metrics['r2'])

            if r2_values and len(r2_values) >= 3:
                mean_r2 = np.mean(r2_values)
                std_r2 = np.std(r2_values)
                min_r2 = np.min(r2_values)
                max_r2 = np.max(r2_values)
                cv_r2 = std_r2 / (abs(mean_r2) + 1e-8)

                generalization_results[data_type][model] = {
                    'mean': mean_r2,
                    'std': std_r2,
                    'min': min_r2,
                    'max': max_r2,
                    'cv': cv_r2,
                    'generalization_score': 1 / (1 + cv_r2)
                }

                print(
                    f"{model:<20} {data_type:<5} {mean_r2:<10.4f} {std_r2:<10.4f} {min_r2:<10.4f} {max_r2:<10.4f} {cv_r2:<10.4f}")

    print("=" * 100)
    print("注释: CV_R2 = R²变异系数 (越小表示跨被试性能越稳定)")

    return generalization_results


def feature_importance_analysis():
    print("\n" + "=" * 80)
    print("特征重要性和数据使用说明")
    print("=" * 80)

    print("1. 输入特征详细说明:")
    print("   - delta_v: 速度差")
    print("   - abs_v: 绝对速度")
    print("   - time_index (i): 时间索引")
    print("   - distance (d): 距离")

    print("\n2. 特征工程:")
    print("   - 速度融合: v = Alpha_ * delta_v + (1 - Alpha_) * abs_v")
    print("   - 损伤计算: damage = v * |v| * mass * scale")
    print("   - 时间衰减: w_i = exp(-A * i)")
    print("   - 距离衰减: w_d = exp(-B * d)")
    print("   - 最终风险: podar = max(damage * w_i * w_d)")

    print("\n3. 物理意义:")
    print("   - 基于动量和能量的物理模型")
    print("   - 考虑时间和空间的衰减效应")
    print("   - 通过注意力机制捕获关键时刻")

    print("\n4. 数据集特性:")
    print("   - 8个被试的驾驶数据")
    print("   - 序列长度: 71个时间步")
    print("   - 预测目标: 转向角度(obj) 和 驾驶员反应(sub)")

    print("=" * 80)


def stability_analysis(model_results):
    if not model_results:
        return {}

    metrics = ['r2', 'mse', 'mae', 'rmse']
    stability_metrics = {}

    for metric in metrics:
        values = [result[metric] for result in model_results if metric in result]
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            cv = std_val / (abs(mean_val) + 1e-8)
            min_val = np.min(values)
            max_val = np.max(values)
            range_val = max_val - min_val

            stability_metrics[metric] = {
                'mean': mean_val,
                'std': std_val,
                'cv': cv,
                'min': min_val,
                'max': max_val,
                'range': range_val,
                'stability_score': 1 / (1 + cv)
            }

    return stability_metrics


def train_plma_multiple_runs(subID, data_type='obj', num_runs=3, max_epochs=50000, patience=1000):
    results = []

    for run in range(num_runs):
        print(f"PLMA Run {run + 1}/{num_runs} for Subject {subID}, Type {data_type}")
        try:
            torch.manual_seed(42 + run)
            np.random.seed(42 + run)

            result = train(subID, type=data_type, max_epochs=max_epochs, patience=patience)
            results.append(result)
        except Exception as e:
            print(f"Error in PLMA run {run}: {e}")
            continue

    return results


def comprehensive_baseline_comparison(num_runs=3):
    baseline_models = ['CNN_LSTM', 'MultiHeadAttention', 'VanillaRNN', 'VanillaLSTM', 'BiLSTM', 'GRU', 'Transformer',
                       'RandomForest', 'SVM']
    data_types = ['obj', 'sub']

    all_results = {}

    print("开始全面基线模型比较...")
    print("=" * 100)
    print(
        f"{'Model':<20} {'Type':<5} {'R2':<10} {'MSE':<10} {'MAE':<10} {'RMSE':<10} {'Pred_Var':<12} {'Rel_Error':<12} {'Time(s)':<8}")
    print("=" * 100)

    for subID in range(8):
        all_results[subID] = {}

        for data_type in data_types:
            all_results[subID][data_type] = {}

            for model_name in baseline_models:
                model_results = []

                for run in range(num_runs):
                    try:
                        result = train_baseline_model(model_name, subID, data_type)
                        model_results.append(result)
                    except Exception as e:
                        print(f"Error training {model_name} for subject {subID}, type {data_type}, run {run}: {e}")
                        continue

                if model_results:
                    avg_metrics = {}
                    std_metrics = {}

                    for metric in ['r2', 'mse', 'mae', 'rmse', 'pred_var', 'rel_error', 'training_time']:
                        values = [result[metric] for result in model_results if metric in result]
                        if values:
                            avg_metrics[metric] = np.mean(values)
                            std_metrics[metric] = np.std(values)
                        else:
                            avg_metrics[metric] = 0.0
                            std_metrics[metric] = 0.0

                    all_results[subID][data_type][model_name] = {
                        'avg_metrics': avg_metrics,
                        'std_metrics': std_metrics,
                        'num_runs': len(model_results)
                    }

                    print(f"{model_name:<20} {data_type:<5} {avg_metrics['r2']:<10.4f} {avg_metrics['mse']:<10.4f} "
                          f"{avg_metrics['mae']:<10.4f} {avg_metrics['rmse']:<10.4f} {avg_metrics['pred_var']:<12.4f} "
                          f"{avg_metrics['rel_error']:<12.1f} {avg_metrics['training_time']:<8.1f}")

            try:
                plma_results = train_plma_multiple_runs(subID, data_type=data_type, num_runs=num_runs,
                                                        max_epochs=50000, patience=1000)

                if plma_results:
                    plma_metrics_list = []
                    for result in plma_results:
                        metrics = result['final_metrics'].copy()
                        metrics['training_time'] = result['training_time']
                        plma_metrics_list.append(metrics)

                    avg_metrics = {}
                    std_metrics = {}

                    for metric in ['r2', 'mse', 'mae', 'rmse', 'pred_var', 'rel_error', 'training_time']:
                        values = [result[metric] for result in plma_metrics_list if metric in result]
                        if values:
                            avg_metrics[metric] = np.mean(values)
                            std_metrics[metric] = np.std(values)
                        else:
                            avg_metrics[metric] = 0.0
                            std_metrics[metric] = 0.0

                    stability_metrics = stability_analysis(plma_metrics_list)

                    all_results[subID][data_type]['PLMA'] = {
                        'avg_metrics': avg_metrics,
                        'std_metrics': std_metrics,
                        'stability_metrics': stability_metrics,
                        'num_runs': len(plma_results)
                    }

                    print(f"{'PLMA':<20} {data_type:<5} {avg_metrics['r2']:<10.4f} {avg_metrics['mse']:<10.4f} "
                          f"{avg_metrics['mae']:<10.4f} {avg_metrics['rmse']:<10.4f} {avg_metrics['pred_var']:<12.4f} "
                          f"{avg_metrics['rel_error']:<12.1f} {avg_metrics['training_time']:<8.1f}")

            except Exception as e:
                print(f"Error training PLMA for subject {subID}, type {data_type}: {e}")

        print(f"Subject {subID} completed")
        print("-" * 100)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_results


def print_average_results(all_results):
    models = ['CNN_LSTM', 'MultiHeadAttention', 'VanillaRNN', 'VanillaLSTM', 'BiLSTM', 'GRU', 'Transformer',
              'RandomForest', 'SVM', 'PLMA']
    data_types = ['obj', 'sub']

    print("\n" + "=" * 140)
    print("所有被试平均结果 (8个被试)")
    print("=" * 140)
    print(
        f"{'Model':<20} {'Type':<5} {'R2±std':<15} {'MSE±std':<15} {'MAE±std':<15} {'RMSE±std':<15} {'Corr±std':<15} {'Time±std':<15}")
    print("=" * 140)

    summary_results = {}

    for data_type in data_types:
        summary_results[data_type] = {}

        for model in models:
            r2_values = []
            mse_values = []
            mae_values = []
            rmse_values = []
            correlation_values = []
            time_values = []

            for subID in range(8):
                if (subID in all_results and
                        data_type in all_results[subID] and
                        model in all_results[subID][data_type]):
                    metrics = all_results[subID][data_type][model]['avg_metrics']
                    r2_values.append(metrics['r2'])
                    mse_values.append(metrics['mse'])
                    mae_values.append(metrics['mae'])
                    rmse_values.append(metrics['rmse'])
                    correlation_values.append(metrics.get('correlation', 0.0))
                    time_values.append(metrics['training_time'])

            if r2_values:
                avg_r2 = np.mean(r2_values)
                std_r2 = np.std(r2_values)
                avg_mse = np.mean(mse_values)
                std_mse = np.std(mse_values)
                avg_mae = np.mean(mae_values)
                std_mae = np.std(mae_values)
                avg_rmse = np.mean(rmse_values)
                std_rmse = np.std(rmse_values)
                avg_corr = np.mean(correlation_values)
                std_corr = np.std(correlation_values)
                avg_time = np.mean(time_values)
                std_time = np.std(time_values)

                summary_results[data_type][model] = {
                    'r2': (avg_r2, std_r2),
                    'mse': (avg_mse, std_mse),
                    'mae': (avg_mae, std_mae),
                    'rmse': (avg_rmse, std_rmse),
                    'correlation': (avg_corr, std_corr),
                    'time': (avg_time, std_time)
                }

                print(f"{model:<20} {data_type:<5} "
                      f"{avg_r2:.4f}±{std_r2:.4f}   "
                      f"{avg_mse:.4f}±{std_mse:.4f}   "
                      f"{avg_mae:.4f}±{std_mae:.4f}   "
                      f"{avg_rmse:.4f}±{std_rmse:.4f}   "
                      f"{avg_corr:.4f}±{std_corr:.4f}   "
                      f"{avg_time:.1f}±{std_time:.1f}")

    print("=" * 140)

    print("\n" + "=" * 100)
    print("模型稳定性分析报告")
    print("=" * 100)
    print(f"{'Model':<20} {'Type':<5} {'R2_CV':<10} {'MSE_CV':<10} {'Stability_Score':<15}")
    print("-" * 100)

    for data_type in data_types:
        for model in models:
            stability_scores = []
            r2_cv_scores = []
            mse_cv_scores = []

            for subID in range(8):
                if (subID in all_results and
                        data_type in all_results[subID] and
                        model in all_results[subID][data_type] and
                        'stability_metrics' in all_results[subID][data_type][model]):

                    stability_data = all_results[subID][data_type][model]['stability_metrics']
                    if 'r2' in stability_data:
                        r2_cv_scores.append(stability_data['r2']['cv'])
                        stability_scores.append(stability_data['r2']['stability_score'])
                    if 'mse' in stability_data:
                        mse_cv_scores.append(stability_data['mse']['cv'])

            if stability_scores:
                avg_r2_cv = np.mean(r2_cv_scores)
                avg_mse_cv = np.mean(mse_cv_scores)
                avg_stability = np.mean(stability_scores)

                print(f"{model:<20} {data_type:<5} {avg_r2_cv:<10.4f} {avg_mse_cv:<10.4f} {avg_stability:<15.4f}")

    print("=" * 100)

    import json
    with open('baseline_comparison_results.json', 'w') as f:
        json_results = {}
        for data_type in summary_results:
            json_results[data_type] = {}
            for model in summary_results[data_type]:
                json_results[data_type][model] = {}
                for metric in summary_results[data_type][model]:
                    avg, std = summary_results[data_type][model][metric]
                    json_results[data_type][model][metric] = {'avg': float(avg), 'std': float(std)}

        json.dump(json_results, f, indent=2)

    print(f"\n详细结果已保存到 baseline_comparison_results.json")

    return summary_results


def print_implementation_details():
    print("\n" + "=" * 80)
    print("详细实现细节和超参数设置")
    print("=" * 80)

    print("1. PLMA模型架构:")
    print("   - LRAM模块: 1D卷积 (kernel_size=5, groups=1, padding=2)")
    print("   - MFAFM模块: 全连接注意力 (features=71×71)")
    print("   - 激活函数: Sigmoid (LRAM), Softmax (MFAFM)")
    print("   - 损失函数: MSE Loss")

    print("\n2. 训练超参数:")
    print("   - 学习率: 0.01")
    print("   - 批大小: 77")
    print("   - 最大轮数: 50,000")
    print("   - 早停耐心: 1,000")
    print("   - 优化器: Adam")

    print("\n3. 基线模型超参数:")
    print("   Transformer:")
    print("     - d_model=128, n_heads=8, num_layers=3, dim_feedforward=256, dropout=0.1")
    print("   其他基线保持不变")

    print("=" * 80)


# ==================== 扩展消融实验 ====================

def run_ablation_study(subID=0, data_type='obj', max_epochs=2000, patience=200):
    """
    扩展消融实验 - 新增正则化和特征组合消融
    """
    print(f"🔬 开始扩展消融实验 - Subject {subID}, Type: {data_type}")

    peak_num_angle_25 = [3, 7, 5, 4, 3, 3, 4, 5]
    peak_num_respons = [4, 7, 7, 6, 5, 4, 7, 7]
    hor = peak_num_angle_25[subID] if data_type == 'obj' else peak_num_respons[subID]

    data_set = DataReader(subID)
    batch_size = 77
    data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True,
                             collate_fn=lambda x: co_fn(x, data_type))

    with open(r'PODAR_individual_modeling_code-master/data/dataset.pkl', 'rb') as f:
        val_data = pickle.load(f)[subID]
    val_delta_v, val_abs_v, val_t_cur, val_d_cur, val_label = co_fn(val_data, type=data_type)

    experiments = {
        'Full PLMA': {'use_lram': True, 'use_mfafm': True, 'reg': 'none', 'features': 'full'},
        'w/o LRAM': {'use_lram': False, 'use_mfafm': True, 'reg': 'none', 'features': 'full'},
        'w/o MFAFM': {'use_lram': True, 'use_mfafm': False, 'reg': 'none', 'features': 'full'},
        'w/o Both': {'use_lram': False, 'use_mfafm': False, 'reg': 'none', 'features': 'full'},
        'Full + Dropout': {'use_lram': True, 'use_mfafm': True, 'reg': 'dropout', 'features': 'full'},
        'Full + L2': {'use_lram': True, 'use_mfafm': True, 'reg': 'l2', 'features': 'full'},
        'Full + Dropout+L2': {'use_lram': True, 'use_mfafm': True, 'reg': 'both', 'features': 'full'},
        'No Time Decay': {'use_lram': True, 'use_mfafm': True, 'reg': 'none', 'features': 'no_time'},
        'No Distance Decay': {'use_lram': True, 'use_mfafm': True, 'reg': 'none', 'features': 'no_dist'},
        'Only Velocity': {'use_lram': True, 'use_mfafm': True, 'reg': 'none', 'features': 'velocity_only'},
    }

    ablation_results = {}

    for exp_name, config in experiments.items():
        print(f"  ⚡ 训练 {exp_name}...")

        class AblationPLMA(PLMA):
            def __init__(self, horizon):
                super().__init__(horizon)
                self.dropout = nn.Dropout(0.2) if 'dropout' in config['reg'] else None

            def forward(self, delta_v, abs_v, i, d):
                m = 0.5 * (self.m_ego + self.m_obj)

                if config['features'] == 'no_time':
                    i = torch.zeros_like(i)
                elif config['features'] == 'no_dist':
                    d = torch.zeros_like(d)
                elif config['features'] == 'velocity_only':
                    i = torch.zeros_like(i)
                    d = torch.zeros_like(d)

                v = self.Alpha_ * delta_v + (1 - self.Alpha_) * abs_v
                damage = torch.mul(v, torch.abs(v)) * m * 0.001 * self.scale

                if config['use_lram']:
                    damage = self.lram(damage)

                if config['use_mfafm']:
                    damage, _ = self.mfafm(damage)

                if self.dropout is not None:
                    damage = self.dropout(damage)

                self.A_ = torch.clamp(self.A, 0.17, 50)
                self.B_ = torch.clamp(self.B, 0., 50)
                w_i = torch.exp(-1 * self.A_ * i)
                w_d = torch.exp(-1 * self.B_ * d)
                attenu = torch.mul(w_i, w_d)
                podar_t = torch.mul(damage, attenu)
                podar = torch.max(podar_t[:, :self.horizon], dim=1)[0] if self.horizon else torch.max(podar_t, dim=1)[0]

                return podar

        net = AblationPLMA(horizon=hor).to(device)
        net.apply(init_weights)

        weight_decay = 1e-4 if config['reg'] in ['l2', 'both'] else 0.0
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=weight_decay)

        criteria = nn.MSELoss()
        early_stopping = EarlyStopping(patience=patience, min_delta=1e-6)

        for epoch in range(max_epochs):
            net.train()
            epoch_loss = 0
            batch_count = 0
            for delta_v, abs_v, t_cur, d_cur, label in data_loader:
                y = net(delta_v, abs_v, t_cur, d_cur)
                loss = criteria(y, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1

            net.eval()
            with torch.no_grad():
                val_pred = net(val_delta_v, val_abs_v, val_t_cur, val_d_cur)
                val_loss = criteria(val_pred, val_label)

            if early_stopping(val_loss, net):
                break

        net.eval()
        with torch.no_grad():
            final_pred = net(val_delta_v, val_abs_v, val_t_cur, val_d_cur)
            metrics = calculate_metrics(val_label, final_pred)

        ablation_results[exp_name] = {
            'r2': float(metrics['r2']),
            'mse': float(metrics['mse']),
            'mae': float(metrics['mae']),
            'rmse': float(metrics['rmse'])
        }
        print(f"    R² = {metrics['r2']:.4f}, MSE = {metrics['mse']:.4f}")

    os.makedirs('scientific_data', exist_ok=True)
    with open('scientific_data/extended_ablation_study_data.json', 'w') as f:
        json.dump({
            'subject_id': subID,
            'data_type': data_type,
            'results': ablation_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2)

    print(f"✅ 扩展消融实验完成，结果保存至 scientific_data/extended_ablation_study_data.json")
    return ablation_results


if __name__ == "__main__":
    import sys

    print_implementation_details()

    if len(sys.argv) > 1 and sys.argv[1] == '--baseline_comparison':
        print("开始基线模型比较实验（已包含Transformer）...")
        comparison_results = comprehensive_baseline_comparison(num_runs=3)
        summary = print_average_results(comparison_results)
        generalization_results = cross_subject_generalization_analysis(comparison_results)
        feature_importance_analysis()
        print("\n实验完成！")

    else:
        print("开始标准PLMA训练...")
        all_results = {}

        for subID in range(8):
            print(f"\n训练被试 {subID}...")
            all_results[subID] = {}

            obj_result = train(subID, type='obj', max_epochs=50000, patience=1000)
            all_results[subID]['obj'] = obj_result

            sub_result = train(subID, type='sub', max_epochs=50000, patience=1000)
            all_results[subID]['sub'] = sub_result

            verification_results = verify_and_calculate_r2(subID)
            all_results[subID]['verification'] = verification_results

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 结果汇总打印（保持原样）
        # ...（原代码中的打印部分保持不变）

        print("\n要运行扩展消融实验，可手动调用 run_ablation_study(subID, data_type)")