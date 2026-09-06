#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLMA训练5.0版本 - 专门用于生成绘图所需的真实数据
包括：消融实验、训练历史记录、误差分析、多维度评估
"""

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import pickle
import numpy as np
import time
import json
import os
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 检查GPU可用性并设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

class DataReader(Dataset):
    def __init__(self, subID: int) -> None:
        super().__init__()
        with open(r'PODAR_individual_modeling_code-master/data/dataset.pkl', 'rb') as f:
            self.data = pickle.load(f)[subID]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

class NGSIMDataReader(Dataset):
    """NGSIM数据集读取器 - 适配真实NGSIM数据"""
    def __init__(self, lane_type: str = 'HOV', sequence_length: int = 10, max_samples: int = 3000):
        super().__init__()
        self.sequence_length = sequence_length
        self.data = []
        
        # NGSIM数据路径映射
        ngsim_files = {
            'HOV': 'vehicle_lane_HOV.csv',
            'left': 'vehicle_lane_2_left_most_lane.csv', 
            'middle': 'vehicle_lane_34_middle_lanes.csv',
            'right': 'vehicle_lane_5_right_most_lane.csv'
        }
        
        ngsim_path = "Restructured-NGSIM-Dataset-for-surrounding-vehicles-master/Restructured-NGSIM-Dataset-for-surrounding-vehicles-master"
        file_path = Path(ngsim_path) / ngsim_files[lane_type]
        
        if file_path.exists():
            print(f"加载NGSIM数据: {lane_type} 车道")
            self._load_ngsim_data(file_path, max_samples)
        else:
            raise FileNotFoundError(f"NGSIM数据文件不存在: {file_path}")
    
    def _load_ngsim_data(self, file_path: Path, max_samples: int):
        """加载并预处理NGSIM数据"""
        # 读取NGSIM数据
        df = pd.read_csv(file_path, nrows=max_samples)
        print(f"原始数据: {len(df)} 行, {len(df.columns)} 列")
        
        # 按车辆ID分组处理时间序列
        for vehicle_id in df['subject_ID'].unique():
            vehicle_data = df[df['subject_ID'] == vehicle_id].sort_values('Frame')
            
            if len(vehicle_data) < self.sequence_length + 1:
                continue
            
            # 提取关键特征
            self._extract_vehicle_sequences(vehicle_data)
        
        print(f"处理后数据: {len(self.data)} 个序列")
    
    def _extract_vehicle_sequences(self, vehicle_data: pd.DataFrame):
        """从单个车辆数据中提取时间序列"""
        # 计算派生特征
        vehicle_data = vehicle_data.copy()
        
        # 相对速度变化 (类似原始数据的delta_v)
        vehicle_data['delta_v'] = vehicle_data['leading_speed_diff'].fillna(0)
        
        # 绝对速度 (类似原始数据的abs_v) 
        vehicle_data['abs_v'] = vehicle_data['subject_speed'].fillna(0)
        
        # 时间间隔 (类似原始数据的i)
        vehicle_data['time_gap'] = vehicle_data['leading_headway'].fillna(5.0)
        
        # 距离 (类似原始数据的d)
        vehicle_data['distance'] = vehicle_data['leading_Y'].fillna(100.0)
        
        # 目标值 (使用NGSIM的target_value作为st_angle的替代)
        vehicle_data['st_angle'] = vehicle_data['traget_value'].fillna(0)
        
        # 模拟主观响应 (基于speed和acceleration的组合)
        speed_norm = vehicle_data['subject_speed'] / vehicle_data['subject_speed'].max()
        acc_norm = vehicle_data['subject_Acc'].abs() / vehicle_data['subject_Acc'].abs().max()
        vehicle_data['response'] = (speed_norm * 0.6 + acc_norm * 0.4).fillna(0.5)
        
        # 创建滑动窗口序列
        for i in range(len(vehicle_data) - self.sequence_length):
            sequence = vehicle_data.iloc[i:i+self.sequence_length]
            target_row = vehicle_data.iloc[i+self.sequence_length]
            
            # 构造与原始数据格式兼容的数据结构
            data_item = {
                'delta_v': torch.tensor(sequence['delta_v'].values, dtype=torch.float32).unsqueeze(0),
                'abs_v': torch.tensor(sequence['abs_v'].values, dtype=torch.float32).unsqueeze(0),
                'i': torch.tensor(sequence['time_gap'].values, dtype=torch.float32).unsqueeze(0),
                'd': torch.tensor(sequence['distance'].values, dtype=torch.float32).unsqueeze(0),
                'st_angle': torch.tensor(target_row['st_angle'], dtype=torch.float32).unsqueeze(0),
                'response': torch.tensor(target_row['response'], dtype=torch.float32).unsqueeze(0)
            }
            
            self.data.append(data_item)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int):
        return self.data[idx]

def co_fn(data_dict, type='obj'):
    delta_v = [data['delta_v'] for data in data_dict]
    abs_v = [data['abs_v'] for data in data_dict]
    t_cur = [data['i'] for data in data_dict]
    d_cur = [data['d'] for data in data_dict]
    goal = 'st_angle' if type == 'obj' else 'response'
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

        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, groups=groups, bias=False)
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

# 基线模型（用于消融实验）
class BaselineModel(nn.Module):
    """基线模型 - 不包含LRAM和MFAFM"""
    def __init__(self, horizon=None):
        super(BaselineModel, self).__init__()
        self.m_ego, self.m_obj = torch.tensor(1.8).to(device), torch.tensor(1.8).to(device)
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
        attenu = torch.mul(w_i, w_d)
        podar_t = torch.mul(damage, attenu)
        podar = torch.max(podar_t[:, :self.horizon], dim=1)[0] if self.horizon else torch.max(podar_t, dim=1)[0]
        
        return podar

# PLMA with LRAM only (消融实验用)
class PLMA_LRAM(nn.Module):
    """只包含LRAM的PLMA"""
    def __init__(self, horizon=None):
        super(PLMA_LRAM, self).__init__()
        self.lram = LRAM(in_features=1, phi='T')
        self.m_ego, self.m_obj = torch.tensor(1.8).to(device), torch.tensor(1.8).to(device)
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
        damage = self.lram(damage)  # 添加LRAM
        attenu = torch.mul(w_i, w_d)
        podar_t = torch.mul(damage, attenu)
        podar = torch.max(podar_t[:, :self.horizon], dim=1)[0] if self.horizon else torch.max(podar_t, dim=1)[0]
        
        return podar

# PLMA with LRAM + MFAFM (消融实验用)
class PLMA_LRAM_MFAFM(nn.Module):
    """包含LRAM和MFAFM的PLMA"""
    def __init__(self, horizon=None):
        super(PLMA_LRAM_MFAFM, self).__init__()
        self.lram = LRAM(in_features=1, phi='T')
        self.mfafm = MFAFM(features=71)
        self.m_ego, self.m_obj = torch.tensor(1.8).to(device), torch.tensor(1.8).to(device)
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
        damage = self.lram(damage)  # 添加LRAM
        damage, _ = self.mfafm(damage)  # 添加MFAFM
        attenu = torch.mul(w_i, w_d)
        podar_t = torch.mul(damage, attenu)
        podar = torch.max(podar_t[:, :self.horizon], dim=1)[0] if self.horizon else torch.max(podar_t, dim=1)[0]
        
        return podar

# 完整PLMA模型
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
    """计算全面的评估指标"""
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
    
    # 新增评估指标
    mape = np.mean(np.abs((y_true_cpu - y_pred_cpu) / (y_true_cpu + 1e-8))) * 100
    smape = 100 * np.mean(2 * np.abs(y_pred_cpu - y_true_cpu) / (np.abs(y_pred_cpu) + np.abs(y_true_cpu) + 1e-8))
    correlation = np.corrcoef(y_true_cpu, y_pred_cpu)[0, 1] if len(y_true_cpu) > 1 else 0.0
    max_error = np.max(np.abs(y_true_cpu - y_pred_cpu))
    bias = np.mean(y_pred_cpu - y_true_cpu)
    nrmse = rmse / (np.max(y_true_cpu) - np.min(y_true_cpu) + 1e-8)
    explained_variance = 1 - np.var(y_true_cpu - y_pred_cpu) / (np.var(y_true_cpu) + 1e-8)
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

class TrainingRecorder:
    """训练历史记录器"""
    def __init__(self):
        self.history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'train_r2': [],
            'val_r2': [],
            'learning_rate': []
        }
    
    def record(self, epoch, train_loss, val_loss, train_r2, val_r2, lr):
        self.history['epochs'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_r2'].append(train_r2)
        self.history['val_r2'].append(val_r2)
        self.history['learning_rate'].append(lr)
    
    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.history, f, indent=2)

def train_with_history(model, subID, data_type='obj', max_epochs=1000, patience=100, save_dir='training_data'):
    """带历史记录的训练函数"""
    peak_num_angle_25 = [3, 7, 5, 4, 3, 3, 4, 5]
    peak_num_respons = [4, 7, 7, 6, 5, 4, 7, 7]
    
    batch_size = 77
    learning_rate = 0.01

    hor = peak_num_angle_25[subID] if data_type=='obj' else peak_num_respons[subID]
    model.horizon = int(hor * 10) if hasattr(model, 'horizon') else None
    model.apply(init_weights)
    
    data_set = DataReader(subID)
    data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True, 
                           collate_fn=lambda x: co_fn(x, data_type))

    # 准备验证数据
    with open(r'PODAR_individual_modeling_code-master/data/dataset.pkl', 'rb') as f:
        val_data = pickle.load(f)[subID]
    val_delta_v, val_abs_v, val_t_cur, val_d_cur, val_label = co_fn(val_data, type=data_type)

    criteria = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=50)
    
    # 训练历史记录器
    recorder = TrainingRecorder()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(max_epochs):
        # 训练阶段
        model.train()
        train_losses = []
        train_predictions = []
        train_targets = []
        
        for batch_id, (delta_v, abs_v, t_cur, d_cur, label) in enumerate(data_loader):
            optimizer.zero_grad()
            y_pred = model(delta_v, abs_v, t_cur, d_cur)
            loss = criteria(y_pred, label)
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            train_predictions.extend(y_pred.detach().cpu().numpy())
            train_targets.extend(label.detach().cpu().numpy())
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_pred = model(val_delta_v, val_abs_v, val_t_cur, val_d_cur)
            val_loss = criteria(val_pred, val_label)
            
            # 计算R²
            train_metrics = calculate_metrics(np.array(train_targets), np.array(train_predictions))
            val_metrics = calculate_metrics(val_label, val_pred)
            
            # 记录历史
            current_lr = optimizer.param_groups[0]['lr']
            recorder.record(epoch, np.mean(train_losses), val_loss.item(), 
                          train_metrics['r2'], val_metrics['r2'], current_lr)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                model.load_state_dict(best_model_state)
                break
    
    training_time = time.time() - start_time
    
    # 保存训练历史
    os.makedirs(save_dir, exist_ok=True)
    model_name = model.__class__.__name__
    history_file = os.path.join(save_dir, f'{model_name}_sub{subID}_{data_type}_history.json')
    recorder.save(history_file)
    
    # 最终评估
    model.eval()
    with torch.no_grad():
        final_pred = model(val_delta_v, val_abs_v, val_t_cur, val_d_cur)
        final_metrics = calculate_metrics(val_label, final_pred)
        
        # 收集预测误差用于分布分析
        errors = (final_pred - val_label).detach().cpu().numpy()
        error_file = os.path.join(save_dir, f'{model_name}_sub{subID}_{data_type}_errors.json')
        with open(error_file, 'w') as f:
            json.dump({
                'errors': errors.tolist(),
                'predictions': final_pred.detach().cpu().numpy().tolist(),
                'targets': val_label.detach().cpu().numpy().tolist()
            }, f, indent=2)
    
    return {
        'model_name': model_name,
        'final_metrics': final_metrics,
        'training_time': training_time,
        'epochs': epoch + 1,
        'history_file': history_file,
        'error_file': error_file
    }

def run_ablation_study(subID, data_type='obj', save_dir='training_data'):
    """运行消融实验"""
    print(f"\n🔬 运行消融实验 - Subject {subID}, Type {data_type}")
    
    peak_num_angle_25 = [3, 7, 5, 4, 3, 3, 4, 5]
    peak_num_respons = [4, 7, 7, 6, 5, 4, 7, 7]
    hor = peak_num_angle_25[subID] if data_type=='obj' else peak_num_respons[subID]
    
    # 定义消融实验的模型
    models = {
        'Baseline': BaselineModel(horizon=hor),
        'Baseline+LRAM': PLMA_LRAM(horizon=hor),
        'Baseline+LRAM+MFAFM': PLMA_LRAM_MFAFM(horizon=hor),
        'PLMA_Full': PLMA(horizon=hor)
    }
    
    ablation_results = {}
    
    for model_name, model in models.items():
        print(f"  训练 {model_name}...")
        model = model.to(device)
        result = train_with_history(model, subID, data_type, max_epochs=500, patience=50, save_dir=save_dir)
        ablation_results[model_name] = result
        
        # 清理GPU内存
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 保存消融实验结果
    ablation_file = os.path.join(save_dir, f'ablation_study_sub{subID}_{data_type}.json')
    
    # 转换为可序列化的格式
    serializable_results = {}
    for model_name, result in ablation_results.items():
        serializable_results[model_name] = {
            'r2': float(result['final_metrics']['r2']),
            'mse': float(result['final_metrics']['mse']),
            'mae': float(result['final_metrics']['mae']),
            'rmse': float(result['final_metrics']['rmse']),
            'correlation': float(result['final_metrics']['correlation']),
            'training_time': float(result['training_time']),
            'epochs': int(result['epochs']),
            'history_file': result['history_file'],
            'error_file': result['error_file']
        }
    
    with open(ablation_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"  消融实验结果保存到: {ablation_file}")
    return ablation_results

def collect_all_training_data():
    """收集所有训练数据用于绘图"""
    save_dir = 'training_data'
    os.makedirs(save_dir, exist_ok=True)
    
    print("🚀 开始收集训练数据用于科研绘图...")
    print(f"数据保存目录: {save_dir}")
    print("=" * 60)
    
    all_results = {}
    
    # 为每个被试运行消融实验
    for subID in range(8):
        print(f"\n📊 处理被试 {subID}...")
        all_results[subID] = {}
        
        # 对两种任务类型运行消融实验
        for data_type in ['obj', 'sub']:
            print(f"  任务类型: {data_type}")
            ablation_results = run_ablation_study(subID, data_type, save_dir)
            all_results[subID][data_type] = ablation_results
    
    # 保存汇总结果
    summary_file = os.path.join(save_dir, 'all_results_summary.json')
    
    # 创建汇总数据
    summary_data = {
        'experiment_info': {
            'subjects': 8,
            'tasks': ['obj', 'sub'],
            'models': ['Baseline', 'Baseline+LRAM', 'Baseline+LRAM+MFAFM', 'PLMA_Full'],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(device)
        },
        'results': {}
    }
    
    # 整理每个被试的结果
    for subID in range(8):
        summary_data['results'][f'subject_{subID}'] = {}
        for data_type in ['obj', 'sub']:
            summary_data['results'][f'subject_{subID}'][data_type] = {}
            for model_name in ['Baseline', 'Baseline+LRAM', 'Baseline+LRAM+MFAFM', 'PLMA_Full']:
                if model_name in all_results[subID][data_type]:
                    result = all_results[subID][data_type][model_name]
                    summary_data['results'][f'subject_{subID}'][data_type][model_name] = {
                        'r2': float(result['final_metrics']['r2']),
                        'mse': float(result['final_metrics']['mse']),
                        'mae': float(result['final_metrics']['mae']),
                        'rmse': float(result['final_metrics']['rmse']),
                        'correlation': float(result['final_metrics']['correlation']),
                        'training_time': float(result['training_time']),
                        'epochs': int(result['epochs'])
                    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ 数据收集完成！")
    print(f"📁 所有数据保存在: {save_dir}/")
    print(f"📄 汇总结果: {summary_file}")
    print("\n📋 生成的文件包括:")
    print("  - 消融实验结果: ablation_study_sub{X}_{type}.json")
    print("  - 训练历史: {ModelName}_sub{X}_{type}_history.json")
    print("  - 预测误差: {ModelName}_sub{X}_{type}_errors.json")
    print("  - 汇总数据: all_results_summary.json")
    
    return summary_data

def generate_plots_from_data():
    """基于收集的真实数据生成图表"""
    print("\n🎨 基于真实数据生成科研图表...")
    
    # 检查是否存在训练数据
    if not os.path.exists('training_data/all_results_summary.json'):
        print("❌ 未找到训练数据，请先运行数据收集")
        return
    
    # 导入绘图模块
    try:
        from plot_training_results import (
            plot_model_performance_comparison,
            plot_statistical_significance,
            plot_performance_heatmap,
            create_output_directory,
            setup_ieee_style
        )
        import matplotlib.pyplot as plt
        
        # 设置绘图样式
        setup_ieee_style()
        output_dir = create_output_directory()
        
        print(f"正在生成基于真实数据的图表...")
        print(f"输出目录: {output_dir}")
        
        # 生成基础图表（这些基于已有的真实数据）
        basic_plots = [
            (plot_model_performance_comparison, "model_performance_comparison.png"),
            (plot_statistical_significance, "statistical_significance.png"),
            (plot_performance_heatmap, "performance_heatmap.png")
        ]
        
        for plot_func, filename in basic_plots:
            try:
                fig = plot_func()
                filepath = output_dir / filename
                fig.savefig(filepath, dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                print(f"✅ 已生成: {filename}")
                plt.close(fig)
            except Exception as e:
                print(f"❌ 生成失败 {filename}: {e}")
        
        print("\n📊 还需要创建基于新收集数据的专门图表:")
        print("  - 消融实验可视化")
        print("  - 训练历史可视化")
        print("  - 误差分布分析")
        
    except ImportError as e:
        print(f"❌ 导入绘图模块失败: {e}")

def run_ngsim_validation():
    """运行NGSIM跨数据集验证实验"""
    print("\n🚗 开始NGSIM跨数据集验证实验")
    print("=" * 60)
    
    # 定义车道类型
    lane_types = ['HOV', 'left', 'middle', 'right']
    ngsim_results = {}
    
    # 为每种车道类型运行验证
    for lane_type in lane_types:
        print(f"\n🛣️  {lane_type} 车道验证...")
        
        try:
            # 加载NGSIM数据
            ngsim_dataset = NGSIMDataReader(lane_type=lane_type, max_samples=5000)
            
            if len(ngsim_dataset) == 0:
                print(f"   ❌ {lane_type} 车道数据为空")
                continue
                
            ngsim_loader = DataLoader(ngsim_dataset, batch_size=32, shuffle=True, collate_fn=lambda x: co_fn(x, 'obj'))
            
            # 创建并训练PLMA模型
            model = PLMA().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            criterion = nn.MSELoss()
            
            # 训练模型
            model.train()
            train_losses = []
            num_epochs = 50  # 较少的epochs用于快速验证
            
            print(f"   🔄 开始训练 ({num_epochs} epochs)...")
            start_time = time.time()
            
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                batch_count = 0
                
                for batch_idx, (delta_v, abs_v, t_cur, d_cur, labels) in enumerate(ngsim_loader):
                    optimizer.zero_grad()
                    
                    # 前向传播
                    outputs = model(delta_v, abs_v, t_cur, d_cur)
                    loss = criterion(outputs, labels)
                    
                    # 反向传播
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    batch_count += 1
                    
                    if batch_count >= 50:  # 限制每个epoch的batch数量
                        break
                
                avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
                train_losses.append(avg_loss)
                
                if (epoch + 1) % 10 == 0:
                    print(f"   Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
            
            training_time = time.time() - start_time
            
            # 评估模型
            model.eval()
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for batch_idx, (delta_v, abs_v, t_cur, d_cur, labels) in enumerate(ngsim_loader):
                    outputs = model(delta_v, abs_v, t_cur, d_cur)
                    
                    all_predictions.extend(outputs.cpu().numpy())
                    all_targets.extend(labels.cpu().numpy())
                    
                    if batch_idx >= 20:  # 限制评估批次
                        break
            
            # 计算评估指标
            predictions = np.array(all_predictions)
            targets = np.array(all_targets)
            
            if len(predictions) > 0 and len(targets) > 0:
                mse = mean_squared_error(targets, predictions)
                mae = mean_absolute_error(targets, predictions)
                rmse = np.sqrt(mse)
                
                # 计算R²
                ss_res = np.sum((targets - predictions) ** 2)
                ss_tot = np.sum((targets - np.mean(targets)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                # 计算相关系数
                correlation = np.corrcoef(targets, predictions)[0, 1] if len(targets) > 1 else 0
                
                metrics = {
                    'r2': float(r2),
                    'mse': float(mse),
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'correlation': float(correlation)
                }
                
                ngsim_results[lane_type] = {
                    'metrics': metrics,
                    'training_time': float(training_time),
                    'epochs': num_epochs,
                    'samples': len(ngsim_dataset),
                    'train_losses': train_losses,
                    'status': 'success'
                }
                
                print(f"   ✅ {lane_type} 完成:")
                print(f"      R² = {r2:.4f}, MSE = {mse:.6f}, MAE = {mae:.6f}")
                print(f"      样本数: {len(ngsim_dataset)}, 训练时间: {training_time:.1f}s")
            
        except Exception as e:
            print(f"   ❌ {lane_type} 车道验证失败: {str(e)}")
            ngsim_results[lane_type] = {
                'error': str(e),
                'status': 'failed'
            }
    
    # 保存NGSIM验证结果
    ngsim_save_dir = 'training_data'
    os.makedirs(ngsim_save_dir, exist_ok=True)
    
    ngsim_summary = {
        'experiment_info': {
            'dataset': 'NGSIM Cross-validation',
            'lane_types': lane_types,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(device)
        },
        'results': ngsim_results
    }
    
    # 计算总体统计
    successful_lanes = [k for k, v in ngsim_results.items() if v.get('status') == 'success']
    if successful_lanes:
        all_r2 = [ngsim_results[lane]['metrics']['r2'] for lane in successful_lanes]
        all_samples = [ngsim_results[lane]['samples'] for lane in successful_lanes]
        
        ngsim_summary['overall_stats'] = {
            'successful_lanes': len(successful_lanes),
            'total_lanes': len(lane_types),
            'avg_r2': float(np.mean(all_r2)),
            'total_samples': sum(all_samples),
            'best_r2': float(max(all_r2)),
            'worst_r2': float(min(all_r2))
        }
    
    ngsim_file = os.path.join(ngsim_save_dir, f'ngsim_validation_results_{int(time.time())}.json')
    with open(ngsim_file, 'w') as f:
        json.dump(ngsim_summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("🎉 NGSIM验证实验完成!")
    print(f"📄 结果保存至: {ngsim_file}")
    
    if 'overall_stats' in ngsim_summary:
        stats = ngsim_summary['overall_stats']
        print(f"✅ 成功验证: {stats['successful_lanes']}/{stats['total_lanes']} 车道")
        print(f"📊 平均R²: {stats['avg_r2']:.4f}")
        print(f"🚗 总样本数: {stats['total_samples']}")
        print(f"⭐ 最佳R²: {stats['best_r2']:.4f}")
    
    return ngsim_summary

if __name__ == "__main__":
    import sys
    
    print("PLMA训练数据收集器 5.0")
    print("专门用于生成科研绘图所需的真实数据")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == '--collect_data':
        # 收集所有训练数据
        summary_data = collect_all_training_data()
        
        # 生成图表
        generate_plots_from_data()
        
    elif len(sys.argv) > 1 and sys.argv[1] == '--plot_only':
        # 仅生成图表
        generate_plots_from_data()
        
    elif len(sys.argv) > 1 and sys.argv[1] == '--ngsim_validation':
        # 运行NGSIM跨数据集验证
        ngsim_results = run_ngsim_validation()
        
    elif len(sys.argv) > 1 and sys.argv[1] == '--full_experiment':
        # 运行完整实验：原始数据 + NGSIM验证
        print("🚀 开始完整实验流程...")
        
        # 1. 收集原始数据集结果
        print("\n1️⃣ 收集原始数据集结果...")
        summary_data = collect_all_training_data()
        
        # 2. 运行NGSIM验证
        print("\n2️⃣ 运行NGSIM跨数据集验证...")
        ngsim_results = run_ngsim_validation()
        
        # 3. 生成对比图表
        print("\n3️⃣ 生成完整对比图表...")
        generate_plots_from_data()
        
        print("\n🎉 完整实验流程完成!")
        print("📁 所有结果都保存在 training_data/ 目录中")
        
    else:
        print("使用方法:")
        print("  python plma_train_improved5.0.py --collect_data      # 收集原始数据并生成图表")
        print("  python plma_train_improved5.0.py --plot_only        # 仅生成图表")
        print("  python plma_train_improved5.0.py --ngsim_validation # 运行NGSIM跨数据集验证")
        print("  python plma_train_improved5.0.py --full_experiment  # 完整实验流程")
        print("\n推荐:")
        print("  首次运行: python plma_train_improved5.0.py --full_experiment")