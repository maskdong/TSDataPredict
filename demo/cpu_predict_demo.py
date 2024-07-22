import pandas as pd
import requests
from io import StringIO
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

# 下载数据集
url = 'https://raw.githubusercontent.com/numenta/NAB/master/data/realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv'
response = requests.get(url)
data = pd.read_csv(StringIO(response.text))

# 查看数据集前几行
print(data.head())

# 数据预处理
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)

# 标准化
scaler = StandardScaler()
data['value'] = scaler.fit_transform(data[['value']])

# 创建序列数据
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length]
        sequences.append(seq)
        targets.append(label)
    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

seq_length = 24
sequences, targets = create_sequences(data['value'].values, seq_length)

# 创建数据加载器
batch_size = 32
dataset = TensorDataset(sequences, targets)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
