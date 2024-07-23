# %%
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# 数据预处理
url = "https://raw.githubusercontent.com/numenta/NAB/master/data/realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv"
df = pd.read_csv(url)

data = df['value'].values
sequence_length = 30


def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


xs, ys = create_sequences(data, sequence_length)

train_ratio = 0.8
train_size = int(len(xs) * train_ratio)
x_train, x_test = xs[:train_size], xs[train_size:]
y_train, y_test = ys[:train_size], ys[train_size:]

x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

from torch.utils.data import DataLoader, Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


train_dataset = TimeSeriesDataset(x_train, y_train)
test_dataset = TimeSeriesDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 模型定义
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerTimeSeries(nn.Module):
    def __init__(self, feature_size=32, num_layers=3, num_heads=8, dropout=0.1):
        super(TransformerTimeSeries, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output[:, -1])
        return output


# 初始化模型
model = TransformerTimeSeries(feature_size=32, num_layers=3, num_heads=8)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 检查是否可以使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 训练和验证函数
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0

    for batch, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(x.unsqueeze(2))  # 添加一个额外的维度作为特征维度
        loss = criterion(output.squeeze(), y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(train_loader)


def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x.unsqueeze(2))  # 添加一个额外的维度作为特征维度
            loss = criterion(output.squeeze(), y)
            test_loss += loss.item()

    return test_loss / len(test_loader)


# 训练参数
num_epochs = 20
best_model = None
best_loss = float('inf')

train_losses = []
test_losses = []

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss = evaluate(model, test_loader, criterion, device)

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    if test_loss < best_loss:
        best_loss = test_loss
        best_model = model.state_dict()

    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

# 加载最佳模型
model.load_state_dict(best_model)


# 预测
x_test = x_test.to(device)
model.eval()
with torch.no_grad():
    predictions = model(x_test.unsqueeze(2)).squeeze().cpu().numpy()

# 获取实际值
actual = y_test.cpu().numpy()

# 绘制结果图
plt.figure(figsize=(10, 6))
plt.plot(actual, label='Actual')
plt.plot(predictions, label='Predicted')
plt.xlabel('Time Step')
plt.ylabel('CPU Utilization')
plt.legend()
plt.title('Actual vs Predicted CPU Utilization')
plt.show()