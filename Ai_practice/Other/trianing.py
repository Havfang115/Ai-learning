import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 设置是否使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 下载并加载MNIST数据集
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 定义网络结构
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(), # 将输入数据展平为一维
            nn.Linear(784, 256), # 全连接层
            nn.ReLU(), # 激活函数
            nn.Linear(256, 64), # 隐藏层
            nn.ReLU(),
            nn.Linear(64, 10) # 输出层
        )
    def forward(self, x):
        return self.net(x)

# 实例化模型，损失函数，优化器
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
def train(model, loader, epoch=5):
    for i in range(epoch):
        model.train()
        running_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device) # 将数据转移到GPU
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录损失
            running_loss += loss.item()
        print('Epoch: {}, Loss: {:.6f}'.format(i+1, running_loss/len(loader)))

# 测试模型
def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy: {:.2f}%'.format(100*correct/total))

# 训练模型
train(model, train_loader, epoch=5)

# 测试模型
test(model, test_loader)