from sklearn.manifold import TSNE # t-SNE降维可视化
import torch
import torch.nn as nn # neural network module
import torch.optim as optim # optimization module
import torchvision
import torchvision.transforms as transforms # data transformation module
from torch.utils.data import DataLoader # data loading module
from einops import rearrange, repeat # rearrange tensor
import numpy as np
import matplotlib.pyplot as plt # visualization module
from matplotlib.lines import Line2D # add legendfor visualization
from IPython import display # display images
import time # time module
import random # random module

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

###1. Vision Transformer (ViT) 模型实现

class ViT(nn.Module):
    def __init__(self, image_size=32, patch_size=4, num_classes=10,
                  dim=128, depth=6, heads=8, mlp_dim=256):
        super(ViT, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        # 图像分块和位置编码
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2))
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 分类器
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes))
        
        # 获取注意力权重
        self.attn_weights = []
        for i, layer in enumerate(self.transformer.layers):
            layer.self_attn.register_forward_hook(
                lambda module, input, output: self.attn_weights.append(output[1]))
    
    # 前向传播
    def forward(self, img):
        self.attn_weights = [] # 清空注意力权重

        # 图像分块 [batch, channels, H, W] -> [batch, num_patches, patch_dim]
        x = self.patch_embedding(img)
        x = rearrange(x, 'b d n -> b n d')

        # 添加类别嵌入和位置嵌入
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=x.shape[0])
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 添加位置嵌入
        x += self.pos_embedding[:, :x.shape[1]]
        
        # Transformer 编码器
        x = self.transformer(x)
        
        # 使用分类token的输出作为分类结果
        cls_output = x[:, 0]
        return self.mlp_head(cls_output)
    
    # 获取最后一层的注意力图
    def get_attention_maps(self):
        if len(self.attn_weights) == 0:
            return None
        
        # 获取最后一层的注意力权重【batch_size, num_heads, num_queries, num_keys】
        last_attn_weights = self.attn_weights[-1]
        return last_attn_weights.mean(dim=1)
    
### 2. 数据集准备（CIFAR-10）

transform = transforms.Compose([
    transforms.Resize(32,32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)

# 类别标签
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse','ship', 'truck']

### 3. 可视化工具函数

def setup_plots():
    # 设置实时更新图表
    plt.figure(figsize=(18, 12))

    # 训练损失图和准确率图
    plt.subplot(2, 3, 1)
    plt.title('Training and Validation Metrics')
    plt.xlabel('Epoch')
    plt.grid(True)

    # 注意力图
    plt.subplot(2, 3, 2)
    plt.title('Attention Visualization')
    plt.axis('off') # 关闭坐标轴

    # 预测示例
    plt.subplot(2, 3, 3)
    plt.title('Predictions')
    plt.axis('off')

    # 学习率曲线
    plt.subplot(2, 3, 4)
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.grid(True)

    # 混淆矩阵
    plt.subplot(2, 3, 5)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.grid(False)

    # 特征空间可视化
    plt.subplot(2, 3, 6)
    plt.title('Feature Space (t-SNE)') # t-SNE降维可视化
    plt.axis('off')

    plt.tight_layout() # 自动调整子图间距
    plt.show() 

# 更新所有图表
def update_plots(epoch, train_losses, val_accuracies, lr_history, model, test_sample):
    fig = plt.gcf()# 获取当前图表
    fig.clf()# 清空当前图表

    # 1. 训练损失图和准确率图
    ax1 = plt.subplot(2, 3, 1)
    ax1.set_title('Training and Validation Metrics')
    ax1.set_xlabel('Epoch')
    ax1.grid(True)

    #    绘制训练损失
    ax1.plot(train_losses, 'b-', label='Training Loss')
    ax1.set_ylabel('Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    #    创建第二个Y轴用于绘制准确率
    ax2 = ax1.twinx() # 共享X轴
    ax2.plot(val_accuracies, 'r-', label='Validation Accuracy')
    ax2.set_ylabel('Accuracy (%)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    #    添加图例
    lines = [Line2D([0], [0], color='b', lw=2),
             Line2D([0], [0], color='r', lw=2)]
    ax1.legend(lines, ['Training Loss', 'Validation Accuracy'], loc = 'upper left')
    
    # 2. 注意力图
    ax3 = fig.add_subplot(2, 3, 2)
    ax3.set_title('Attention Visualization')
    ax3.axis('off')

    #    获取注意力权重
    if model is not None and test_sample is not None:
        with torch.no_grad():
            model.eval() # 评估模式
            test_sample = test_sample.to(device).unsqueeze(0) # 增加batch维度
            output = model(test_sample) # 前向传播
            attn_maps = model.get_attention_maps() # 获取注意力权重

            #    绘制注意力图
            if attn_maps is not None:
                attn_map = attn_maps[0, 0, 1:].detach().cpu().numpy() # 取出第一个头的注意力权重
                attn_map = attn_map.reshape(8, 8) # 重塑成8x8的矩阵

                #    显示原图
                img = test_sample.squeeze(0).permute(1, 2, 0).cpu().numpy() 
                img = (img - img.min())/(img.max() - img.min()) # 归一化

                #    显示注意力图热力图
                ax3.imshow(img)
                ax3.imshow(attn_map, cmap='hot', alpha=0.5, extent=(0, 32, 32, 0))
                ax3.set_title('Attention Heatmap (Epoch {epoch})')

    # 3. 预测示例
    ax4 = fig.add_subplot(2, 3, 3)
    ax4.set_title('Predictions Example')
    ax4.axis('off')

    #    随机选择一张测试样本
    if test_sample is not None:
        with torch.no_grad():
            model.eval() # 评估模式
            test_sample = test_sample.unsqueeze(0).to(device) # 增加batch维度
            output = model(test_sample) # 前向传播
            _, predicted = torch.max(output, 1) # 预测类别
            probs = torch.softmax(output, dim=1)[0] * 100 # 预测概率

            #    显示图片
            img = test_sample.squeeze(0).permute(1, 2, 0).cpu().numpy()
            img = (img - img.min())/(img.max() - img.min()) # 归一化
            ax4.imshow(img)

            #    显示预测结果
            pred_text = f"Predicted: {classes[predicted.item()]}\n"
            for i, prob in enumerate(probs):
                pred_text += f"{classes[i]}: {prob.item():.2f}%\n"
            ax4.text(35, 15, pred_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

    # 4. 学习率曲线
    ax5 = plt.subplot(2, 3, 4)
    ax5.set_title('Learning Rate')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Learning Rate')
    ax5.grid(True)
    ax5.plot(lr_history, 'g-')
    ax5.set_ylim(0, max(lr_history)*1.1)# 自动调整Y轴范围

    # 5. 混淆矩阵
    ax6 = plt.subplot(2, 3, 5)
    ax6.set_title('Confusion Matrix')
    ax6.set_xlabel('Predicted Label')
    ax6.set_ylabel('True Label')
    ax6.grid(False)

    #    计算混淆矩阵
    if epoch % 5 == 0 and epoch > 0:
        confusion_matrix = np.zeros((10, 10))
        model.eval() # 评估模式
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                for t, p in zip(labels.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

        #    归一化
        conf_norm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        ax6.imshow(conf_norm, interpolation='nearest', cmap=plt.cm.Blues)
        ax6.set_xticks(np.arange(10))# 设置X轴标签
        ax6.set_yticks(np.arange(10))# 设置Y轴标签
        ax6.set_xticklabels(classes, rotation=45, ha='right')# 设置X轴标签文本
        ax6.set_yticklabels(classes)# 设置Y轴标签文本
        ax6.set_ylim(9.5, -0.5)# 自动调整Y轴范围

        #    在格子中显示数字
        thresh = conf_norm.max() / 2.# 设定阈值
        for i in range(10):
            for j in range(10):
                ax6.text(j, i, f"{conf_norm[i, j]*100:.2f}", ha='center', va='center', color='white' if conf_norm[i, j] > thresh else 'black')
    
    # 6. 特征空间可视化
    ax7 = plt.subplot(2, 3, 6)
    ax7.set_title('Feature Space (t-SNE)')
    ax7.axis('off')

    #    计算特征空间
    if epoch % 10 == 0 and epoch > 0:
        features = []
        labels_list = []
        model.eval() # 评估模式
        with torch.no_grad():
            for i, (images, labels) in enumerate(train_loader):
                if i > 2: # 只取前3个batch
                    break
                images = images.to(device)
                x = model.patch_embedding(images)
                x = rearrange(x, 'b d n -> b n d')
                cls_tokens = repeat(model.cls_token, '1 1 d -> b 1 d', b=x.shape[0])
                x = torch.cat((cls_tokens, x), dim=1)
                x += model.pos_embedding[:, :(x.shape[1])]
                # 只取分类token的输出作为特征
                features.append(x[:, 0].cpu())
                labels_list.append(labels)

