import torch
import torch.nn as nn
import torch.nn.functional as F
def global_average_pooling(tensor):
    # 对张量应用全局平均池化
    return F.adaptive_avg_pool2d(tensor, (1, 1))

# 示例
tensor = torch.randn(32, 256)  # 模拟一个四维张量
dense = nn.Linear(256, 56 * 56)
print(dense(tensor).shape)