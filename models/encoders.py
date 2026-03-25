import torch
import torch.nn as nn
import torch.nn.functional as F


# class MapEncoder(nn.Module):

#     def __init__(self, map_dim= 128):
#         super(MapEncoder, self).__init__()
#         self.map_dim=map_dim
#         self.map_feature_dim=64
#         # in_channels=1 → 输入通道数，比如灰度图只有1通道；RGB图是3通道。
#         # out_channels=32 → 输出特征图（feature map）数量，也就是“滤波器”数量。
#         # kernel_size=5 → 卷积核大小 5×5。
#         # stride=1 → 步长，每次卷积移动1个像素。
#         # padding=2 → 边缘填充2个像素，保证卷积后特征图大小不变（same convolution）
        
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.relu1 = nn.ReLU()

#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.relu2 = nn.ReLU()
        
#         self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
#         self.bn3 = nn.BatchNorm2d(64)
#         self.relu3 = nn.ReLU()
        
#         self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.bn4 = nn.BatchNorm2d(64)
#         self.relu4 = nn.ReLU()

#         self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
#         self.bn5 = nn.BatchNorm2d(128)
#         self.relu5 = nn.ReLU()
        
#         self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
#         self.bn6 = nn.BatchNorm2d(128)
#         self.relu6 = nn.ReLU()
        
#         self.conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
#         self.bn7 = nn.BatchNorm2d(256)
#         self.relu7 = nn.ReLU()
        
#         self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
#         self.bn8 = nn.BatchNorm2d(256)
#         self.relu8 = nn.ReLU()
        
#         self.fc1 = nn.Linear(256, 128)
#         self.relu_fc1 = nn.ReLU()
        
#         self.fc2 = nn.Linear(128,  self.map_feature_dim)
#         self.relu_fc2 = nn.ReLU()
        
#         self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.max_pool4 = nn.MaxPool2d(kernel_size=4, stride=4)


#     def forward(self, x):

#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu1(x)

#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu2(x)

#         x = self.max_pool2(x)
#         # [B, 32, 64, 64]

#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = self.relu3(x)
        
#         x = self.conv4(x)
#         x = self.bn4(x)
#         x = self.relu4(x)

#         x = self.max_pool4(x)

#         x = self.conv5(x)
#         x = self.bn5(x)
#         x = self.relu5(x)
        
#         x = self.conv6(x)
#         x = self.bn6(x)
#         x = self.relu6(x)
        
#         x = self.max_pool4(x)

#         x = self.conv7(x)
#         x = self.bn7(x)
#         x = self.relu7(x)
        
#         x = self.conv8(x)
#         x = self.bn8(x)
#         x = self.relu8(x)
#         x = self.max_pool4(x)

#         x = x.view(x.size(0), -1).contiguous()

#         x = self.fc1(x)
#         x = self.relu_fc1(x)
        
#         x = self.fc2(x)
#         x = self.relu_fc2(x)

#         # output: [B,  self.map_feature_dim]
        
#         return x
    
#     def get_map_feature_dim(self):
#         return self.map_feature_dim

    


# class GNet(nn.Module):
#     def __init__(self, state_dim=4, map_feat_dim=64, goal_dim=4, hidden_dim=256):
#         super().__init__()

#         # print("state_dim + map_feat_dim + goal_dim:",state_dim + map_feat_dim + goal_dim)
        
#         self.mlp = nn.Sequential(
#             nn.Linear(state_dim + map_feat_dim + goal_dim, hidden_dim),
#             nn.ReLU(),
            
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),

#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),

#             nn.Linear(hidden_dim, 64),
#             nn.ReLU(),

#             nn.Linear(64, 1)   # 输出 cost-to-go
#         )

#     def forward(self, x, map_feat, goal):

#         inp = torch.cat([x, map_feat, goal], dim=-1)

#         return self.mlp(inp)
    

import torch.nn.functional as F

def sample_feature(feat_map, points, map_range=10.0):
    """
    feat_map: [B, C, H, W]
    points:   [B, 2]  (world coords)

    return:   [B, C]
    """

    B, C, H, W = feat_map.shape

    # ===== world → [-1, 1] =====
    px = points[:, 0] / map_range
    py = points[:, 1] / map_range   

    grid = torch.stack([px, py], dim=-1)  # [B,2]
    grid = grid.unsqueeze(1).unsqueeze(1) # [B,1,1,2]

    sampled = F.grid_sample(
        feat_map,
        grid,
        mode='bilinear',
        align_corners=True
    )  # [B,C,1,1]

    return sampled.view(B, C)



class MapEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        """
        input:  [B, 1, 128, 128]
        output: [B, 64, 32, 32]
        """

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)     # 64

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)     # 32

        x = self.conv5(x)

        return x   # ⭐ 保留 spatial

    def get_map_feature_dim(self):
        return 64
    
    

class GNet(nn.Module):
    def __init__(self, state_dim=4, map_feat_dim=64, goal_dim=4, hidden_dim=256):
        super().__init__()

        # ⭐ 多了一个 feat_goal
        input_dim = state_dim + goal_dim + map_feat_dim * 2

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x, map_feat, goal):

        # ===== 只取位置 =====
        p = x[..., :2]
        g = goal[..., :2]

        # ===== sample =====
        feat_x = sample_feature(map_feat, p)
        feat_goal = sample_feature(map_feat, g)

        # ===== concat =====
        inp = torch.cat([
            x,
            goal,
            feat_x,
            feat_goal
        ], dim=-1)

        return self.mlp(inp)