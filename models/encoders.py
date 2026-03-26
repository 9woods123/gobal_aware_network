import torch
import torch.nn as nn
import torch.nn.functional as F


import torch.nn.functional as F


def sample_feature(feat_map, points, map_range=10.0):
    """
    feat_map: [1, C, H, W]
    points:   [B, 2]

    return:   [B, C]
    """

    # print("feat_map.shape:",feat_map.shape)
    # print("points.shape:",points.shape)

    # feat_map.shape: torch.Size([1, 64, 32, 32])
    # points.shape: torch.Size([512, 2])

    Bp = points.shape[0]  ## Bp=512

    # ===== world → [-1,1] =====
    px = points[:, 0] / map_range
    py = points[:, 1] / map_range

    # ⭐ 关键：变成 [1, B, 1, 2]
    grid = torch.stack([px, py], dim=-1)   # [B,2]
    grid = grid.view(1, Bp, 1, 2)          # ✅


    sampled = F.grid_sample(
        feat_map,   # [1,C,H,W]
        grid,       # [1,B,1,2]
        mode='bilinear',
        align_corners=True
    )  # → [1, C, B, 1]

    # ===== reshape 成 [B,C] =====
    sampled = sampled.squeeze(-1)          # [1,C,B]
    sampled = sampled.permute(2, 1, 0)     # [B,C,1]
    sampled = sampled.squeeze(-1)          # [B,C]

    return sampled



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

        B = x.shape[0]

        # ===== expand goal =====
        goal_expand = goal.expand(B, -1)

        p = x[..., :2]
        g = goal_expand[..., :2]

        # ===== sample =====
        feat_x = sample_feature(map_feat, p)
        feat_goal = sample_feature(map_feat, g)

        # ===== concat =====
        inp = torch.cat([
            x,
            goal_expand,
            feat_x,
            feat_goal
        ], dim=-1)

        return self.mlp(inp)