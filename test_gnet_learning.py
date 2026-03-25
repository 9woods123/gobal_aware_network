

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from experts.grid import Map2D
from experts.dist_map import DistanceField
from models.encoders import MapEncoder, GNet


# =========================
# map → image
# =========================
def map2d_to_image(map2d, size=128):
    img = np.zeros((size, size), dtype=np.float32)

    for (ix, iy), v in map2d.grid.items():
        p = map2d.grid_to_world((ix, iy))

        x = int((p[0] + 10) / 20 * size)
        y = int((p[1] + 10) / 20 * size)

        if 0 <= x < size and 0 <= y < size:
            img[y, x] = v

    return img


# =========================
# 随机 goal
# =========================
def sample_goal(map2d):
    keys = list(map2d.grid.keys())
    while True:
        idx = keys[np.random.randint(len(keys))]
        if map2d.grid[idx] < 0.5:  # free
            return map2d.grid_to_world(idx)


# =========================
# 从 dist_map 采样点
# =========================
def sample_points_from_distmap(df, N):
    keys = list(df.dist_map.keys())
    idxs = np.random.choice(len(keys), size=N)

    pts = []
    for i in idxs:
        idx = keys[i]
        pts.append(df.map2d.grid_to_world(idx))

    return np.array(pts, dtype=np.float32)


# =========================
# 训练（多地图 + 多goal）
# =========================

def train_gnet(map_encoder, g_net, device):

    optimizer = torch.optim.Adam(
        list(map_encoder.parameters()) + list(g_net.parameters()),
        lr=1e-3
    )

    for map_it in range(200):   # 外层：地图

        # ===== 1️⃣ 生成一个地图 =====
        map2d = Map2D(resolution=0.2)
        map2d.init_dense_maze(
            K=np.random.randint(2, 4),
            cell_size=3.0,
            wall_thickness=0.5,
            seed=np.random.randint(1000)
        )

        # ===== 2️⃣ map → tensor（只做一次！）=====
        img = map2d_to_image(map2d, size=128)
        map_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)
        
        loss=0

        # ===== 4️⃣ 多个 goal =====
        for goal_it in range(50):   # ⭐ 一个地图配多个goal

            map_feat_single = map_encoder(map_tensor)  # [1, C]

            goal = sample_goal(map2d)
            
            goal_tensor = torch.tensor([[goal[0], goal[1]]], dtype=torch.float32).to(device)

            # ===== distance field（只对 goal）=====
            df = DistanceField(map2d, mode="conservative", collision_radius=0.5)
            df.compute(goal)

            # ===== 采样 =====
            pts = sample_points_from_distmap(df, N=512)
            x_tensor = torch.from_numpy(pts).to(device)

            G_gt = df.query_tensor(x_tensor).to(device)

            # ===== expand =====
            B = x_tensor.shape[0]
            map_feat = map_feat_single.expand(B, -1)
            goal_expand = goal_tensor.expand(B, -1)

            # ===== forward =====
            G_pred = g_net(x_tensor, map_feat, goal_expand)

            loss = nn.MSELoss()(G_pred, G_gt)

            # ===== optimize =====
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        
        print(f"[Iter {map_it}] loss = {loss.item():.4f}")
            
    print(f"[Map {map_it}] done")


# =========================
# 测试 + 可视化
# =========================
def evaluate_and_visualize(map_encoder, g_net, device):

    # 固定一个测试地图
    map2d = Map2D(resolution=0.2)
    map2d.init_dense_maze(K=4, cell_size=3.0, wall_thickness=0.5, seed=42)

    goal = (4.0, 9.0)

    df = DistanceField(map2d, mode="conservative", collision_radius=0.5)
    df.compute(goal)

    img = map2d_to_image(map2d, size=128)
    map_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)
    goal_tensor = torch.tensor([[goal[0], goal[1]]], dtype=torch.float32).to(device)

    map_feat = map_encoder(map_tensor)

    xs, ys = [], []
    vals_gt, vals_pred = [], []

    for (ix, iy), dist in df.dist_map.items():
        p = map2d.grid_to_world((ix, iy))
        x, y = p

        pt = torch.tensor([[x, y]], dtype=torch.float32).to(device)

        G_p = g_net(pt, map_feat, goal_tensor).item()

        xs.append(x)
        ys.append(y)
        vals_gt.append(dist)
        vals_pred.append(G_p)

    xs = np.array(xs)
    ys = np.array(ys)
    vals_gt = np.array(vals_gt)
    vals_pred = np.array(vals_pred)
    vals_err = np.abs(vals_gt - vals_pred)

    # 障碍
    obs_x, obs_y = [], []
    for (ix, iy), prob in map2d.grid.items():
        if prob > 0.9:
            p = map2d.grid_to_world((ix, iy))
            obs_x.append(p[0])
            obs_y.append(p[1])

    # ===== plot =====
    plt.figure(figsize=(15, 4))

    goal_x, goal_y = goal

    # ---------- GT ----------
    plt.subplot(1, 3, 1)
    plt.scatter(xs, ys, c=vals_gt, cmap='jet', s=8)
    plt.scatter(obs_x, obs_y, c='black', s=8)
    plt.scatter(goal_x, goal_y, c='red', s=120, marker='*')
    plt.title("GT Distance Field")

    # ---------- Pred ----------
    plt.subplot(1, 3, 2)
    plt.scatter(xs, ys, c=vals_pred, cmap='jet', s=8)
    plt.scatter(obs_x, obs_y, c='black', s=8)
    plt.scatter(goal_x, goal_y, c='red', s=120, marker='*')
    plt.title("Predicted G")

    # ---------- Error ----------
    plt.subplot(1, 3, 3)
    plt.scatter(xs, ys, c=vals_err, cmap='hot', s=8)
    plt.scatter(obs_x, obs_y, c='black', s=8)
    plt.scatter(goal_x, goal_y, c='red', s=120, marker='*')
    plt.title("Error")

    plt.suptitle(f"Map  Goal ")
    plt.tight_layout()
    plt.show()



def evaluate_multiple(map_encoder, g_net, device, num_maps=3, goals_per_map=3):

    map_encoder.eval()
    g_net.eval()

    all_errors = []

    for map_id in range(num_maps):

        # ===== 随机地图 =====
        map2d = Map2D(resolution=0.2)
        map2d.init_dense_maze(
            K=np.random.randint(2, 4),
            cell_size=3.0,
            wall_thickness=0.5,
            seed=np.random.randint(1000)
        )

        img = map2d_to_image(map2d, size=128)
        map_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)

        # ⭐ 只encode一次
        with torch.no_grad():
            map_feat = map_encoder(map_tensor)

        print(f"\n===== Map {map_id} =====")

        for goal_id in range(goals_per_map):

            goal = sample_goal(map2d)

            goal_tensor = torch.tensor([[goal[0], goal[1]]], dtype=torch.float32).to(device)

            df = DistanceField(map2d, mode="conservative", collision_radius=0.5)
            df.compute(goal)

            xs, ys = [], []
            vals_gt, vals_pred = [], []

            with torch.no_grad():

                for (ix, iy), dist in df.dist_map.items():

                    p = map2d.grid_to_world((ix, iy))
                    x, y = p

                    pt = torch.tensor([[x, y]], dtype=torch.float32).to(device)

                    G_p = g_net(pt, map_feat, goal_tensor).item()

                    xs.append(x)
                    ys.append(y)
                    vals_gt.append(dist)
                    vals_pred.append(G_p)

            vals_gt = np.array(vals_gt)
            vals_pred = np.array(vals_pred)
            vals_err = np.abs(vals_gt - vals_pred)

            mean_err = vals_err.mean()
            max_err = vals_err.max()

            all_errors.append(mean_err)

            print(f"[Map {map_id} | Goal {goal_id}] mean err = {mean_err:.3f}, max err = {max_err:.3f}")

            # ===== 可视化（只画前几个）=====
            if map_id < 4 and goal_id < 4:

                xs = np.array(xs)
                ys = np.array(ys)

                # 障碍
                obs_x, obs_y = [], []
                for (ix, iy), prob in map2d.grid.items():
                    if prob > 0.9:
                        p = map2d.grid_to_world((ix, iy))
                        obs_x.append(p[0])
                        obs_y.append(p[1])

                plt.figure(figsize=(15, 4))

                goal_x, goal_y = goal

                # ---------- GT ----------
                plt.subplot(1, 3, 1)
                plt.scatter(xs, ys, c=vals_gt, cmap='jet', s=8)
                plt.scatter(obs_x, obs_y, c='black', s=8)
                plt.scatter(goal_x, goal_y, c='red', s=120, marker='*')
                plt.title("GT Distance Field")

                # ---------- Pred ----------
                plt.subplot(1, 3, 2)
                plt.scatter(xs, ys, c=vals_pred, cmap='jet', s=8)
                plt.scatter(obs_x, obs_y, c='black', s=8)
                plt.scatter(goal_x, goal_y, c='red', s=120, marker='*')
                plt.title("Predicted G")

                # ---------- Error ----------
                plt.subplot(1, 3, 3)
                plt.scatter(xs, ys, c=vals_err, cmap='hot', s=8)
                plt.scatter(obs_x, obs_y, c='black', s=8)
                plt.scatter(goal_x, goal_y, c='red', s=120, marker='*')
                plt.title("Error")

                plt.suptitle(f"Map {map_id} Goal {goal_id}")
                plt.tight_layout()
                plt.show()

    print("\n======================")
    print(f"Overall mean error: {np.mean(all_errors):.4f}")
    print("======================")




# =========================
# MAIN
# =========================
def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    map_encoder = MapEncoder().to(device)
    g_net = GNet(
        state_dim=2,
        map_feat_dim=map_encoder.get_map_feature_dim(),
        goal_dim=2
    ).to(device)

    train_gnet(map_encoder, g_net, device)

    # evaluate_and_visualize(map_encoder, g_net, device)
    evaluate_multiple(
        map_encoder,
        g_net,
        device,
        num_maps=5,
        goals_per_map=5
    )

if __name__ == "__main__":
    main()