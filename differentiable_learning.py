import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from experts.grid import Map2D
from experts.dist_map import DistanceField
from models.diff_models import GlobalPlannerModel

def visualize_distance_field(map2d, dist_field, path=None, start=None, goal=None):

    xs, ys = [], []
    vals = []

    # ========= 收集数据 =========
    for (ix, iy), prob in map2d.grid.items():
        p = map2d.grid_to_world((ix, iy))

        xs.append(p[0])
        ys.append(p[1])

        if (ix, iy) in dist_field.dist_map:
            vals.append(dist_field.dist_map[(ix, iy)])
        else:
            vals.append(np.nan)

    xs = np.array(xs)
    ys = np.array(ys)
    vals = np.array(vals)

    plt.figure(figsize=(7, 6))

    # ========= 距离场 =========
    sc = plt.scatter(xs, ys, c=vals, cmap='jet', s=8)
    plt.colorbar(sc, label="Distance to Goal")

    # ========= 障碍 =========
    obs_x, obs_y = [], []
    for (ix, iy), prob in map2d.grid.items():
        if prob > 0.9:
            p = map2d.grid_to_world((ix, iy))
            obs_x.append(p[0])
            obs_y.append(p[1])

    plt.scatter(obs_x, obs_y, c='black', s=8)
    plt.scatter([], [], c='black', s=80, label='obstacle')

    # ========= ⭐ 路径 =========
    if path is not None and len(path) > 0:
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], color='blue', linewidth=3, label='optimal path')
        plt.scatter(path[:, 0], path[:, 1], c='cyan', s=12)

    # ========= 起点终点 =========
    if start is not None:
        plt.scatter(start[0], start[1], c='green', s=80, label='start')

    if goal is not None:
        plt.scatter(goal[0], goal[1], c='red', s=80, label='goal')

    plt.title("Distance Field (Backward Dijkstra)")
    plt.axis("equal")
    plt.legend()
    plt.show()


def map2d_to_image(map2d, size=128):
    """
    转成 [size, size] 的图像
    0: free (白)
    1: obs (黑)
    0.5: unknown (灰)
    """
    img = np.ones((size, size)) * 0.5  # 默认 unknown

    # 找范围
    xs = [idx[0] for idx in map2d.grid.keys()]
    ys = [idx[1] for idx in map2d.grid.keys()]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    for (ix, iy), v in map2d.grid.items():
        # 归一化到 [0, size)
        x = int((ix - min_x) / (max_x - min_x + 1) * (size - 1))
        y = int((iy - min_y) / (max_y - min_y + 1) * (size - 1))

        img[y, x] = v   # 注意 y,x

    return img


def test_diff_model():

    map2d = Map2D(resolution=0.2)

    map2d.init_dense_maze(
        K=4,
        cell_size=3.0,
        wall_thickness=0.5,
        seed=5
    )

    start = (-9, -6)
    goal  = (4.0, 9.0)




    # ========= 距离场 =========
    df = DistanceField(map2d, mode="conservative", collision_radius=0.5)
    dist_map = df.compute(goal)



    # ========= 保存地图 =========
    img = map2d_to_image(map2d, size=128)


    plt.imshow(img, cmap='gray_r', vmin=0, vmax=1)
    plt.title("Map2D")
    plt.axis('off')
    plt.savefig("map.png", dpi=150)
    plt.show()


    # ========= 转 tensor =========
    map_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)


    x0 = torch.tensor([[start[0], start[1], 0.0, 0.0]], dtype=torch.float32)
    goal_tensor = torch.tensor([[goal[0], goal[1],0,0]], dtype=torch.float32)


    # ========= 模型 =========
    model = GlobalPlannerModel(
        state_dim=4,
        map_dim=128,
        goal_dim=4,
        hidden_dim=256
    )


    # ========= forward =========
    x_list, G_list, l_list = model.forward(
        map_tensor,
        x0,
        goal_tensor,
        H=20
    )



def train_on_single_map():

    # ========= 地图 =========
    map2d = Map2D(resolution=0.2)
    map2d.init_dense_maze(K=4, cell_size=3.0, wall_thickness=0.5, seed=5)

    start = (-9, -6)
    goal  = (4.0, 9.0)

    distance_field = DistanceField(map2d, mode="conservative", collision_radius=0.5)
    _ = distance_field.compute(goal)

    img = map2d_to_image(map2d, size=128)
    map_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)

    x0 = torch.tensor([[start[0], start[1], 0.0, 0.0]], dtype=torch.float32)
    goal_tensor = torch.tensor([[goal[0], goal[1], 0.0, 0.0]], dtype=torch.float32)

    # ========= 模型 =========
    model = GlobalPlannerModel(
        state_dim=4,
        map_dim=128,
        goal_dim=4,
        hidden_dim=256
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ========= 训练 =========
    for it in range(1000):

        optimizer.zero_grad()

        x_list, G_list, l_list = model.forward(
            map_tensor,
            x0,
            goal_tensor,
            H=30   # ⭐ 稍微长一点更好
        )

        loss, loss_dict = model.compute_loss(
            x_list,
            G_list,
            l_list,
            distance_field=distance_field
        )

        loss.backward()
        optimizer.step()

        # ========= 打印 =========
        if it % 50 == 0:
            print(f"[Iter {it}] Loss: {loss.item():.4f}", loss_dict)

    return model, map2d, distance_field


if __name__ == "__main__":
    train_on_single_map()