

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import matplotlib.pyplot as plt
# import torch.nn.functional as F

# from experts.grid import Map2D
# from experts.dist_map import DistanceField
# from models.encoders import MapEncoder, GNet


# # =========================
# # map → image
# # =========================
# def map2d_to_image(map2d, size=128):
#     img = np.zeros((size, size), dtype=np.float32)

#     for (ix, iy), v in map2d.grid.items():
#         p = map2d.grid_to_world((ix, iy))

#         x = int((p[0] + 10) / 20 * size)
#         y = int((p[1] + 10) / 20 * size)

#         if 0 <= x < size and 0 <= y < size:
#             img[y, x] = v

#     return img


# # =========================
# # 随机 goal
# # =========================
# def sample_goal(map2d):
#     keys = list(map2d.grid.keys())
#     while True:
#         idx = keys[np.random.randint(len(keys))]
#         if map2d.grid[idx] < 0.5:  # free
#             return map2d.grid_to_world(idx)


# # =========================
# # 从 dist_map 采样点
# # =========================
# def sample_points_from_distmap(df, N):
#     keys = list(df.dist_map.keys())
#     idxs = np.random.choice(len(keys), size=N)

#     pts = []
#     for i in idxs:
#         idx = keys[i]
#         pts.append(df.map2d.grid_to_world(idx))

#     return np.array(pts, dtype=np.float32)


# # =========================
# # 训练（多地图 + 多goal）
# # =========================
# def get_neighbors(x, step=0.2):
#     """
#     x: [B,2]
#     return: [B, K, 2]
#     """

#     dx = torch.tensor([
#         [ step, 0],
#         [-step, 0],
#         [0,  step],
#         [0, -step],
#         [ step,  step],
#         [ step, -step],
#         [-step,  step],
#         [-step, -step],
#     ], dtype=torch.float32).to(x.device)

#     neighbors = x.unsqueeze(1) + dx.unsqueeze(0)  # [B,8,2]

#     return neighbors

# def select_best_next(df, x_tensor):
#     """
#     x_tensor: [B,2]

#     return:
#         x_next: [B,2]
#     """

#     neighbors = get_neighbors(x_tensor)   # [B,8,2]

#     B, K, _ = neighbors.shape

#     # flatten 查询 dist_map
#     neighbors_flat = neighbors.view(-1, 2)

#     dist_vals = df.query_tensor(neighbors_flat)  # [B*K,1]
#     dist_vals = dist_vals.view(B, K)

#     # 找最小的
#     idx = torch.argmin(dist_vals, dim=1)  # [B]

#     # gather
#     x_next = neighbors[torch.arange(B), idx]

#     return x_next

# def bellman_loss(g_net, map_feat, goal_tensor, x_tensor, df):

#     # 当前
#     G_x = g_net(x_tensor, map_feat, goal_tensor)

#     # teacher 下一步
#     x_next = select_best_next(df, x_tensor)

#     # 下一步 G
#     G_next = g_net(x_next, map_feat, goal_tensor).detach()  # ⭐ detach！

#     # step cost
#     l = torch.norm(x_next - x_tensor, dim=-1, keepdim=True)

#     target = l + G_next

#     return F.mse_loss(G_x, target)

# def train_gnet(map_encoder, g_net, device):

#     optimizer = torch.optim.Adam(
#         list(map_encoder.parameters()) + list(g_net.parameters()),
#         lr=1e-3
#     )

#     for map_it in range(200):   # 外层：地图

#         # ===== 1️⃣ 生成一个地图 =====
#         map2d = Map2D(resolution=0.2)
#         map2d.init_dense_maze(
#             K=np.random.randint(2, 4),
#             cell_size=3.0,
#             wall_thickness=0.5,
#             seed=np.random.randint(1000)
#         )

#         # ===== 2️⃣ map → tensor（只做一次！）=====
#         img = map2d_to_image(map2d, size=128)
#         map_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)
        
#         loss=0

#         # ===== 4️⃣ 多个 goal =====
#         for goal_it in range(50):   # ⭐ 一个地图配多个goal

#             map_feat= map_encoder(map_tensor)  # [1, C]

#             goal = sample_goal(map2d)
            
#             goal_tensor = torch.tensor([[goal[0], goal[1]]], dtype=torch.float32).to(device)

#             # ===== distance field（只对 goal）=====
#             df = DistanceField(map2d, mode="conservative", collision_radius=0.5)
#             df.compute(goal)

#             # ===== 采样 =====
#             pts = sample_points_from_distmap(df, N=512)
#             x_tensor = torch.from_numpy(pts).to(device)

#             G_gt = df.query_tensor(x_tensor).to(device)

#             # ===== expand =====

#             # ===== forward =====
#             G_pred = g_net(x_tensor, map_feat, goal_tensor)

#             loss_sup = F.mse_loss(G_pred, G_gt)

#             loss_bellman = bellman_loss(
#                 g_net, map_feat, goal_tensor, x_tensor, df
#             )

#             loss = loss_sup + 0.5 * loss_bellman

#             # ===== optimize =====
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
        
        
#         print(f"[Iter {map_it}] loss = {loss.item():.4f}")
            
#     print(f"[Map {map_it}] done")


# # =========================
# # 测试 + 可视化
# # =========================
# def evaluate_and_visualize(map_encoder, g_net, device):

#     # 固定一个测试地图
#     map2d = Map2D(resolution=0.2)
#     map2d.init_dense_maze(K=4, cell_size=3.0, wall_thickness=0.5, seed=42)

#     goal = (4.0, 9.0)

#     df = DistanceField(map2d, mode="conservative", collision_radius=0.5)
#     df.compute(goal)

#     img = map2d_to_image(map2d, size=128)
#     map_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)
#     goal_tensor = torch.tensor([[goal[0], goal[1]]], dtype=torch.float32).to(device)

#     map_feat = map_encoder(map_tensor)

#     xs, ys = [], []
#     vals_gt, vals_pred = [], []

#     for (ix, iy), dist in df.dist_map.items():
#         p = map2d.grid_to_world((ix, iy))
#         x, y = p

#         pt = torch.tensor([[x, y]], dtype=torch.float32).to(device)

#         G_p = g_net(pt, map_feat, goal_tensor).item()

#         xs.append(x)
#         ys.append(y)
#         vals_gt.append(dist)
#         vals_pred.append(G_p)

#     xs = np.array(xs)
#     ys = np.array(ys)
#     vals_gt = np.array(vals_gt)
#     vals_pred = np.array(vals_pred)
#     vals_err = np.abs(vals_gt - vals_pred)

#     # 障碍
#     obs_x, obs_y = [], []
#     for (ix, iy), prob in map2d.grid.items():
#         if prob > 0.9:
#             p = map2d.grid_to_world((ix, iy))
#             obs_x.append(p[0])
#             obs_y.append(p[1])

#     # ===== plot =====
#     plt.figure(figsize=(15, 4))

#     goal_x, goal_y = goal

#     # ---------- GT ----------
#     plt.subplot(1, 3, 1)
#     plt.scatter(xs, ys, c=vals_gt, cmap='jet', s=8)
#     plt.scatter(obs_x, obs_y, c='black', s=8)
#     plt.scatter(goal_x, goal_y, c='red', s=120, marker='*')
#     plt.title("GT Distance Field")

#     # ---------- Pred ----------
#     plt.subplot(1, 3, 2)
#     plt.scatter(xs, ys, c=vals_pred, cmap='jet', s=8)
#     plt.scatter(obs_x, obs_y, c='black', s=8)
#     plt.scatter(goal_x, goal_y, c='red', s=120, marker='*')
#     plt.title("Predicted G")

#     # ---------- Error ----------
#     plt.subplot(1, 3, 3)
#     plt.scatter(xs, ys, c=vals_err, cmap='hot', s=8)
#     plt.scatter(obs_x, obs_y, c='black', s=8)
#     plt.scatter(goal_x, goal_y, c='red', s=120, marker='*')
#     plt.title("Error")

#     plt.suptitle(f"Map  Goal ")
#     plt.tight_layout()
#     plt.show()



# def evaluate_multiple(map_encoder, g_net, device, num_maps=3, goals_per_map=3):

#     map_encoder.eval()
#     g_net.eval()

#     all_errors = []

#     for map_id in range(num_maps):

#         # ===== 随机地图 =====
#         map2d = Map2D(resolution=0.2)
#         map2d.init_dense_maze(
#             K=np.random.randint(2, 4),
#             cell_size=3.0,
#             wall_thickness=0.5,
#             seed=np.random.randint(1000)
#         )

#         img = map2d_to_image(map2d, size=128)
#         map_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)

#         # ⭐ 只encode一次
#         with torch.no_grad():
#             map_feat = map_encoder(map_tensor)

#         print(f"\n===== Map {map_id} =====")

#         for goal_id in range(goals_per_map):

#             goal = sample_goal(map2d)

#             goal_tensor = torch.tensor([[goal[0], goal[1]]], dtype=torch.float32).to(device)

#             df = DistanceField(map2d, mode="conservative", collision_radius=0.5)
#             df.compute(goal)

#             xs, ys = [], []
#             vals_gt, vals_pred = [], []

#             with torch.no_grad():

#                 for (ix, iy), dist in df.dist_map.items():

#                     p = map2d.grid_to_world((ix, iy))
#                     x, y = p

#                     pt = torch.tensor([[x, y]], dtype=torch.float32).to(device)

#                     G_p = g_net(pt, map_feat, goal_tensor).item()

#                     xs.append(x)
#                     ys.append(y)
#                     vals_gt.append(dist)
#                     vals_pred.append(G_p)

#             vals_gt = np.array(vals_gt)
#             vals_pred = np.array(vals_pred)
#             vals_err = np.abs(vals_gt - vals_pred)

#             mean_err = vals_err.mean()
#             max_err = vals_err.max()

#             all_errors.append(mean_err)

#             print(f"[Map {map_id} | Goal {goal_id}] mean err = {mean_err:.3f}, max err = {max_err:.3f}")

#             # ===== 可视化（只画前几个）=====
#             if map_id < 4 and goal_id < 4:

#                 xs = np.array(xs)
#                 ys = np.array(ys)

#                 # 障碍
#                 obs_x, obs_y = [], []
#                 for (ix, iy), prob in map2d.grid.items():
#                     if prob > 0.9:
#                         p = map2d.grid_to_world((ix, iy))
#                         obs_x.append(p[0])
#                         obs_y.append(p[1])

#                 plt.figure(figsize=(15, 4))

#                 goal_x, goal_y = goal

#                 # ---------- GT ----------
#                 plt.subplot(1, 3, 1)
#                 plt.scatter(xs, ys, c=vals_gt, cmap='jet', s=8)
#                 plt.scatter(obs_x, obs_y, c='black', s=8)
#                 plt.scatter(goal_x, goal_y, c='red', s=120, marker='*')
#                 plt.title("GT Distance Field")

#                 # ---------- Pred ----------
#                 plt.subplot(1, 3, 2)
#                 plt.scatter(xs, ys, c=vals_pred, cmap='jet', s=8)
#                 plt.scatter(obs_x, obs_y, c='black', s=8)
#                 plt.scatter(goal_x, goal_y, c='red', s=120, marker='*')
#                 plt.title("Predicted G")

#                 # ---------- Error ----------
#                 plt.subplot(1, 3, 3)
#                 plt.scatter(xs, ys, c=vals_err, cmap='hot', s=8)
#                 plt.scatter(obs_x, obs_y, c='black', s=8)
#                 plt.scatter(goal_x, goal_y, c='red', s=120, marker='*')
#                 plt.title("Error")

#                 plt.suptitle(f"Map {map_id} Goal {goal_id}")
#                 plt.tight_layout()
#                 plt.show()

#     print("\n======================")
#     print(f"Overall mean error: {np.mean(all_errors):.4f}")
#     print("======================")




# # =========================
# # MAIN
# # =========================
# def main():

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     map_encoder = MapEncoder().to(device)
#     g_net = GNet(
#         state_dim=2,
#         map_feat_dim=map_encoder.get_map_feature_dim(),
#         goal_dim=2
#     ).to(device)

#     train_gnet(map_encoder, g_net, device)

#     # evaluate_and_visualize(map_encoder, g_net, device)
#     evaluate_multiple(
#         map_encoder,
#         g_net,
#         device,
#         num_maps=5,
#         goals_per_map=5
#     )

# if __name__ == "__main__":
#     main()



import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

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
def get_neighbors(x, step=0.2):
    """
    x: [B,2]
    return: [B, K, 2]
    """

    dx = torch.tensor([
        [ step, 0],
        [-step, 0],
        [0,  step],
        [0, -step],
        [ step,  step],
        [ step, -step],
        [-step,  step],
        [-step, -step],
    ], dtype=torch.float32).to(x.device)

    neighbors = x.unsqueeze(1) + dx.unsqueeze(0)  # [B,8,2]

    return neighbors




def select_best_next(df, x_tensor):
    """
    x_tensor: [B,2]

    return:
        x_next: [B,2]
    """

    neighbors = get_neighbors(x_tensor)   # [B,8,2]

    B, K, _ = neighbors.shape

    # flatten 查询 dist_map
    neighbors_flat = neighbors.view(-1, 2)

    dist_vals = df.query_tensor(neighbors_flat)  # [B*K,1]
    dist_vals = dist_vals.view(B, K)

    # 找最小的
    idx = torch.argmin(dist_vals, dim=1)  # [B]

    # gather
    x_next = neighbors[torch.arange(B), idx]

    return x_next

def bellman_loss(g_net, map_feat, goal_tensor, x_tensor, df):

    # 当前
    G_x = g_net(x_tensor, map_feat, goal_tensor)

    # teacher 下一步
    x_next = select_best_next(df, x_tensor)

    # 下一步 G
    G_next = g_net(x_next, map_feat, goal_tensor).detach()  # ⭐ detach！

    # step cost
    l = torch.norm(x_next - x_tensor, dim=-1, keepdim=True)

    target = l + G_next

    return F.mse_loss(G_x, target)

def eikonal_loss(G, x):
    """
    G: [B,1]
    x: [B,2] with requires_grad=True
    """

    grad = torch.autograd.grad(
        outputs=G,
        inputs=x,
        grad_outputs=torch.ones_like(G),
        create_graph=True
    )[0]   # [B,2]

    grad_norm = torch.norm(grad, dim=1)

    return ((grad_norm - 1.0) ** 2).mean()


def train_gnet(map_encoder, g_net, device):

    optimizer = torch.optim.Adam(
        list(map_encoder.parameters()) + list(g_net.parameters()),
        lr=1e-3
    )

    for map_it in range(2500):

        # ===== 1️⃣ map =====
        map2d = Map2D(resolution=0.2)
        map2d.init_dense_maze(
            K=np.random.randint(2, 4),
            cell_size=3.0,
            wall_thickness=0.5,
            seed=np.random.randint(1000)
        )

        # ===== 2️⃣ map tensor =====
        img = map2d_to_image(map2d, size=128)
        map_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)

        # ===== 统计 =====
        loss_sup_list = []
        loss_bell_list = []
        loss_total_list = []

        # ===== 多 goal =====
        for goal_it in range(50):

            map_feat = map_encoder(map_tensor)

            goal = sample_goal(map2d)
            goal_tensor = torch.tensor([[goal[0], goal[1]]], dtype=torch.float32).to(device)

            df = DistanceField(map2d, mode="conservative", collision_radius=0.5)
            df.compute(goal)

            pts = sample_points_from_distmap(df, N=512)
            
            x_tensor = torch.from_numpy(pts).float().to(device)
            x_tensor = x_tensor.clone().detach().requires_grad_(True)



            G_gt = df.query_tensor(x_tensor).to(device)

            # ===== forward =====
            G_pred = g_net(x_tensor, map_feat, goal_tensor)

            loss_sup = F.mse_loss(G_pred, G_gt)
            
            loss_eikonal=eikonal_loss(G_pred,x_tensor)

            loss_bellman = bellman_loss(
                g_net, map_feat, goal_tensor, x_tensor, df
            )

            loss = loss_sup + 10* loss_bellman

            # ===== optimize =====
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ===== 记录 =====
            loss_sup_list.append(loss_sup.item())
            loss_bell_list.append(loss_bellman.item())
            loss_total_list.append(loss.item())

            # ===== 每10个goal打印一次（细粒度）=====
            if goal_it % 10 == 0:
                print(
                    f"[Map {map_it} | Goal {goal_it}] "
                    f"sup={loss_sup.item():.3f} "
                    f"bell={loss_bellman.item():.3f} "
                    f"total={loss.item():.3f}"
                )

        # ===== 每个 map 的统计 =====
        print(
            f"\n[Map {map_it} SUMMARY] "
            f"sup={np.mean(loss_sup_list):.3f} "
            f"bell={np.mean(loss_bell_list):.3f} "
            f"total={np.mean(loss_total_list):.3f}\n"
        )

    print("✅ Training Done!")


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



def evaluate_multiple(map_encoder, g_net, device, num_maps=5, goals_per_map=5):

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
        num_maps=10,
        goals_per_map=5
    )

if __name__ == "__main__":
    main()