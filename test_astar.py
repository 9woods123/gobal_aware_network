from astar import DynamicAstar
from grid import Map2D



import numpy as np
import matplotlib.pyplot as plt

# ====== 你的类 ======
# 假设 Map2D 和 DynamicAstar 已经 import

def visualize(map2d, path=None, start=None, goal=None):
    xs, ys, colors = [], [], []

    for (ix, iy), v in map2d.grid.items():
        p = map2d.grid_to_world((ix, iy))
        xs.append(p[0])
        ys.append(p[1])

        if v < 0.1:
            colors.append([1,1,1])      # free
        elif v > 0.9:
            colors.append([0,0,0])      # obstacle
        else:
            colors.append([0.7,0.7,0.7])  # unknown（更亮一点）

    plt.figure(figsize=(6,6))
    plt.scatter(xs, ys, c=colors, s=5)

    if path is not None and len(path) > 0:
        path = np.array(path)
        plt.plot(path[:,0], path[:,1], 'r-', linewidth=2)

    if start is not None:
        plt.scatter(start[0], start[1], c='green', s=80)

    if goal is not None:
        plt.scatter(goal[0], goal[1], c='blue', s=80)

    plt.axis("equal")
    plt.title("A* Path Planning")
    plt.show()


def test_astar():

    # =========================
    # 1. 创建地图
    # =========================
    map2d = Map2D(resolution=0.2)

    map2d.init_dense_maze(K=4, cell_size=3.0, wall_thickness=0.5, seed=5)


    # =========================
    # 2. 起点终点
    # =========================
    start = (-9, -6)
    goal  = (4.0, 9.0)

    # =========================
    # 3. A*
    # =========================
    planner = DynamicAstar(
        resolution=map2d.resolution/2, ## condition minStepLength>__resolution
        minStepLength=map2d.resolution,
        collision_radius=1,
        gird=map2d
    )

    planner.setStart(start)
    planner.setGoal(goal)

    path, success, nodes = planner.pathPlan(start_g=0)

    # =========================
    # 4. 打印结果
    # =========================
    print("Success:", success)
    print("Path length:", len(path))

    if len(path) > 0:
        print("First point:", path[0])
        print("Last point:", path[-1])

    print("Expanded nodes:", len(nodes))

    # =========================
    # 5. 可视化
    # =========================
    visualize(map2d, path, start, goal)


if __name__ == "__main__":
    test_astar()