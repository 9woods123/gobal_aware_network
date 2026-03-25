import numpy as np
import matplotlib.pyplot as plt

from experts.grid import Map2D
from experts.dist_map import DistanceField


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

def test_distancemap():

    map2d = Map2D(resolution=0.2)

    map2d.init_dense_maze(
        K=4,
        cell_size=3.0,
        wall_thickness=0.5,
        seed=5
    )

    start = (-9, -6)
    goal  = (4.0, 9.0)

    df = DistanceField(map2d, mode="conservative",collision_radius=0.5)

    dist_map = df.compute(goal)
    
    path, success = df.get_optimal_path(start, goal)

    print("Path success:", success)
    print("Path length:", len(path))
    print("Total reachable nodes:", len(dist_map))
    print("Distance at start:", df.query(start))
    print("Distance at goal:", df.query(goal))

    # ⭐ 这里传 path
    visualize_distance_field(map2d, df, path, start, goal)

if __name__ == "__main__":
    test_distancemap()