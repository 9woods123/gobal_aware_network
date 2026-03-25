import heapq
import numpy as np


class DistanceField:
    def __init__(self, map2d, collision_radius ,mode="conservative"):
        """
        mode:
            "conservative" : unknown 当障碍
            "optimistic"   : unknown 可通行
        """
        self.map2d = map2d
        self.mode = mode
        self.dist_map = {}
        self.collision_radius=collision_radius
        self.motions = [
            (1,0), (-1,0), (0,1), (0,-1),
            (1,1), (-1,1), (1,-1), (-1,-1)
        ]

    # =========================
    # 可通行判断
    # =========================
    def is_traversable(self, p):
        """
        判断以 p 为中心、collision_radius 为半径的圆是否可通行
        """

        r = self.collision_radius
        res = self.map2d.resolution

        # 转成 grid 半径
        r_grid = int(np.ceil(r / res))

        cx, cy = self.map2d.world_to_grid(p)

        for dx in range(-r_grid, r_grid + 1):
            for dy in range(-r_grid, r_grid + 1):

                # ====== 圆形mask ======
                if dx*dx + dy*dy > r_grid*r_grid:
                    continue

                idx = (cx + dx, cy + dy)
                prob = self.map2d.grid[idx]

                # ====== 障碍 ======
                if prob > 0.9:
                    return False

                # ====== unknown策略 ======
                if self.mode == "conservative" and 0.1 <= prob <= 0.9:
                    return False

        return True

    # =========================
    # 核心：反向 Dijkstra
    # =========================
    def compute(self, goal):

        goal_idx = self.map2d.world_to_grid(goal)

        self.dist_map = {}
        visited = set()

        heap = []
        heapq.heappush(heap, (0.0, goal_idx))

        self.dist_map[goal_idx] = 0.0

        while heap:
            dist, idx = heapq.heappop(heap)

            if idx in visited:
                continue
            visited.add(idx)

            for dx, dy in self.motions:
                nidx = (idx[0] + dx, idx[1] + dy)

                p = self.map2d.grid_to_world(nidx)

                if not self.is_traversable(p):
                    continue

                step_cost = np.hypot(dx, dy) * self.map2d.resolution
                new_dist = dist + step_cost

                if nidx not in self.dist_map or new_dist < self.dist_map[nidx]:
                    self.dist_map[nidx] = new_dist
                    heapq.heappush(heap, (new_dist, nidx))
        return self.dist_map

    # =========================
    # 查询（numpy）
    # =========================
    def query(self, x, default=100.0):
        idx = self.map2d.world_to_grid(x)
        return self.dist_map.get(idx, default)

    # =========================
    # 查询（batch numpy）
    # =========================
    def query_batch(self, xs, default=100.0):
        values = []
        for x in xs:
            values.append(self.query(x, default))
        return np.array(values).reshape(-1, 1)

    # =========================
    # 查询（给 PyTorch）
    # =========================
    def query_tensor(self, x_tensor, default=100.0):
        """
        x_tensor: [B, D] (D >= 2)
        return:   [B, 1]
        """

        # ⭐ 只取前两维 (x, y)
        if x_tensor.shape[-1] > 2:
            pos = x_tensor[..., :2]
        else:
            pos = x_tensor

        xs = pos.detach().cpu().numpy()
        vals = self.query_batch(xs, default)

        return x_tensor.new_tensor(vals)

    # =========================
    # 可视化（debug用）
    # =========================
    def to_dense_map(self):
        """
        转成 grid array（用于可视化）
        """
        xs = [idx[0] for idx in self.dist_map.keys()]
        ys = [idx[1] for idx in self.dist_map.keys()]

        if len(xs) == 0:
            return None

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        grid = np.full((max_x-min_x+1, max_y-min_y+1), np.inf)

        for (ix, iy), v in self.dist_map.items():
            grid[ix-min_x, iy-min_y] = v

        return grid
    

        # =========================
    # 🔥 提取最优路径（从 start 到 goal）
    # =========================
    def get_optimal_path(self, start, goal=None, max_steps=1000):
        """
        从 start 沿着 distance field 下降，得到最优路径

        return:
            path: [(x,y), ...]
            success: bool
        """

        path = []
        current = self.map2d.world_to_grid(start)

        if current not in self.dist_map:
            return [], False

        path.append(self.map2d.grid_to_world(current))

        for _ in range(max_steps):

            current_dist = self.dist_map.get(current, np.inf)

            # 如果已经接近 goal（距离接近0）
            if current_dist < 1e-3:
                return path, True

            best_next = None
            best_dist = current_dist

            # 遍历邻居
            for dx, dy in self.motions:
                nidx = (current[0] + dx, current[1] + dy)

                if nidx not in self.dist_map:
                    continue

                d = self.dist_map[nidx]

                if d < best_dist:
                    best_dist = d
                    best_next = nidx

            # ❗ 没找到更小的（卡住了）
            if best_next is None:
                return path, False

            current = best_next
            path.append(self.map2d.grid_to_world(current))

        return path, False