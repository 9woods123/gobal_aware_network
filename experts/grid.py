import numpy as np
from collections import defaultdict

class Map2D:
    def __init__(self, resolution=0.2):
        self.resolution = resolution
        self.grid = defaultdict(lambda: 0.5)
        self.known = []
        self.unknown = []

    def world_to_grid(self, p):
        return (
            int(np.floor(p[0] / self.resolution)),
            int(np.floor(p[1] / self.resolution))
        )

    def grid_to_world(self, idx):
        return np.array([
            (idx[0] + 0.5) * self.resolution,
            (idx[1] + 0.5) * self.resolution
        ])

    def entropy(self, p):
        eps = 1e-6
        p = np.clip(p, eps, 1 - eps)
        return -p * np.log(p) - (1 - p) * np.log(1 - p)

    def init_circle_known(self, R, bound):
        for ix in range(-bound, bound):
            for iy in range(-bound, bound):
                d = np.hypot(ix, iy) * self.resolution
                if d < R:
                    self.grid[(ix, iy)] = 0.0
                    self.known.append((ix, iy))
                else:
                    self.grid[(ix, iy)] = 0.5
                    self.unknown.append((ix, iy))
               
    # =========================
    # 概率 & 状态
    # =========================
    def get_prob(self, p):
        idx = self.world_to_grid(p)
        return self.grid[idx]

    def is_free(self, p):
        return self.get_prob(p) < 0.1

    def is_occupied(self, p):
        return self.get_prob(p) > 0.9

    def is_unknown(self, p):
        prob = self.get_prob(p)
        return 0.1 <= prob <= 0.9


    # =========================
    # ⭐ A* 碰撞检测（核心新增）
    # =========================
    def collision_detection(self, p1, p2, radius=0.3, use_los=False):
        """
        use_los:
            True  -> 连续线段检测（精确）
            False -> 仅检测终点（快速）
        """

        if use_los:
            return self._collision_los(p1, p2, radius)
        else:
            return self._collision_endpoint(p2, radius)


    # =========================
    # 精确：line-of-sight
    # =========================
    def _collision_los(self, p1, p2, radius):
        p1 = np.array(p1)
        p2 = np.array(p2)

        dist = np.linalg.norm(p2 - p1)

        step = self.resolution * 0.5
        steps = int(np.ceil(dist / step))

        for i in range(steps + 1):
            t = i / max(steps, 1)
            p = (1 - t) * p1 + t * p2

            if self.check_circle_collision(p, radius):
                return True

        return False


    # =========================
    # 快速：只检查终点
    # =========================
    def _collision_endpoint(self, p, radius):
        """
        仅检查终点附近（快速近似）
        """
        return self.check_circle_collision(p, radius)


    def check_circle_collision(self, center, radius):
        """
        检查一个圆是否碰到障碍
        """
        r_grid = int(np.ceil(radius / self.resolution))

        cx, cy = self.world_to_grid(center)

        for dx in range(-r_grid, r_grid + 1):
            for dy in range(-r_grid, r_grid + 1):

                if dx*dx + dy*dy > r_grid*r_grid:
                    continue

                idx = (cx + dx, cy + dy)

                if self.grid[idx] > 0.9:  # occupied
                    return True

        return False

    def init_rectangle_known(self, center, width, height, bound):
        """
        初始化一个轴对齐的长方形已知区域

        center: (cx, cy) in world coordinates [m]
        width, height: rectangle size [m]
        bound_m: map half-size in meters (map spans [-bound_m, bound_m])
        """

        cx, cy = center
        hw = width * 0.5
        hh = height * 0.5

        # meters -> grid cells
        bound_cells = int(np.ceil(bound / self.resolution))

        self.known.clear()
        self.unknown.clear()

        for ix in range(-bound_cells, bound_cells):
            for iy in range(-bound_cells, bound_cells):

                p = self.grid_to_world((ix, iy))

                if (abs(p[0] - cx) <= hw) and (abs(p[1] - cy) <= hh):
                    self.grid[(ix, iy)] = 0.0   # free / known
                    self.known.append((ix, iy))
                else:
                    self.grid[(ix, iy)] = 0.5   # unknown
                    self.unknown.append((ix, iy))


    def entropy_at(self, p):
        return self.entropy(self.grid[self.world_to_grid(p)])

    def add_random_rectangular_obstacles(
        self,
        n_obs=4,
        w_range=(0.5, 4),
        h_range=(0.5, 6),
        seed=22
    ):
        
        ## seed=0

        if seed is not None:
            np.random.seed(seed)

        known_world = np.array(
            [self.grid_to_world(idx) for idx in self.known]
        )

        for _ in range(n_obs):
            c = known_world[np.random.randint(len(known_world))]
            w = np.random.uniform(*w_range)
            h = np.random.uniform(*h_range)

            for idx in list(self.known):
                p = self.grid_to_world(idx)
                if abs(p[0] - c[0]) <= w/2 and abs(p[1] - c[1]) <= h/2:
                    self.grid[idx] = 1.0
    
    def add_random_rectangular_obstacles(
        self,
        n_obs=4,
        w_range=(0.5, 4),
        h_range=(0.5, 6),
        seed=22,
        w_bound=3,
        h_bound=3
        ):

            rng = np.random.default_rng(seed)

            for _ in range(n_obs):

                # 1️⃣ 全局 world 坐标采样
                c = np.array([
                    rng.uniform(-w_bound, w_bound),
                    rng.uniform(-h_bound, h_bound)
                ])

                w = rng.uniform(*w_range)
                h = rng.uniform(*h_range)

                # 2️⃣ 遍历所有 grid（包括 unknown）
                for idx in list(self.grid.keys()):

                    p = self.grid_to_world(idx)

                    if abs(p[0] - c[0]) <= w/2 and abs(p[1] - c[1]) <= h/2:
                        self.grid[idx] = 1.0

    def init_T_corridor(
        self,
        center=(0.0, 0.0),
        w_vert=2.0,
        h_vert=10.0,
        w_horiz=8.0,
        h_horiz=2.0,
        wall_thickness=0.6,
        bound=15.0,
        wall_mode="both"   # "both", "left", "right", "none"
    ):

        """
        初始化一个 T 型走廊环境：
        - 走廊内部：已知自由 (0.0)
        - 走廊墙体：已知障碍 (1.0)
        - 外部区域：未知 (0.5)
        """

        cx, cy = center
        bound_cells = int(np.ceil(bound / self.resolution))

        self.known.clear()
        self.unknown.clear()

        def in_vertical_corridor(p):
            return (
                abs(p[0] - cx) <= w_vert * 0.5 and
                abs(p[1] - cy) <= h_vert * 0.5
            )

        def in_horizontal_corridor(p):
            return (
                abs(p[0] - cx) <= w_horiz * 0.5 and
                abs(p[1] - (cy + h_vert * 0.5 - h_horiz * 0.5)) <= h_horiz * 0.5
            )

        def in_corridor(p):
            return in_vertical_corridor(p) or in_horizontal_corridor(p)

        def in_wall(p):
            # 是否在竖直走廊高度范围内
            in_y_range = abs(p[1] - cy) <= h_vert * 0.5

            # 左右墙区域
            in_left_wall = (
                cx - w_vert * 0.5 - wall_thickness <= p[0] < cx - w_vert * 0.5 and
                in_y_range
            )

            in_right_wall = (
                cx + w_vert * 0.5 < p[0] <= cx + w_vert * 0.5 + wall_thickness and
                in_y_range
            )

            if wall_mode == "both":
                return in_left_wall or in_right_wall
            elif wall_mode == "left":
                return in_left_wall
            elif wall_mode == "right":
                return in_right_wall
            elif wall_mode == "none":
                return False
            else:
                raise ValueError(f"Unknown wall_mode: {wall_mode}")


        for ix in range(-bound_cells, bound_cells):
            for iy in range(-bound_cells, bound_cells):

                idx = (ix, iy)
                p = self.grid_to_world(idx)

                if in_corridor(p):
                    self.grid[idx] = 0.0
                    self.known.append(idx)

                elif in_wall(p):
                    self.grid[idx] = 1.0
                    self.known.append(idx)

                else:
                    self.grid[idx] = 0.5
                    self.unknown.append(idx)

    def init_dense_maze(self, K=3, cell_size=3.0, wall_thickness=0.5, seed=42):
        """
        在 Map2D 中生成稠密迷宫（Kruskal生成完美迷宫）
        - 自动填充 self.grid、self.known、self.unknown
        """
        np.random.seed(seed)
        M = 2*K + 1
        maze = np.ones((M, M), dtype=int)
        for i in range(K):
            for j in range(K):
                maze[2*i+1, 2*j+1] = 0  # 通道中心

        # 并查集
        ftr = np.arange(K*K)
        def findftr(x):
            if ftr[x] != x:
                ftr[x] = findftr(ftr[x])
            return ftr[x]

        # 边列表
        edges = []
        for i in range(K):
            for j in range(K):
                if j < K-1: edges.append((i,j,0))  # 右边
                if i < K-1: edges.append((i,j,1))  # 下边
        np.random.shuffle(edges)

        for i,j,flag in edges:
            xy = i*K + j
            nxy = xy + (1 if flag==0 else K)
            f1,f2 = findftr(xy), findftr(nxy)
            if f1 != f2:
                ftr[f1] = f2
                maze[2*i+1+flag, 2*j+2-flag] = 0

        # ----------------------
        # 填充 Map2D 网格
        # ----------------------
        self.known.clear()
        self.unknown.clear()

        grid_cell_count = int(np.ceil(cell_size / self.resolution))
        wall_cell_count = int(np.ceil(wall_thickness / self.resolution))
        grid_size = M*grid_cell_count

        offset = grid_size // 2
        for ix in range(M):
            for iy in range(M):
                val = float(maze[ix, iy])
                gx_start = ix*grid_cell_count - wall_cell_count//2
                gx_end   = ix*grid_cell_count + grid_cell_count + wall_cell_count//2
                gy_start = iy*grid_cell_count - wall_cell_count//2
                gy_end   = iy*grid_cell_count + grid_cell_count + wall_cell_count//2
                gx_start = max(0, gx_start)
                gy_start = max(0, gy_start)
                gx_end = min(grid_size, gx_end)
                gy_end = min(grid_size, gy_end)

                for gx in range(gx_start, gx_end):
                    for gy in range(gy_start, gy_end):
                        idx = (gx - offset, gy - offset)  # ← 平移到中心
                        self.grid[idx] = val
                        if val == 0.0 or val == 1.0:
                            if idx not in self.known:
                                self.known.append(idx)
                        else:
                            if idx not in self.unknown:
                                self.unknown.append(idx)



    def  add_continuous_unknown(self, centers=[(0.0,0.0)], radius=2.0):
        """
        在已知迷宫上添加连续未知区域（世界坐标输入）
        centers: list of (x, y) in world coordinates [m]
        radius: 控制未知区域半径 [m]
        """
        radius_cells = int(np.ceil(radius / self.resolution))  # 转换为栅格半径
        new_unknown = []

        for idx in self.known[:]:
            x_idx, y_idx = idx
            p = self.grid_to_world(idx)  # 当前格子世界坐标
            for cx, cy in centers:
                if np.hypot(p[0] - cx, p[1] - cy) <= radius:
                    self.grid[idx] = 0.5
                    if idx in self.known:
                        self.known.remove(idx)
                    new_unknown.append(idx)
                    break  # 已经变为未知，不用检测其他中心

        self.unknown.extend(new_unknown)