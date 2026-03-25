import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders import MapEncoder,GNet


class DynamicsModel(nn.Module):
    def __init__(self, dt=0.1, v_max=2.0):
        super().__init__()
        self.dt = dt
        self.v_max = v_max

    def forward(self, x, u):

        # print("==== DYNAMICS INPUT ====")

        # print("x:", x.detach().cpu().numpy())
        # print("u:", u.detach().cpu().numpy())

        px, py, vx, vy = torch.split(x, 1, dim=-1)
        ax, ay = torch.split(u, 1, dim=-1)
        
        vx_next = vx + ax * self.dt
        vy_next = vy + ay * self.dt

        v = torch.cat([vx_next, vy_next], dim=-1)

        # print("v before clip:", v.detach().cpu().numpy())

        v = torch.tanh(v / self.v_max) * self.v_max

        # print("v after clip:", v.detach().cpu().numpy())

        vx_next, vy_next = torch.split(v, 1, dim=-1)

        px_next = px + vx_next * self.dt
        py_next = py + vy_next * self.dt

        # print("px_next:", px_next.detach().cpu().numpy())
        # print("py_next:", py_next.detach().cpu().numpy())
        # print("========================")

        return torch.cat([px_next, py_next, vx_next, vy_next], dim=-1)


    
class PolicyNet(nn.Module):
    def __init__(self, state_dim, map_feat_dim, goal_dim , hidden_dim=256, GNet_output_dim=1, control_dim=2):
        super().__init__()
        self.max_acc = 1.0  # 最大加速度
        self.mlp = nn.Sequential(
            nn.Linear(state_dim + map_feat_dim + goal_dim + GNet_output_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),

            nn.Linear(64, control_dim)  # 输出控制
        )


    def forward(self, x, map_feat, goal, G):
        inp = torch.cat([x, map_feat, goal, G], dim=-1)
        u = self.mlp(inp)

        # ⭐ 限幅
        u = torch.tanh(u) * self.max_acc

        return u





class GlobalPlannerModel(nn.Module):
    def __init__(self, state_dim, map_dim, goal_dim, hidden_dim):
        super().__init__()

        self.map_encoder = MapEncoder(map_dim)


        self.g_net = GNet(state_dim, self.map_encoder.get_map_feature_dim(), goal_dim)
        self.policy_net = PolicyNet(state_dim, self.map_encoder.get_map_feature_dim(), goal_dim)
        self.dynamics = DynamicsModel()

        # loss 权重（可以调）
        self.lambda_terminal = 0.0
        self.lambda_bellman = 0
        self.lambda_sup = 1.0


    # =========================
    # forward：只做 rollout
    # =========================
    def forward(self, map, x0, goal, H=20):

        map_feature = self.map_encoder(map)

        x_k = x0
        x_list = [x_k]   # ⭐ 初始状态先放进去
        G_list = []
        l_list = []

        for k in range(H):

            G_k = self.g_net(x_k, map_feature, goal)
            u = self.policy_net(x_k, map_feature, goal, G_k)

            x_next = self.dynamics(x_k, u)

            p_k = x_k[..., :2]
            p_next = x_next[..., :2]
            l_k = torch.norm(p_next - p_k, dim=-1, keepdim=True)

            G_list.append(G_k)
            l_list.append(l_k)

            x_k = x_next
            x_list.append(x_k)   # ⭐ append next

        # x_list:  [x0, x1, ..., xH]
        # G_list:  [G0, G1, ..., G_{H-1}]
        # l_list:  [l0, l1, ..., l_{H-1}]

        return x_list, G_list, l_list


    def compute_loss(self, x_list, G_list, l_list, distance_field=None):

        H = len(G_list)

        # ---------- Terminal Loss ----------
        G_T = G_list[-1]
        loss_terminal = (G_T ** 2).mean()

        # ---------- Bellman Loss ----------
        # loss_bellman = 0
        # for k in range(H - 1):
        #     G_k = G_list[k]
        #     G_next = G_list[k + 1]
        #     l_k = l_list[k]

        #     target = l_k + G_next.detach()  # ⚠️ 必须 detach
        #     loss_bellman += F.mse_loss(G_k, target)

        # loss_bellman /= (H - 1)

        # ----------  Supervision ----------
        loss_sup = 0
        if distance_field is not None:
            for k in range(H):
                x_k = x_list[k]
                G_k = G_list[k]
                
                G_gt = distance_field.query_tensor(x_k)

                loss_sup += F.mse_loss(G_k, G_gt)

            loss_sup /= H

        # ---------- Total ----------
        total_loss = (
            self.lambda_terminal * loss_terminal
            # + self.lambda_bellman * loss_bellman
            + self.lambda_sup * loss_sup
        )

        return total_loss, {
            "terminal": loss_terminal.item(),
            # "bellman": loss_bellman.item(),
            "sup": loss_sup.item() if distance_field is not None else 0
        }
    



def test_model():


   test_gbmodel = GlobalPlannerModel(state_dim=4, map_dim=128, goal_dim=4, hidden_dim=256)





if __name__ == "__main__":
    test_model()