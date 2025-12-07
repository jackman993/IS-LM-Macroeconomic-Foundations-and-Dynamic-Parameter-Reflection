import torch
from torch.utils.data import TensorDataset

def generate_islm_data(
    n_samples: int = 500,
    # IS 方程參數
    C0: float = 50.0,   # 自發消費
    I0: float = 40.0,   # 自發投資
    c: float = 0.6,     # 邊際消費傾向 (0<c<1)
    b: float = 2.0,     # 投資對利率敏感度 (>0)
    # LM 方程參數
    k: float = 0.5,     # 貨幣需求對 Y 的敏感度 (>0)
    h: float = 1.0,     # 貨幣需求對 r 的敏感度 (>0)
    # 政策／環境變數取樣範圍
    G_range=(50.0, 300.0),
    T_range=(50.0, 150.0),
    M_range=(0.5, 3.0),
    P_range=(0.5, 2.0),
):
    """
    依照教科書 IS-LM 模型，對每一組 (G,T,M,P) 解出均衡 (Y*, r*)，
    並回傳可丟進 DataLoader 的 TensorDataset。

    IS:  Y = C0 + c*(Y - T) + I0 - b*r + G
    LM:  M/P = k*Y - h*r
    """

    # 1. 隨機產生政策／環境變數 (G, T, M, P)
    G = torch.empty(n_samples).uniform_(*G_range)
    T = torch.empty(n_samples).uniform_(*T_range)
    M = torch.empty(n_samples).uniform_(*M_range)
    P = torch.empty(n_samples).uniform_(*P_range)

    # 2. 把它們疊成輸入矩陣 x
    x = torch.stack([G, T, M, P], dim=1)   # shape: [N, 4]

    # 3. 解析解 2×2 線性方程組，求 (Y*, r*)

    # 先把常數展開成張量方便計算
    C0_t = torch.full_like(G, C0)
    I0_t = torch.full_like(G, I0)

    # IS 變形：
    # (1 - c) Y + b r = C0 - c T + I0 + G  ≡ A
    one_minus_c = 1.0 - c
    A = C0_t - c * T + I0_t + G

    # LM: k Y - h r = M/P
    # 解這組聯立方程得到：
    #   r* = [kA/(1-c) - M/P] / [k b/(1-c) + h]
    #   Y* = (A - b r*) / (1-c)

    denom_r = k * b / one_minus_c + h          # 分母：常數
    r_star = (k * A / one_minus_c - M / P) / denom_r
    Y_star = (A - b * r_star) / one_minus_c

    y = torch.stack([Y_star, r_star], dim=1)   # shape: [N, 2]

    return TensorDataset(x, y)


# 保留原函式名，讓 train.py 不用改
def generate_mock_data(n_samples: int = 500):
    return generate_islm_data(n_samples=n_samples)
