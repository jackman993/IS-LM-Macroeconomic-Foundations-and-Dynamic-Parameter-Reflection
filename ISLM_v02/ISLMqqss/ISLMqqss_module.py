# qqss_module.py
"""
QQSS 四通道超簡化張力模組 v0

概念：
- 有 4 個通道的張力狀態 z = [z1, z2, z3, z4]
- 每一期來一個 shock_s (標量)，映射到四通道 → shock_vec
- 四通道之間有一點點「互相影響」（干涉），再更新 z_t -> z_{t+1}
- 最後用 z_t 去調整原始 (G, T, M, P)，得到 "有效政策" (G_eff, T_eff, M_eff, P_eff)

注意：這是玩具版（超簡化），目的是先有一個可以跑的 QQSS 骨架。
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class QQSSConfig:
    # 張力記憶係數（越接近 1，越有慣性）
    memory: float = 0.8
    # shock 影響強度
    shock_gain: float = 0.3
    # 通道間干涉強度
    interference_gain: float = 0.2
    # z 對政策變數的影響強度（整體縮放）
    # 調大 alpha：0.05 → 1.0（stress test）
    policy_gain: float = 1.0


class QQSS4Channel:
    """
    QQSS 四通道張力狀態機（超簡化版）

    通道直覺：
    z1: 結構 / 長期趨勢張力
    z2: 景氣循環張力
    z3: 政策操作張力
    z4: 金融 / 預期張力
    """

    def __init__(self, config: QQSSConfig | None = None):
        self.config = config or QQSSConfig()
        # 初始張力狀態，全 0
        self.z = np.zeros(4, dtype=float)

        # 固定的「干涉矩陣」：對角為 1，非對角小一點，表示通道間有互相影響
        self.interference_matrix = (
            np.eye(4) + self.config.interference_gain * (np.ones((4, 4)) - np.eye(4))
        )

    def reset(self, z0: np.ndarray | None = None):
        """重設張力狀態。 z0 若為 None 則歸零。"""
        if z0 is None:
            self.z = np.zeros(4, dtype=float)
        else:
            z0 = np.asarray(z0, dtype=float)
            assert z0.shape == (4,)
            self.z = z0.copy()

    def _shock_to_channels(self, shock_s: float) -> np.ndarray:
        """
        把一個標量 shock_s 映射到四個通道。
        這裡先用一個超簡化的固定權重：
        - z1, z3 跟 shock 同向
        - z2, z4 部分反向（表示有些通道會抵消）
        """
        # 你未來可以依照 QQSS 真正定義改這裡
        weights = np.array([+1.0, -0.7, +0.5, -0.3], dtype=float)
        return self.config.shock_gain * shock_s * weights

    def step(self, shock_s: float) -> np.ndarray:
        """
        前進一個時間步：
        - shock_s: 當期的「總體衝擊」標量（可正可負）

        回傳更新後的 z_t（shape: (4,)）。
        """
        # 1) shock 映到四通道
        shock_vec = self._shock_to_channels(shock_s)  # (4,)

        # 2) 通道間干涉：先把目前 z 做一次 mixing
        mixed_z = self.interference_matrix @ self.z  # (4,)

        # 3) 更新規則：新張力 = 記憶 * 舊張力 + 非線性(shock + mixed_z)
        #    用 tanh 當作簡單的「飽和」非線性，避免張力爆炸
        nonlinear_part = np.tanh(mixed_z + shock_vec)
        self.z = self.config.memory * self.z + (1.0 - self.config.memory) * nonlinear_part

        return self.z.copy()

    def effective_policy(self, G: float, T: float, M: float, P: float):
        """
        給定原始政策變數 (G, T, M, P)，
        用目前的張力狀態 z 產生「有效政策」(G_eff, T_eff, M_eff, P_eff)。

        超簡化假設：
        delta = C @ z
        [G_eff, T_eff, M_eff, P_eff] = [G, T, M, P] + policy_gain * delta
        """
        # C 矩陣：把四通道張力映到四個政策變數的調整量
        C = np.array(
            [
                [+1.0, +0.2, +0.3, 0.0],   # 對 G 的調整：四通道不同權重
                [-0.2, +1.0, 0.0, +0.3],   # 對 T 的調整
                [0.0, +0.3, +1.0, +0.2],   # 對 M 的調整
                [+0.3, 0.0, -0.2, +1.0],   # 對 P 的調整
            ],
            dtype=float,
        )

        delta = C @ self.z  # shape: (4,)
        adj = self.config.policy_gain * delta

        G_eff = G + adj[0]
        T_eff = T + adj[1]
        M_eff = M + adj[2]
        P_eff = P + adj[3]

        return G_eff, T_eff, M_eff, P_eff
