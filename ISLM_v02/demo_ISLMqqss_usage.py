# demo_qqss_usage.py

import numpy as np
import sys
import os

# 添加 ISLMqqss 子資料夾到路徑
sys.path.append(os.path.join(os.path.dirname(__file__), 'ISLMqqss'))

from ISLMqqss_module import QQSS4Channel
from simulate_islm_equilibrium import solve_islm_equilibrium

def main():
    print("=" * 60)
    print("🪄 QQSS 四通道 + IS-LM 均衡 Demo")
    print("=" * 60)
    
    qqss = QQSS4Channel()

    # 設一組「名義」政策路徑（很粗）
    T_steps = 20
    G_path = np.full(T_steps, 150.0)
    Ttax_path = np.full(T_steps, 100.0)
    M_path = np.full(T_steps, 1.5)
    P_path = np.full(T_steps, 1.0)

    # 給一個簡單 shock 序列：前面平穩，中間 stress test，後面回歸
    shocks = np.zeros(T_steps)
    shocks[8] = -10.0   # 大型負面衝擊（stress test）
    shocks[9] = +10.0   # 大型正面衝擊

    Y_list = []
    r_list = []

    print("\n📊 時間序列模擬：")
    print("-" * 60)
    
    for t in range(T_steps):
        shock_t = shocks[t]

        # 1) QQSS 更新張力狀態
        z_t = qqss.step(shock_t)

        # 2) 用張力狀態修正原始政策 → 有效政策
        G_eff, T_eff, M_eff, P_eff = qqss.effective_policy(
            G_path[t], Ttax_path[t], M_path[t], P_path[t]
        )

        # 3) 丟進 IS-LM 解析老師，得到當期均衡
        Y_t, r_t = solve_islm_equilibrium(G_eff, T_eff, M_eff, P_eff)

        print(
            f"t={t:02d} | shock={shock_t:+6.2f} | "
            f"z_sum={z_t.sum():+.3f} | "
            f"G_eff={G_eff:7.2f} | Y={Y_t:7.2f} | r={r_t:6.2f}"
        )

        Y_list.append(Y_t)
        r_list.append(r_t)
    
    print("-" * 60)
    print(f"\n📈 Y 範圍: {min(Y_list):.2f} ~ {max(Y_list):.2f}")
    print(f"📈 r 範圍: {min(r_list):.2f} ~ {max(r_list):.2f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
