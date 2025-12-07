# simulate_islm_equilibrium.py

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非互動模式，直接存檔

from ISLMMode import ISLMModel

# 輸出目錄
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------- 1. 解析 IS-LM 均衡解 ----------

def solve_islm_equilibrium(
    G: float,
    T: float,
    M: float,
    P: float,
    C0: float = 50.0,
    I0: float = 40.0,
    c: float = 0.6,
    b: float = 2.0,
    k: float = 0.5,
    h: float = 1.0,
):
    """
    給一組 (G,T,M,P) 以及 IS-LM 參數，回傳解析均衡解 (Y*, r*).

    IS:  Y = C0 + c*(Y - T) + I0 - b*r + G
    LM:  M/P = k*Y - h*r
    """
    one_minus_c = 1.0 - c
    A = C0 - c * T + I0 + G
    denom_r = k * b / one_minus_c + h

    r_star = (k * A / one_minus_c - M / P) / denom_r
    Y_star = (A - b * r_star) / one_minus_c

    return Y_star, r_star


# ---------- 2. 主程式 ----------

def main():
    print("=" * 60)
    print("📊 IS-LM Equilibrium Simulation")
    print("=" * 60)
    
    # 參數設定
    G = 150.0
    T = 100.0
    M = 1.5
    P = 1.0
    C0 = 50.0
    I0 = 40.0
    c = 0.6
    b = 2.0
    k = 0.5
    h = 1.0

    # 解析均衡點
    Y_star, r_star = solve_islm_equilibrium(G, T, M, P, C0, I0, c, b, k, h)
    print(f"\n[Analytical]  Y* = {Y_star:.4f},  r* = {r_star:.4f}")

    # Y 範圍
    Y_min = max(1e-3, 0.5 * Y_star)
    Y_max = 1.5 * Y_star
    Y_grid = np.linspace(Y_min, Y_max, 200)

    # IS 直線: r = [A - (1-c)*Y] / b
    A = C0 - c * T + I0 + G
    r_is = (A - (1.0 - c) * Y_grid) / b

    # LM 直線: r = (k*Y - M/P) / h
    r_lm = (k * Y_grid - M / P) / h

    # 載入 NN 模型（可選）
    nn_Y, nn_r = None, None
    try:
        model = ISLMModel.load_from_checkpoint("islm_neurocore_model.ckpt")
        model.eval()
        with torch.no_grad():
            x = torch.tensor([[G, T, M, P]], dtype=torch.float32)
            y_hat = model(x)[0]
            nn_Y, nn_r = float(y_hat[0]), float(y_hat[1])
        print(f"[NN approx]   Y_hat = {nn_Y:.4f},  r_hat = {nn_r:.4f}")
    except:
        print("[Warning] 無法載入 NN 模型，只畫解析解。")

    # 畫圖
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(Y_grid, r_is, label="IS Curve", linewidth=2, color='#e74c3c')
    ax.plot(Y_grid, r_lm, label="LM Curve", linewidth=2, color='#3498db')

    ax.scatter([Y_star], [r_star], s=100, marker="o", color='#2ecc71',
               edgecolors='black', linewidths=1.5, zorder=5,
               label=f"Equilibrium\nY*={Y_star:.2f}, r*={r_star:.2f}")

    if nn_Y and nn_r:
        ax.scatter([nn_Y], [nn_r], s=100, marker="X", color='#9b59b6',
                   edgecolors='black', linewidths=1.5, zorder=6,
                   label=f"NN Approx\nŶ={nn_Y:.2f}, r̂={nn_r:.2f}")

    ax.set_xlabel("Real GDP (Y)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Real Interest Rate (r)", fontsize=12, fontweight='bold')
    ax.set_title("IS-LM Equilibrium", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "islm_equilibrium.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n📊 圖表已儲存: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
