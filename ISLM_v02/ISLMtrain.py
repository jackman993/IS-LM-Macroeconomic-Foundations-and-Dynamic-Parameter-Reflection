# train.py
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from ISLMMode import ISLMModel
from ISLMdata_generator import generate_mock_data

# 輸出目錄
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("🧠 IS-LM Neural Network Training")
    print("=" * 60)
    
    # ---- 1. 基本設定 ----
    pl.seed_everything(42)

    N_SAMPLES   = 2000   # 可以調大一點
    BATCH_SIZE  = 64
    MAX_EPOCHS  = 50
    LEARNING_LR = 1e-3
    VAL_RATIO   = 0.2    # 20% 做驗證

    # ---- 2. 產生資料 (G,T,M,P) → (Y*, r*) ----
    print("\n📊 生成訓練數據...")
    dataset = generate_mock_data(n_samples=N_SAMPLES)
    
    # ---- 輸出訓練數據到 CSV ----
    x_data = dataset.tensors[0].numpy()  # (G, T, M, P)
    y_data = dataset.tensors[1].numpy()  # (Y*, r*)
    
    train_df = pd.DataFrame({
        'G': x_data[:, 0],
        'T': x_data[:, 1],
        'M': x_data[:, 2],
        'P': x_data[:, 3],
        'Y_star': y_data[:, 0],
        'r_star': y_data[:, 1]
    })
    
    csv_path = os.path.join(OUTPUT_DIR, "training_data.csv")
    train_df.to_csv(csv_path, index=False)
    print(f"   ✅ 訓練數據已儲存: {csv_path}")
    print(f"   📋 樣本數: {N_SAMPLES}")
    print(f"   📋 輸入: G, T, M, P")
    print(f"   📋 輸出: Y*, r*")

    n_val = int(len(dataset) * VAL_RATIO)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # ---- 3. 建 IS-LM 近似模型 ----
    model = ISLMModel(lr=LEARNING_LR)

    # ---- 4. Lightning Trainer 設定 ----
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",    # 有 GPU 就用，沒有就 CPU
        devices=1,
        log_every_n_steps=10
    )

    # ---- 5. 開始訓練 ----
    trainer.fit(model, train_loader, val_loader)

    # ---- 6. 存模型 ----
    ckpt_path = "islm_neurocore_model.ckpt"
    trainer.save_checkpoint(ckpt_path)
    print(f"\n✅ Model checkpoint saved to: {ckpt_path}\n")

    # ---- 7. 印出權重，看看近似到什麼樣子 ----
    if hasattr(model, "pretty_print_weights"):
        model.pretty_print_weights()


if __name__ == "__main__":
    main()
