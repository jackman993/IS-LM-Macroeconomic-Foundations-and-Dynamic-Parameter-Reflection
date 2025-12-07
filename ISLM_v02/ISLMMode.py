import torch
import pytorch_lightning as pl
from torch import nn

class ISLMModel(pl.LightningModule):
    """
    IS-LM Equilibrium Approximator

    Input : x = [G, T, M, P]
    Output: y_hat = [Y*, r*]

    這裡刻意只用一層 Linear：
    - 讓網路近似教科書 IS-LM 的線性均衡解
    - 權重 matrix 大致對應各政策變數對 (Y, r) 的邊際影響
    """

    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        # 4 維輸入 (G,T,M,P) → 2 維輸出 (Y,r)
        self.linear = nn.Linear(4, 2, bias=True)
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, 4]
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch                         # y = [Y*, r*] from data_generator
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    # 可選：訓練完之後你可以用這個看權重解讀經濟意義
    def pretty_print_weights(self):
        with torch.no_grad():
            W = self.linear.weight.clone().cpu()
            b = self.linear.bias.clone().cpu()
        names_in  = ["G", "T", "M", "P"]
        names_out = ["Y", "r"]
        print("=== IS-LM approx weights ===")
        for j, out_name in enumerate(names_out):
            row = W[j]
            terms = [f"{row[i]:+.4f}*{names_in[i]}" for i in range(4)]
            print(f"{out_name} ≈ " + " ".join(terms) + f"  {b[j]:+.4f}")
