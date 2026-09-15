"""实验作业二：ConvLSTM 的算法应用与改进 —— 完整实验脚本。

功能：
1. 弹跳小球序列数据集生成（可复现，随机种子 42）；
2. 基线 ConvLSTM + 6 个对照实验 + 最优组合，逐一训练并记录
   （参数量 / 测试 MSE / 测试 MAE / 耗时 / 逐 epoch 曲线）；
3. 每个实验保存一张"训练损失 + 测试 MSE"曲线图
   （result_baseline.png / result_exp1.png ... result_exp6.png）；
4. 最优组合在种子 42/43/44 下 3 次复测（填表 6）；
5. 生成"输入帧 | 真实帧 | 预测帧"对比图（result_pred.png）；
6. 结果逐实验增量写入 results_summary.json（中途中断也不丢）。

运行（CPU 亦可，规模较小）：
    python convLSTM_experiments.py
"""

import json
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")  # 无 GUI 环境也可保存图片
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

torch.manual_seed(42)
np.random.seed(42)

N_TRAIN = 1000
N_TEST = 200
N_TOTAL = N_TRAIN + N_TEST
T_FRAMES = 12   # 每段序列总帧数，足够支撑"看前 8 帧预测第 9 帧"


# ---------------------------------------------------------------- 数据合成 ---
def make_sequences(n_seq=N_TOTAL, T=T_FRAMES, size=32, r=2, seed=42):
    """生成弹跳小球序列，返回 (n_seq, T, size, size, 1) 二值图像。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    seqs = np.zeros((n_seq, T, size, size), np.float32)
    for s in range(n_seq):
        x, y = rng.uniform(4, size - 5, 2)
        vx, vy = rng.choice([-1, 1], 2) * rng.uniform(0.8, 1.6, 2)
        for t in range(T):
            x, y = x + vx, y + vy
            if x < r or x > size - r:
                vx = -vx
            if y < r or y > size - r:
                vy = -vy
            seqs[s, t] = ((xx - x) ** 2 + (yy - y) ** 2 <= r * r)
    return seqs[..., None]


def make_dataset(k_in=4):
    """切分数据：前 k_in 帧作输入，第 k_in 帧作目标。"""
    data = make_sequences()
    train = data[:N_TRAIN]
    test = data[N_TRAIN:]
    train_x = torch.tensor(train[:, :k_in]).permute(0, 1, 4, 2, 3)
    train_y = torch.tensor(train[:, k_in]).squeeze(-1)
    test_x = torch.tensor(test[:, :k_in]).permute(0, 1, 4, 2, 3)
    test_y = torch.tensor(test[:, k_in]).squeeze(-1)
    return train_x, train_y, test_x, test_y


# ---------------------------------------------------------------- 模型 -------
class ConvLSTMCell(nn.Module):
    """单层 ConvLSTM 细胞：卷积一次同时计算 4 个门（i/f/g/o）。"""

    def __init__(self, in_ch, hid_ch, k=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, k, padding=k // 2)
        self.hid = hid_ch

    def forward(self, x, state):
        h, c = state
        z = self.conv(torch.cat([x, h], dim=1))
        i, f, g, o = z.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        c_new = f * c + i * torch.tanh(g)
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class ConvLSTM(nn.Module):
    def __init__(self, in_ch=1, hid=32, k=3, layers=1):
        super().__init__()
        chs = [in_ch] + [hid] * layers
        self.cells = nn.ModuleList(
            [ConvLSTMCell(chs[i], chs[i + 1], k) for i in range(layers)])
        self.out = nn.Conv2d(hid, 1, 3, padding=1)

    def forward(self, x):   # x: (B, T, C, H, W)
        b, _, _, hsize, wsize = x.shape
        states = [(torch.zeros(b, c.hid, hsize, wsize),
                   torch.zeros(b, c.hid, hsize, wsize))
                  for c in self.cells]
        for t in range(x.shape[1]):
            xt = x[:, t]
            for j, cell in enumerate(self.cells):
                states[j] = cell(xt, states[j])
                xt = states[j][0]
        return self.out(states[-1][0]).squeeze(1)


class FCLSTM(nn.Module):
    """结构对照组：把每帧 32×32 展平为 1024 维，用全连接 LSTM 建模。"""

    def __init__(self, in_size=32 * 32, hid=32):
        super().__init__()
        self.lstm = nn.LSTM(in_size, hid, batch_first=True)
        self.head = nn.Linear(hid, in_size)
        self.size = 32

    def forward(self, x):   # x: (B, T, 1, 32, 32)
        b, t, c, h, w = x.shape
        x = x.reshape(b, t, c * h * w)            # (B, T, 1024)
        out, _ = self.lstm(x)                     # (B, T, hid)
        pred = self.head(out[:, -1])              # 取最后一帧隐状态重建
        return pred.reshape(b, h, w)              # (B, 32, 32)


# ---------------------------------------------------------------- 训练 -------
def train_eval(model, train_x, train_y, test_x, test_y,
               loss_fn=None, lr=0.001, epochs=5, bs=64, seed=42, verbose=True):
    """训练并逐 epoch 记录损失与测试指标，返回历史与最终结果。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    loss_fn = loss_fn or nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = train_x.shape[0]

    train_losses, test_mses, test_maes = [], [], []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(train_x[idx]), train_y[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(test_x)
            mse = (pred - test_y).pow(2).mean().item()
            mae = (pred - test_y).abs().mean().item()
            tr = loss.item()

        train_losses.append(tr)
        test_mses.append(mse)
        test_maes.append(mae)
        if verbose:
            print(f"    epoch {ep + 1}/{epochs}  train_loss {tr:.5f}  "
                  f"test_MSE {mse:.5f}  test_MAE {mae:.5f}", flush=True)

    dt = time.time() - t0
    with torch.no_grad():
        final_pred = model(test_x)
    return {
        "train_losses": train_losses,
        "test_mses": test_mses,
        "test_maes": test_maes,
        "mse": test_mses[-1],
        "mae": test_maes[-1],
        "time": dt,
        "pred": final_pred,
    }


def plot_curve(hist, title, fname):
    """保存"训练损失 + 测试 MSE"双面板曲线图。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(hist["train_losses"], color="tab:blue", marker="o", ms=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("训练损失")
    axes[0].set_title("训练损失曲线")
    axes[0].grid(alpha=0.3)

    axes[1].plot(hist["test_mses"], color="tab:red", marker="s", ms=4)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("测试 MSE")
    axes[1].set_title("测试 MSE 曲线")
    axes[1].grid(alpha=0.3)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  -> 已保存 {fname}", flush=True)


def make_loss(name):
    return {"mse": nn.MSELoss(), "l1": nn.L1Loss()}[name]


def num_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------- 实验编排 ----
def main():
    summary = {}

    def save_summary():
        with open("results_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def record(key, params, hist):
        summary[key] = dict(params=params, mse=hist["mse"],
                            mae=hist["mae"], time=hist["time"],
                            test_mses=hist["test_mses"])
        save_summary()
        print(f"    [记录] 参数量 {params}  MSE {hist['mse']:.5f}  "
              f"MAE {hist['mae']:.5f}  耗时 {hist['time']:.1f}s", flush=True)

    # 共享数据集
    tr4, ty4, te4, tey4 = make_dataset(k_in=4)
    tr8, ty8, te8, tey8 = make_dataset(k_in=8)

    # ---------- 组 0 基线 ----------
    print("=" * 70, flush=True)
    print("[组0] 基线：ConvLSTM(1,32,k=3,1层) 看4帧 MSE", flush=True)
    m = ConvLSTM()
    h = train_eval(m, tr4, ty4, te4, tey4, loss_fn=make_loss("mse"))
    plot_curve(h, "图1  基线训练损失与测试 MSE", "result_baseline.png")
    record("0_baseline", num_params(m), h)

    # ---------- 组 1 加深层数 ----------
    print("=" * 70, flush=True)
    print("[组1] 加深层数：ConvLSTM 2 层", flush=True)
    m = ConvLSTM(layers=2)
    h = train_eval(m, tr4, ty4, te4, tey4, loss_fn=make_loss("mse"))
    plot_curve(h, "图2  实验1：加深层数（2 层）", "result_exp1.png")
    record("1_layers2", num_params(m), h)

    # ---------- 组 2 隐藏通道 ----------
    print("=" * 70, flush=True)
    print("[组2] 隐藏通道 32 -> 64", flush=True)
    m = ConvLSTM(hid=64)
    h = train_eval(m, tr4, ty4, te4, tey4, loss_fn=make_loss("mse"))
    plot_curve(h, "图3  实验2：隐藏通道 64", "result_exp2.png")
    record("2_hid64", num_params(m), h)

    # ---------- 组 3 卷积核 ----------
    print("=" * 70, flush=True)
    print("[组3] 卷积核 K=3 -> 5", flush=True)
    m = ConvLSTM(k=5)
    h = train_eval(m, tr4, ty4, te4, tey4, loss_fn=make_loss("mse"))
    plot_curve(h, "图4  实验3：卷积核 5×5", "result_exp3.png")
    record("3_k5", num_params(m), h)

    # ---------- 组 4 输入帧数 ----------
    print("=" * 70, flush=True)
    print("[组4] 输入帧数 4 -> 8", flush=True)
    m = ConvLSTM()
    h = train_eval(m, tr8, ty8, te8, tey8, loss_fn=make_loss("mse"))
    plot_curve(h, "图5  实验4：输入帧数 8", "result_exp4.png")
    record("4_k8", num_params(m), h)

    # ---------- 组 5 结构对照 ----------
    print("=" * 70, flush=True)
    print("[组5] 结构对照：全连接 LSTM", flush=True)
    m = FCLSTM()
    h = train_eval(m, tr4, ty4, te4, tey4, loss_fn=make_loss("mse"))
    plot_curve(h, "图6  实验5：全连接 LSTM 对照", "result_exp5.png")
    record("5_fclstm", num_params(m), h)

    # ---------- 组 6 损失函数 ----------
    print("=" * 70, flush=True)
    print("[组6] 损失函数 MSE -> L1", flush=True)
    m = ConvLSTM()
    h = train_eval(m, tr4, ty4, te4, tey4, loss_fn=make_loss("l1"))
    plot_curve(h, "图7  实验6：L1 损失", "result_exp6.png")
    record("6_l1", num_params(m), h)

    # ---------- 组 7 最优组合 ----------
    print("=" * 70, flush=True)
    print("[组7] 最优组合：隐藏 64 + 卷积核 5 + 看 8 帧（MSE）", flush=True)
    m = ConvLSTM(hid=64, k=5)
    h = train_eval(m, tr8, ty8, te8, tey8, loss_fn=make_loss("mse"))
    record("7_optimal", num_params(m), h)
    opt_pred = h["pred"]

    # ---------- 表 6：最优组合 3 次复测 ----------
    print("=" * 70, flush=True)
    print("[表6] 最优组合 3 次复测（种子 42 / 43 / 44）", flush=True)
    retest = []
    for s in (42, 43, 44):
        m = ConvLSTM(hid=64, k=5)
        h = train_eval(m, tr8, ty8, te8, tey8, loss_fn=make_loss("mse"), seed=s)
        retest.append(dict(seed=s, mse=h["mse"], mae=h["mae"], time=h["time"]))
        print(f"    种子 {s}: MSE {h['mse']:.5f}  MAE {h['mae']:.5f}  "
              f"耗时 {h['time']:.1f}s", flush=True)
    summary["retest"] = retest
    summary["retest_avg"] = dict(
        mse=float(np.mean([r["mse"] for r in retest])),
        mae=float(np.mean([r["mae"] for r in retest])),
        time=float(np.mean([r["time"] for r in retest])),
    )
    save_summary()

    # ---------- 图 8：预测帧对比 ----------
    print("=" * 70, flush=True)
    print("[图8] 生成 输入帧|真实帧|预测帧 对比图", flush=True)
    per_sample = ((opt_pred - tey8) ** 2).reshape(N_TEST, -1).mean(dim=1).numpy()
    order = np.argsort(per_sample)
    samples = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]

    fig, axes = plt.subplots(3, 6, figsize=(12, 6.5))
    labels = [f"输入帧 t={t}" for t in range(4)] + ["真实帧", "预测帧"]
    for row, sidx in enumerate(samples):
        for col in range(4):
            axes[row, col].imshow(te8[sidx, col, 0].numpy(), cmap="gray_r")
        axes[row, 4].imshow(tey8[sidx].numpy(), cmap="gray_r")
        axes[row, 5].imshow(opt_pred[sidx].numpy(), cmap="gray_r")
        for col in range(6):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        axes[row, 0].set_ylabel(f"样本 {sidx}\nMSE={per_sample[sidx]:.4f}",
                                fontsize=8)
    for col in range(6):
        axes[0, col].set_title(labels[col], fontsize=9)
    fig.suptitle("图8  输入帧 | 真实帧 | 预测帧 对比（最优组合模型）", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("result_pred.png", dpi=150)
    plt.close(fig)
    print("  -> 已保存 result_pred.png（易/中/难样本）", flush=True)

    print("=" * 70, flush=True)
    print("全部实验完成，结果已写入 results_summary.json", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
