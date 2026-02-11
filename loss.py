import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridLoss(nn.Module):
    def __init__(
        self,
        vad_floor=0.35,        # 放松：最低仍保留 35%
        vad_percentile=0.6,    # 只区分“相对低能量”
        energy_margin=1.4
    ):
        super().__init__()

        self.vad_floor = vad_floor
        self.vad_percentile = vad_percentile
        self.energy_margin = energy_margin

        self.register_buffer("window", torch.hann_window(960).pow(0.5))
        self.register_buffer("window_small", torch.hann_window(240).pow(0.5))

    # ------------------------------------------------------------------
    # 能量排序式 VAD（无阈值、无时间结构）
    # ------------------------------------------------------------------
    def get_energy_weight(self, mag):
        """
        mag: (B, F, T)
        return: (B, 1, T) in [0, 1]
        """
        # 帧能量
        e = mag.mean(dim=1)  # (B, T)
        e = torch.log(e + 1e-8)

        # utterance 内排序
        q = torch.quantile(
            e, self.vad_percentile, dim=1, keepdim=True
        )

        # 相对能量距离
        w = (e - q) / (e.std(dim=1, keepdim=True) + 1e-8)

        # sigmoid 映射，连续、无硬边界
        w = torch.sigmoid(w)

        return w.unsqueeze(1)

    def loudness_loss(self, y_pred, y_true):
        """
        Utterance-level log RMS consistency
        稳定，不受极值影响
        """
        rms_pred = torch.sqrt(torch.mean(y_pred ** 2, dim=-1) + 1e-8)
        rms_true = torch.sqrt(torch.mean(y_true ** 2, dim=-1) + 1e-8)

        log_diff = torch.log(rms_pred + 1e-8) - torch.log(rms_true + 1e-8)
        return torch.mean(log_diff ** 2)

    # ------------------------------------------------------------------
    def forward(self, pred_stft, true_stft):
        pr, pi = pred_stft[..., 0], pred_stft[..., 1]
        tr, ti = true_stft[..., 0], true_stft[..., 1]

        pred_mag = torch.sqrt(pr**2 + pi**2 + 1e-12)
        true_mag = torch.sqrt(tr**2 + ti**2 + 1e-12)

        # ------------------ relaxed energy weight ------------------
        with torch.no_grad():
            e_w = self.get_energy_weight(true_mag)

        vad_w = self.vad_floor + (1.0 - self.vad_floor) * e_w
        vad_w = vad_w.expand_as(true_mag)

        # ------------------ RI loss ------------------
        pr_c = pr / (pred_mag.pow(0.7) + 1e-8)
        pi_c = pi / (pred_mag.pow(0.7) + 1e-8)
        tr_c = tr / (true_mag.pow(0.7) + 1e-8)
        ti_c = ti / (true_mag.pow(0.7) + 1e-8)

        ri_loss = torch.mean(
            vad_w * ((pr_c - tr_c) ** 2 + (pi_c - ti_c) ** 2)
        )

        # ------------------ mag loss ------------------
        mag_loss = torch.mean(
            vad_w * (pred_mag.pow(0.3) - true_mag.pow(0.3)).pow(2)
        )

        # ------------------ energy ceiling（只防泄露） ------------------
        ratio = pred_mag / (true_mag + 1e-4)
        overflow = F.relu(ratio - self.energy_margin)

        energy_ceiling_loss = torch.mean(
            (1.0 - e_w).expand_as(pred_mag) * overflow
        )

        # ------------------ time domain ------------------
        win = self.window.to(pred_mag.device)
        y_pred = torch.istft(pr + 1j * pi, 960, 480, 960, window=win)
        y_true = torch.istft(tr + 1j * ti, 960, 480, 960, window=win)
        
        # ------------------ loudness anchor loss ------------------
        loudness_anchor_loss = self.loudness_loss(y_pred, y_true)
        
        # ------------------ short window transient loss ------------------
        win_s = self.window_small.to(pred_mag.device)

        pred_s = torch.stft(
            y_pred, 240, 120, 240, window=win_s, return_complex=True
        )
        true_s = torch.stft(
            y_true, 240, 120, 240, window=win_s, return_complex=True
        )

        pred_e = torch.abs(pred_s).mean(dim=1)
        true_e = torch.abs(true_s).mean(dim=1)

        short_time_loss = torch.mean(
            F.relu(pred_e - true_e)
        )

        # ------------------ SI-SNR ------------------
        proj = torch.sum(y_true * y_pred, -1, keepdim=True) * y_true / (
            torch.sum(y_true ** 2, -1, keepdim=True) + 1e-8
        )

        sisnr = -torch.log10(
            torch.sum(proj ** 2, -1) /
            (torch.sum((y_pred - proj) ** 2, -1) + 1e-8) + 1e-8
        ).mean()

        total = (
            30  *   ri_loss +
            70  *   mag_loss +
            10  *   loudness_anchor_loss +
            0.4 *   energy_ceiling_loss +
            30  *   short_time_loss +
                    sisnr
        )

        return (
            total,
            ri_loss,
            mag_loss,
            loudness_anchor_loss,
            energy_ceiling_loss,
            short_time_loss,
            sisnr,
        )

# ------------------------------------------------------------------
# Debug
# ------------------------------------------------------------------
if __name__ == "__main__":
    loss_fn = HybridLoss()

    pred = torch.randn(2, 481, 100, 2)
    true = torch.randn(2, 481, 100, 2)

    true[:, :, 40:60, :] *= 1e-4  # 模拟静音

    out = loss_fn(pred, true)
    print("Total:", out[0].item())
    print("Short-time loss:", out[-1].item())