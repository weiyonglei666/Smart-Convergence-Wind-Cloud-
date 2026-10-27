"""DynaFore-Net core extracted from the formal 16-step experiment."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .attention import PMDEncoderLayerPVA, PMDEncoderLayerTRA


class AdaptiveStatistics(nn.Module):
    """内部 token 统计与外部最近运行统计的样本级可学习融合。"""
    def __init__(self, c, d, use_gate=True, use_adain=True, window=72):
        super().__init__()
        self.use_gate, self.use_adain = use_gate, use_adain
        self.window = int(window)
        self.external = nn.Sequential(nn.Linear(c*2, d*2), nn.GELU(), nn.Linear(d*2, d*2))
        self.gate = nn.Sequential(nn.Linear(d*4, d), nn.GELU(), nn.Linear(d, d*2), nn.Sigmoid())

    def forward(self, z, x):
        mu = z.mean(1, keepdim=True)
        sd = z.std(1, keepdim=True, unbiased=False).clamp_min(1e-5)
        recent = x[:, -self.window:]
        ext = self.external(torch.cat([recent.mean(1), recent.std(1, unbiased=False)], -1))
        em, es = ext.chunk(2, -1)
        em, es = em[:, None], F.softplus(es[:, None])+1e-5
        gate = self.gate(torch.cat([mu, sd, em, es], -1)) if self.use_gate else mu.new_full((*mu.shape[:-1], mu.shape[-1]*2), .5)
        gm, gs = gate.chunk(2, -1)
        tm, ts = gm*mu+(1-gm)*em, gs*sd+(1-gs)*es
        # 去掉 AdaIN 时保留融合统计的加性注入，避免与去掉整个 NSAN 重复。
        return (z-mu)/sd*ts+tm if self.use_adain else z+(tm-mu)


class AdaptiveSpectrum(nn.Module):
    def __init__(self, patches, d):
        super().__init__()
        self.logits = nn.Parameter(torch.full((patches//2+1, d), 2.))
        self.state = nn.Linear(d, d)
    def forward(self, z):
        dtype = z.dtype
        spec = torch.fft.rfft(z.float(), dim=1, norm="ortho")
        gate = torch.sigmoid(self.logits[None] + self.state(z.mean(1)).float()[:, None])
        return torch.fft.irfft(spec*gate, n=z.shape[1], dim=1, norm="ortho").to(dtype)


class DynaForeNet(nn.Module):
    def __init__(self, c, f, lookback, horizon, stats, cfg, flags=None):
        super().__init__()
        if horizon != 16:
            raise ValueError("This release supports the 16-step forecasting task only")
        if lookback < 32 or lookback % 16:
            raise ValueError("lookback must be a multiple of 16 and at least 32")
        self.power_scale = float(cfg.get("capacity_mw", 20.0))
        if self.power_scale <= 0:
            raise ValueError("capacity_mw must be positive")
        self.flags = flags or {}
        self.energy_type = self.flags.get("energy_type", "wind")
        self.c, self.horizon, self.patch = c, horizon, 16
        d = cfg["d_model"]
        n_heads = int(cfg.get("n_heads", 4))
        d_ff = int(cfg.get("d_ff", 4 * d))
        nsan_window = int(cfg.get("nsan_window", 72))
        if nsan_window > lookback:
            # Debug runs may use a shorter history; the production contract is
            # 72 steps for both the 288-step wind and 96-step PV windows.
            nsan_window = lookback
        n = lookback//self.patch
        self.n = n
        self.embedding = nn.Linear(self.patch, d)
        self.position = nn.Parameter(torch.randn(1, n, d)*.01)
        self.var_position = nn.Parameter(torch.randn(1, c, 1, d)*.01)
        self.pva = PMDEncoderLayerPVA(d, n_heads, d_ff=d_ff, dropout=cfg["dropout"])
        self.tra = nn.ModuleList([PMDEncoderLayerTRA(d, n_heads, d_ff=d_ff, dropout=cfg["dropout"]) for _ in range(cfg["layers"])])
        self.statistics = AdaptiveStatistics(c, d, self.on("style_gate"), self.on("adain"), window=nsan_window)
        self.spectrum = AdaptiveSpectrum(n, d)
        self.multiscale = nn.ModuleList([nn.Linear(d, d) for _ in range(3)])
        self.scale_fuse = nn.Sequential(nn.Linear(d*3, d), nn.GELU(), nn.LayerNorm(d))
        self.future = nn.Sequential(nn.Linear(f, d), nn.GELU(), nn.Linear(d, d))
        self.horizon_pos = nn.Parameter(torch.randn(1, horizon, d)*.01)
        self.cross = nn.MultiheadAttention(d, n_heads, dropout=cfg["dropout"], batch_first=True)
        self.stats_proj = nn.Sequential(nn.Linear(4, d), nn.GELU())
        self.head = nn.Sequential(nn.Linear(d*3, d), nn.GELU(), nn.Dropout(cfg["dropout"]), nn.Linear(d, 1))
        self.mechanism = nn.Sequential(nn.Linear(d+7, d), nn.GELU(), nn.Linear(d, 2))
        self.pv_mechanism = nn.Sequential(nn.Linear(d+7, d), nn.GELU(), nn.Linear(d, 2))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        nn.init.zeros_(self.mechanism[-1].weight)
        nn.init.zeros_(self.mechanism[-1].bias)
        nn.init.zeros_(self.pv_mechanism[-1].weight)
        nn.init.zeros_(self.pv_mechanism[-1].bias)
        self.register_buffer("xm", torch.as_tensor(stats["x_mean"]))
        self.register_buffer("xs", torch.as_tensor(stats["x_std"]))
        self.register_buffer("fm", torch.as_tensor(stats["f_mean"]))
        self.register_buffer("fs", torch.as_tensor(stats["f_std"]))
        self.future_names = cfg["future_columns"]

    def on(self, name):
        return self.flags.get(name, True)

    def _future_index(self, *names):
        """Resolve a named future channel while accepting legacy checkpoints."""
        for name in names:
            if name in self.future_names:
                return self.future_names.index(name)
        for suffix in names:
            for index, name in enumerate(self.future_names):
                if name.endswith(suffix):
                    return index
        raise KeyError(f"future condition is missing {names}; got {self.future_names}")

    def history(self, x):
        b = len(x)
        patches = x.transpose(1, 2).unfold(-1, self.patch, self.patch)
        means = patches.mean(-1, keepdim=True)
        # 没有局部标准差缩放：保留真实的局部振幅。
        shape = patches-means if self.on("pmd") else patches
        z = self.embedding(shape)+self.position[:, None]+self.var_position
        if self.on("pva"):
            last, _ = self.pva(z[:, :, -1])
            z = torch.cat([z[:, :, :-1], last[:, :, None]], 2)
        z = z.reshape(b*self.c, self.n, -1)
        m = means.reshape(b*self.c, self.n, 1)
        for layer in self.tra:
            z, _ = layer(z, m if self.on("tra") and self.on("pmd") else torch.zeros_like(m))
        z = z.reshape(b, self.c, self.n, -1).mean(1)
        if self.on("nsan"):
            z = self.statistics(z, x)
        if self.on("asb"):
            z = self.spectrum(z)
        if self.on("multiscale"):
            parts = []
            for scale, proj in zip([1, 2, 4], self.multiscale):
                pooled = z if scale == 1 else F.avg_pool1d(z.transpose(1, 2), scale, ceil_mode=True).transpose(1, 2)
                resized = F.interpolate(pooled.transpose(1, 2), size=self.n, mode="linear", align_corners=False).transpose(1, 2)
                parts.append(proj(resized))
            z = z+self.scale_fuse(torch.cat(parts, -1))
        return z

    def forward(self, x, future, return_aux: bool = False):
        if not self.on("nwp"):
            future = torch.zeros_like(future)
        memory = self.history(x)
        query = self.future(future)+self.horizon_pos
        if self.on("cross"):
            context, _ = self.cross(query, memory, memory, need_weights=False)
        else:
            context = memory.mean(1, keepdim=True).expand(-1, self.horizon, -1)
        power = (x[..., 0]*self.xs[0]+self.xm[0])/self.power_scale
        last = power[:, -1]
        ramp = last-power[:, -4]
        s = torch.stack([last, power[:, -16:].mean(1), power[:, -16:].std(1, unbiased=False), ramp], -1)
        state = self.stats_proj(s)[:, None].expand(-1, self.horizon, -1)
        y = last[:, None]+self.head(torch.cat([context, query, state], -1)).squeeze(-1)
        raw_f = future*self.fs+self.fm
        recent = s[:, None].expand(-1, self.horizon, -1)
        if not self.on("ramp_gate"):
            recent = torch.cat([recent[..., :3], torch.zeros_like(recent[..., 3:])], -1)
        daylight = None
        if self.energy_type == "pv":
            # PV-specific correction: use future irradiance (NWP or the
            # explicit measured-future ablation) and an exact daylight mask.
            # The final clamp is part of the output head, so training and
            # inference share the same feasible set.
            ghi_idx = self._future_index("GHI")
            gti_idx = self._future_index("GTI", "DNI")
            irradiance = torch.maximum(raw_f[..., ghi_idx], raw_f[..., gti_idx]).clamp_min(0.0)
            irradiance_norm = (irradiance / 1000.0).clamp(0.0, 1.0)
            daylight = (irradiance > 20.0).to(y.dtype)
            energy = torch.cat([recent, irradiance_norm[..., None], daylight[..., None], irradiance_norm.square()[..., None]], -1)
            if self.on("guided_correction"):
                residual, gate = self.pv_mechanism(torch.cat([context, energy], -1)).unbind(-1)
                amplitude = .03 + .07 * daylight + .03 * irradiance_norm
                y = y + daylight * amplitude * torch.tanh(residual) * torch.sigmoid(gate)
            out = torch.clamp(y, 0.0, 1.0) * daylight
            return (out, {"pre_clamp": y, "daylight": daylight}) if return_aux else out
        # Wind-specific guided residual using recent operating state and
        # future wind components (NWP or the explicit measured-future ablation).
        u = self._future_index("future__wind_u", "nwp__pl925__u")
        v = self._future_index("future__wind_v", "nwp__pl925__v")
        wind = torch.sqrt(raw_f[..., u]**2+raw_f[..., v]**2+1e-6)/25
        high = torch.sigmoid((recent[..., 0]-.7)*10) if self.on("high_gate") else torch.zeros_like(y)
        low = torch.sigmoid((.15-recent[..., 0])*10) if self.on("low_gate") else torch.zeros_like(y)
        ramp_state = ramp[:, None].expand_as(y) if self.on("ramp_gate") else torch.zeros_like(y)
        energy = torch.cat([recent, wind[..., None], high[..., None], low[..., None]], -1)
        if self.on("guided_correction"):
            residual, gate = self.mechanism(torch.cat([context, energy], -1)).unbind(-1)
            amplitude = .03+.07*torch.maximum(high, low)+.05*torch.tanh(ramp_state.abs()*5)
            y = y+amplitude*torch.tanh(residual)*torch.sigmoid(gate)
        out = torch.clamp(y, 0.0, 1.0)
        return (out, {"pre_clamp": y, "daylight": torch.ones_like(y)}) if return_aux else out
