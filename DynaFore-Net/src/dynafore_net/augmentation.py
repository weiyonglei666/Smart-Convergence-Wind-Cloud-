"""Train-time feature-aware augmentation and virtual fine-phase reconstruction."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

class FeatureAwareAugment(nn.Module):
    """特征感知 AutoDA 的任务适配：三层策略、幅值/时间/频率操作、联合梯度。

    候选操作保留功率标签的一致性；为复现正式实验保留原通道掩码。
    为该任务实现的可微端口，并非未经修改运行上游 AutoDA 全部操作。
    """
    def __init__(self, c, stats, fine=True, power_scale=20.0):
        super().__init__()
        self.fine = fine
        self.power_scale = float(power_scale)
        self.policies = nn.ModuleList([nn.Sequential(nn.Linear(c*5, 64), nn.GELU(), nn.Linear(64, 12)) for _ in range(3)])
        mask = torch.zeros(c)
        mask[:11] = 1
        mask[15:18] = 1
        self.register_buffer("mask", mask)
        self.register_buffer("xm", torch.as_tensor(stats["x_mean"]))
        self.register_buffer("xs", torch.as_tensor(stats["x_std"]))

    @staticmethod
    def fine_reconstruction(x, state):
        # 15 个虚拟相位只用于训练增强，不作为真实新增标签。
        phase = torch.linspace(0, 1, 16, device=x.device, dtype=x.dtype)[None, None, :, None]
        left, right = x[:, :-1, None], x[:, 1:, None]
        noise = .01*state[:, None, None, None]*torch.randn_like(left)
        fine = (1-phase)*left+phase*right+torch.sin(torch.pi*phase)*noise
        # 端点在 fine 中完整保留；中间状态经局部平均形成历史增强视图。
        view = fine[:, :, 1:-1].mean(2)
        return torch.cat([x[:, :1], view[:, 1:], x[:, -1:]], 1)

    def forward(self, x, y):
        regularizer = x.new_zeros(())
        previous = None
        for policy in self.policies:
            feats = torch.cat([x.mean(1), x.std(1, unbiased=False), x.amin(1), x.amax(1), x[:, -1]-x[:, -4]], -1)
            logits, strengths = policy(feats).chunk(2, -1)
            prob = logits.softmax(-1)
            choice = F.gumbel_softmax(logits, tau=1., hard=True)
            strength = strengths.sigmoid()
            views, labels = [x], [y]
            # 功率幅值改变时，目标作完全一致的容量归一化变换。
            a = 1+.08*(strength[:, 1]-.5)
            views.append(x*a[:, None, None])
            views[-1] = views[-1].clone()
            views[-1][..., 0] = (a[:, None]*(x[..., 0]*self.xs[0]+self.xm[0])-self.xm[0])/self.xs[0]
            labels.append((y*a[:, None]).clamp(0, 1))
            shift = .02*(strength[:, 2]-.5)
            shifted = x.clone()
            shifted[..., 0] = shifted[..., 0]+self.power_scale*shift[:, None]/self.xs[0]
            views.append(shifted)
            labels.append((y+shift[:, None]).clamp(0, 1))
            views.append(x+.02*strength[:, 3, None, None]*torch.randn_like(x))
            labels.append(y)
            smooth = F.avg_pool1d(F.pad(x.transpose(1, 2), (1, 1), mode="replicate"), 3, 1).transpose(1, 2)
            views.append(x+.15*strength[:, 4, None, None]*(smooth-x))
            labels.append(y)
            spec = torch.fft.rfft(x.float(), dim=1)
            frequency = torch.linspace(0, 1, spec.shape[1], device=x.device)[None, :, None]
            filtered = torch.fft.irfft(spec*(1-.15*strength[:, 5, None, None]*frequency), n=x.shape[1], dim=1).to(x.dtype)
            views.append(filtered)
            labels.append(y)
            mixed = (torch.stack(views, 1)*choice[:, :, None, None]).sum(1)
            # 保留正式实验的通道掩码与非功率端点；掩码的范围见实现说明。
            xnew = x+(mixed-x)*self.mask
            x = torch.cat([x[:, :1], xnew[:, 1:-1], x[:, -1:]], 1)
            # 幅值变换也应覆盖功率端点，与 y 保持一致。
            power = mixed[..., 0]
            x = torch.cat([power[..., None], x[..., 1:]], -1)
            y = (torch.stack(labels, 1)*choice[:, :, None]).sum(1)
            entropy = -(prob*(prob+1e-8).log()).sum(-1).mean()
            regularizer = regularizer-.001*entropy
            if previous is not None:
                regularizer = regularizer+.001*F.mse_loss(prob, previous)
            previous = prob
        if self.fine:
            state = x[..., 0].diff(dim=1).abs().mean(1).clamp(0, 1)
            fine = self.fine_reconstruction(x, state)
            x = x+(fine-x)*self.mask
        return x, y, regularizer
