import torch
from torch import nn
from torch.nn import functional as F

class SmallModel(nn.Module):
    """Small baselines, not reimplementations of named papers."""
    def __init__(self, cfg):
        super().__init__()
        self.task = cfg["task"]
        c, w = cfg["channels"], cfg.get("width", 16)
        self.aux = None
        if self.task == "spectra":
            self.encoder = nn.Sequential(nn.Conv1d(c,w,7,padding=3), nn.ReLU(),
                nn.Conv1d(w,2*w,5,padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(8), nn.Flatten())
            self.heads = nn.ModuleDict({k: nn.Linear(16*w,n) for k,n in cfg["heads"].items()})
        else:
            self.encoder = nn.Sequential(nn.Conv2d(c,w,3,padding=1), nn.ReLU(),
                nn.Conv2d(w,w,3,padding=1), nn.ReLU())
            self.down = nn.Sequential(nn.MaxPool2d(2),nn.Conv2d(w,2*w,3,padding=1),nn.ReLU())
            if self.task == "segmentation":
                self.decoder = nn.Sequential(nn.Conv2d(3*w,w,3,padding=1),nn.ReLU())
                self.heads = nn.ModuleDict({k:nn.Conv2d(w,n,1) for k,n in cfg["heads"].items()})
                if cfg.get("aux_weight",0): self.aux = nn.Conv2d(w,1,1)
            else:
                self.heads = nn.ModuleDict({k:nn.Linear(2*w,n) for k,n in cfg["heads"].items()})
    def forward(self, x, head):
        z = self.encoder(x)
        if self.task != "spectra":
            low = self.down(z)
            if self.task == "segmentation":
                up = F.interpolate(low, size=z.shape[-2:], mode="bilinear", align_corners=False)
                z = self.decoder(torch.cat((z,up),dim=1))
            else: z = low.mean(dim=(-2,-1))
        return self.heads[head](z), self.aux(z) if self.aux is not None else None
