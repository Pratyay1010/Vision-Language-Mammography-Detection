import torch
import torch.nn as nn


class MetaCoOp(nn.Module):
    
    def __init__(self, ctx_len=4, dim=256, hidden_dim=None):
        super().__init__()
        self.ctx_len = ctx_len
        self.dim = dim
        hidden_dim = hidden_dim or dim
        
        self.static_ctx = nn.Parameter(torch.randn(ctx_len, dim))
        self.meta_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ctx_len * dim)
        )

    def forward(self, img_feat):
        """Generate image-specific context vectors.
        
        Args:
            img_feat: Image features of shape (B, dim)
        Returns:
            Context vectors of shape (B, ctx_len, dim)
        """
        batch_size = img_feat.shape[0]
        dyn_ctx = self.meta_net(img_feat).view(batch_size, self.ctx_len, self.dim)
        return self.static_ctx.unsqueeze(0) + dyn_ctx
    
    def get_context_tokens(self, as_string=False):
        """Return placeholder tokens for prompt construction."""
        if as_string:
            return " ".join(["X"] * self.ctx_len)
        return self.static_ctx