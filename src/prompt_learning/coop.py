import torch
import torch.nn as nn


class CoOp(nn.Module):
    
    def __init__(self, ctx_len=4, dim=256):
        super().__init__()
        self.ctx = nn.Parameter(torch.randn(ctx_len, dim))
        self.ctx_len = ctx_len

    def forward(self):
        return self.ctx
    
    def get_context_tokens(self, as_string=False):
        """Return placeholder tokens for prompt construction."""
        if as_string:
            return " ".join(["X"] * self.ctx_len)
        return self.ctx