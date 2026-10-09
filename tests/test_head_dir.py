import torch

from intervention.hooks import HeadDirSteer, applied

H, DH = 4, 8


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"num_attention_heads": H, "hidden_size": H * DH, "head_dim": DH})()
        attn = torch.nn.Module()
        attn.o_proj = torch.nn.Linear(H * DH, H * DH, bias=False)
        torch.nn.init.eye_(attn.o_proj.weight)
        layer = torch.nn.Module()
        layer.self_attn = attn
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([layer])


def run(z, iv):
    model = Tiny()
    with torch.no_grad(), applied(model, iv):
        return model.model.layers[0].self_attn.o_proj(z.reshape(1, 1, -1)).view(H, DH)


def means():
    g = torch.Generator().manual_seed(0)
    return {0: torch.randn(H, DH, generator=g)}, {0: torch.randn(H, DH, generator=g)}


def test_own_reflects_across_midpoint():
    es, en = means()
    out = run(es[0], HeadDirSteer({0: [1, 2]}, -2.0, es, en))
    assert torch.allclose(out[1:3], en[0][1:3], atol=1e-5)
    assert torch.allclose(out[[0, 3]], es[0][[0, 3]])


def test_pull_lands_on_target_from_anywhere():
    es, en = means()
    z = torch.randn(H, DH, generator=torch.Generator().manual_seed(1))
    out = run(z, HeadDirSteer({0: [1]}, 1.0, es, en, pull=True))
    u = (es[0][1] - en[0][1]) / (es[0][1] - en[0][1]).norm()
    mid = (es[0][1] + en[0][1]) / 2
    assert torch.allclose((out[1] - mid) @ u, (es[0][1] - mid) @ u, atol=1e-5)
    assert torch.allclose(out[[0, 2, 3]], z[[0, 2, 3]])
