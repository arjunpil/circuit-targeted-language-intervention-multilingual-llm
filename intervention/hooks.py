from contextlib import contextmanager

import torch


def head_dim(cfg):
    return getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads


def head_contributions(o_proj, z, n_heads, dh):
    W = o_proj.weight.view(o_proj.out_features, n_heads, dh)
    return torch.einsum("bthk,dhk->bthd", z.view(*z.shape[:2], n_heads, dh), W)


def _coef(coef, layer):
    return coef[layer] if isinstance(coef, dict) else coef


class ResidualSteer:
    def __init__(self, layers, coef, means, dirs):
        self.layers, self.coef, self.means, self.dirs = layers, coef, means, dirs

    def attach(self, model):
        w = model.model.embed_tokens.weight
        handles = []
        for l in self.layers:
            mu, v, c = self.means[l].to(w), self.dirs[l].to(w), _coef(self.coef, l)

            def fn(module, inputs, output, mu=mu, v=v, c=c):
                h = output[0] if isinstance(output, tuple) else output
                h = h + c * ((h - mu) @ v)[..., None] * v
                return (h, *output[1:]) if isinstance(output, tuple) else h

            handles.append(model.model.layers[l].register_forward_hook(fn))
        return handles


class GatedResidualSteer(ResidualSteer):
    def __init__(self, layers, coef, means, dirs, n_source=20):
        super().__init__(layers, coef, means, dirs)
        self.n_source = n_source

    def attach(self, model):
        w = model.model.embed_tokens.weight
        ref = {}
        handles = []
        for l in self.layers:
            mu, v, c = self.means[l].to(w), self.dirs[l].to(w), _coef(self.coef, l)

            def fn(module, inputs, output, l=l, mu=mu, v=v, c=c):
                h = output[0] if isinstance(output, tuple) else output
                p = (h - mu) @ v
                if l not in ref:
                    if h.shape[0] != 1:
                        raise ValueError("gated steering takes one prompt at a time")
                    ref[l] = p[:, :self.n_source].mean()
                    gate = torch.ones_like(p)
                else:
                    gate = (p * ref[l] > 0).to(p.dtype)
                h = h + c * (p * gate)[..., None] * v
                return (h, *output[1:]) if isinstance(output, tuple) else h

            handles.append(model.model.layers[l].register_forward_hook(fn))
        return handles


class HeadSteer:
    def __init__(self, heads, coef, dirs):
        self.heads, self.coef, self.dirs = heads, coef, dirs

    def attach(self, model):
        H, DH = model.config.num_attention_heads, head_dim(model.config)
        handles = []
        for l, hs in self.heads.items():
            o = model.model.layers[l].self_attn.o_proj
            v = self.dirs[l].to(o.weight)
            u = torch.zeros(H * DH, device=v.device, dtype=v.dtype)
            for h in hs:
                u[h * DH:(h + 1) * DH] = o.weight[:, h * DH:(h + 1) * DH].T @ v
            c = _coef(self.coef, l)

            def fn(module, inputs, output, u=u, v=v, c=c):
                return output + c * (inputs[0] @ u)[..., None] * v

            handles.append(o.register_forward_hook(fn))
        return handles


class HeadDirSteer:
    def __init__(self, heads, coef, means_a, means_b):
        self.heads, self.coef, self.means_a, self.means_b = heads, coef, means_a, means_b

    def attach(self, model):
        H, DH = model.config.num_attention_heads, head_dim(model.config)
        handles = []
        for l, hs in self.heads.items():
            o = model.model.layers[l].self_attn.o_proj
            U = torch.zeros(H, DH, device=o.weight.device, dtype=o.weight.dtype)
            M = torch.zeros_like(U)
            for h in hs:
                a, b = self.means_a[l][h].to(U), self.means_b[l][h].to(U)
                U[h], M[h] = (a - b) / (a - b).norm(), (a + b) / 2
            c = _coef(self.coef, l)

            def fn(module, inputs, U=U, M=M, c=c):
                z = inputs[0].reshape(*inputs[0].shape[:-1], *U.shape)
                p = ((z - M) * U).sum(-1, keepdim=True)
                return ((z + c * p * U).reshape(inputs[0].shape), *inputs[1:])

            handles.append(o.register_forward_pre_hook(fn))
        return handles


class HeadScale:
    def __init__(self, heads, alpha):
        self.heads, self.alpha = heads, alpha

    def attach(self, model):
        DH = head_dim(model.config)
        handles = []
        for l, hs in self.heads.items():
            o = model.model.layers[l].self_attn.o_proj
            scale = torch.ones(o.in_features, device=o.weight.device, dtype=o.weight.dtype)
            for h in hs:
                scale[h * DH:(h + 1) * DH] = _coef(self.alpha, l)

            def fn(module, inputs, scale=scale):
                return (inputs[0] * scale, *inputs[1:])

            handles.append(o.register_forward_pre_hook(fn))
        return handles


@contextmanager
def applied(model, *interventions):
    handles = []
    try:
        for iv in interventions:
            handles += iv.attach(model)
        yield
    finally:
        for h in handles:
            h.remove()
