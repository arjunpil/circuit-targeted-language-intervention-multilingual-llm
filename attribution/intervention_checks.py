import copy
import json
from contextlib import contextmanager
from pathlib import Path

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm


SEED = 42
DTYPE = torch.float64
TOL = 1e-10
NONTRIVIAL = 1e-3

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)


# Qwen2RMSNorm casts to float32 even in float64
def rmsnorm_native_dtype(self, x):
    variance = x.pow(2).mean(-1, keepdim=True)
    return self.weight * (x * torch.rsqrt(variance + self.variance_epsilon))


Qwen2RMSNorm.forward = rmsnorm_native_dtype


config = Qwen2Config(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    max_position_embeddings=32,
    initializer_range=0.3,   # default init gives near-flat attention
    tie_word_embeddings=False,
    use_cache=False,
    attn_implementation="sdpa",   # eager does softmax in fp32
)

model = Qwen2ForCausalLM(config).to(DTYPE).eval()

# q/k/v biases are zero at init
with torch.no_grad():
    for name, p in model.named_parameters():
        if name.endswith("bias"):
            p.normal_(0.0, 0.1)

D = config.hidden_size
H = config.num_attention_heads
KV = config.num_key_value_heads
DH = config.head_dim
GROUP = H // KV      # query head h uses kv head h // GROUP

input_ids = torch.randint(0, config.vocab_size, (2, 10))
T = input_ids.shape[1]


@contextmanager
def hooks(*specs):
    handles = []
    try:
        for module, fn, pre in specs:
            register = module.register_forward_pre_hook if pre else module.register_forward_hook
            handles.append(register(fn))
        yield
    finally:
        for handle in handles:
            handle.remove()


def logits(m, *specs):
    with torch.no_grad(), hooks(*specs):
        return m(input_ids=input_ids).logits


def head_contributions(o_proj, z):
    # per-head output in residual space, (B, T, H, D)
    W = o_proj.weight.view(D, H, DH)
    return torch.einsum("bthk,dhk->bthd", z.view(*z.shape[:2], H, DH), W)


def capture_z(m):
    store = {}

    def make(layer_idx):
        def fn(module, inputs):
            store[layer_idx] = inputs[0].detach().clone()
        return fn

    specs = [(layer.self_attn.o_proj, make(i), True) for i, layer in enumerate(m.model.layers)]
    logits(m, *specs)
    return store


def capture_attn_out(m, layer_idx, *specs):
    store = {}

    def fn(module, inputs, output):
        store["out"] = output[0].detach().clone()

    logits(m, (m.model.layers[layer_idx].self_attn, fn, False), *specs)
    return store["out"]


def max_diff(a, b):
    return (a - b).abs().max().item()


def position_variation(delta):
    # 0 if the change is the same at every position
    flat = delta.reshape(-1, D)
    spread = (flat - flat.mean(0, keepdim=True)).norm(dim=-1).max()
    scale = flat.norm(dim=-1).mean()
    return (spread / scale).item() if scale > 0 else 0.0


def unit(n):
    v = torch.randn(n, dtype=DTYPE)
    return v / v.norm()


clean = logits(model)
layers = model.model.layers
results = {}


# 1. decomposition
z_clean = capture_z(model)
decomp_err = max(
    max_diff(head_contributions(layer.self_attn.o_proj, z_clean[i]).sum(2),
             layer.self_attn.o_proj(z_clean[i]))
    for i, layer in enumerate(layers)
)
results["decomposition"] = {
    "max_abs_diff": decomp_err,
    "pass": decomp_err < TOL,
}


# 2. head-output steering vs residual steering
def head_steer(heads, v, pos_mask):
    def fn(module, inputs, output):
        c = head_contributions(module, inputs[0])
        for h in heads:
            c[:, :, h] = c[:, :, h] + pos_mask[None, :, None] * v
        return c.sum(2)
    return fn


def resid_steer(vec, pos_mask):
    def fn(module, inputs, output):
        return (output[0] + pos_mask[None, :, None] * vec,) + tuple(output[1:])
    return fn


selected = {0: [1], 1: [0, 2]}   # heads 0 and 2 use different kv heads
v = 2.0 * unit(D)
masks = {
    "all_positions": torch.ones(T, dtype=DTYPE),
    "last_position": torch.nn.functional.one_hot(torch.tensor(T - 1), T).to(DTYPE),
}

results["head_steering"] = {}
for mask_name, mask in masks.items():
    by_head = logits(model, *[(layers[l].self_attn.o_proj, head_steer(hs, v, mask), False)
                              for l, hs in selected.items()])
    by_resid = logits(model, *[(layers[l].self_attn, resid_steer(len(hs) * v, mask), False)
                               for l, hs in selected.items()])
    diff = max_diff(by_head, by_resid)
    moved = max_diff(by_head, clean)
    results["head_steering"][mask_name] = {
        "max_abs_diff_head_vs_residual": diff,
        "logit_change_vs_clean": moved,
        "pass": diff < TOL and moved > NONTRIVIAL,
    }


# 3. v_proj bias edit
bias_layer, bias_group = 0, 1
delta = 0.5 * torch.randn(DH, dtype=DTYPE)

bias_edited = copy.deepcopy(model)
with torch.no_grad():
    bias_edited.model.layers[bias_layer].self_attn.v_proj.bias[bias_group * DH:(bias_group + 1) * DH] += delta

W_O = layers[bias_layer].self_attn.o_proj.weight
group_heads = [h for h in range(H) if h // GROUP == bias_group]
u = sum(W_O[:, h * DH:(h + 1) * DH] @ delta for h in group_heads)

by_edit = logits(bias_edited)
by_resid = logits(model, (layers[bias_layer].self_attn, resid_steer(u, masks["all_positions"]), False))
diff = max_diff(by_edit, by_resid)
moved = max_diff(by_edit, clean)
results["value_bias_edit"] = {
    "query_heads_reached": group_heads,
    "max_abs_diff_edit_vs_constant_residual": diff,
    "logit_change_vs_clean": moved,
    "pass": diff < TOL and moved > NONTRIVIAL,
}


# 4. W_O projection edit
proj_layer, proj_head = 1, 2
v_hat = unit(D)
P = torch.eye(D, dtype=DTYPE) - torch.outer(v_hat, v_hat)
cols = slice(proj_head * DH, (proj_head + 1) * DH)

edited = copy.deepcopy(model)
with torch.no_grad():
    o = edited.model.layers[proj_layer].self_attn.o_proj
    o.weight[:, cols] = P @ o.weight[:, cols]

c_clean = head_contributions(layers[proj_layer].self_attn.o_proj, z_clean[proj_layer])
z_edit = capture_z(edited)
c_edit = head_contributions(edited.model.layers[proj_layer].self_attn.o_proj, z_edit[proj_layer])

other = [h for h in range(H) if h != proj_head]
results["projection_edit"] = {
    "v_component_before": (c_clean[:, :, proj_head] @ v_hat).abs().max().item(),
    "v_component_after": (c_edit[:, :, proj_head] @ v_hat).abs().max().item(),
    "max_abs_diff_other_heads": max_diff(c_edit[:, :, other], c_clean[:, :, other]),
    "max_abs_diff_vs_projected_clean": max_diff(c_edit[:, :, proj_head], c_clean[:, :, proj_head] @ P),
}
r = results["projection_edit"]
r["pass"] = (r["v_component_before"] > NONTRIVIAL and r["v_component_after"] < TOL
             and r["max_abs_diff_other_heads"] < TOL and r["max_abs_diff_vs_projected_clean"] < TOL)


# 5. head scaling
scale_layer, scale_head = 0, 1
cols = slice(scale_head * DH, (scale_head + 1) * DH)


def scale_z(alpha):
    def fn(module, inputs):
        z = inputs[0].clone()
        z[..., cols] = alpha * z[..., cols]
        return (z,) + tuple(inputs[1:])
    return fn


o_scale = layers[scale_layer].self_attn.o_proj
zeroed = copy.deepcopy(model)
with torch.no_grad():
    zeroed.model.layers[scale_layer].self_attn.o_proj.weight[:, cols] = 0.0

attn_clean = capture_attn_out(model, scale_layer)
attn_doubled = capture_attn_out(model, scale_layer, (o_scale, scale_z(2.0), True))
c_head = head_contributions(o_scale, z_clean[scale_layer])[:, :, scale_head]

results["head_scaling"] = {
    "alpha1_vs_clean": max_diff(logits(model, (o_scale, scale_z(1.0), True)), clean),
    "alpha0_vs_zeroed_W_O": max_diff(logits(model, (o_scale, scale_z(0.0), True)), logits(zeroed)),
    "alpha2_extra_vs_head_contribution": max_diff(attn_doubled - attn_clean, c_head),
}
r = results["head_scaling"]
r["pass"] = all(x < TOL for x in r.values())


# does the residual change depend on the input?
steer_layer = 0
attn_clean_0 = capture_attn_out(model, steer_layer)
attn_steered_0 = capture_attn_out(
    model, steer_layer,
    (layers[steer_layer].self_attn.o_proj, head_steer(selected[steer_layer], v, masks["all_positions"]), False))

variation = {
    "head_steering": position_variation(attn_steered_0 - attn_clean_0),
    "value_bias_edit": position_variation(capture_attn_out(bias_edited, bias_layer) - attn_clean_0),
    "projection_edit": position_variation(c_edit[:, :, proj_head] - c_clean[:, :, proj_head]),
    "head_scaling": position_variation(attn_doubled - attn_clean),
}
results["input_dependence"] = {
    "position_variation": variation,
    "pass": (variation["head_steering"] < TOL and variation["value_bias_edit"] < TOL
             and variation["projection_edit"] > 0.1 and variation["head_scaling"] > 0.1),
}


def check_pass(entry):
    if "pass" in entry:
        return entry["pass"]
    return all(check_pass(sub) for sub in entry.values())


print("\n=== Phase 0: intervention checks (tiny Qwen2, GQA 4q/2kv, float64) ===")
for name, entry in results.items():
    print(f"\n[{'PASS' if check_pass(entry) else 'FAIL'}] {name}")
    print(json.dumps(entry, indent=2))

passed = all(check_pass(entry) for entry in results.values())
print(f"\nPASS: {passed}")

out = {
    "seed": SEED,
    "dtype": str(DTYPE),
    "tolerance": TOL,
    "model": {"architecture": "tiny random Qwen2", "layers": config.num_hidden_layers,
              "query_heads": H, "kv_heads": KV, "head_dim": DH, "hidden_size": D},
    "results": results,
    "pass": passed,
}
with open(RESULTS_DIR / "intervention_checks.json", "w") as f:
    json.dump(out, f, indent=2)

if not passed:
    raise SystemExit("Phase 0 intervention checks FAILED.")
print("\nSaved results to phase0/results/intervention_checks.json")
