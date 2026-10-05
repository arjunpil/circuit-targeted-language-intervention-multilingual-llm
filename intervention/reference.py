KL = {
    "Qwen2.5-1.5B": {"zh": (10.15, 4.52, -3.2), "es": (7.25, 4.90, -2.8), "ru": (8.70, 5.40, -2.2), "hi": (9.60, 7.02, 5.0)},
    "Llama-3.2-1B": {"zh": (9.05, 4.77, -5.0), "es": (6.52, 3.48, -3.7), "ru": (7.77, 3.49, -3.5), "hi": (8.06, 6.13, -3.6)},
}
GEN = {"Llama-3.2-1B": {"zh": (0.64, 0.03), "es": (0.62, 0.23), "ru": (0.59, 0.02), "hi": (0.81, 0.04)}}


def coef(model_name, lang):
    return KL.get(model_name.split("/")[-1], {}).get(lang, (None, None, None))[2]


def last_two(model):
    L = model.config.num_hidden_layers
    return [L - 2, L - 1]
