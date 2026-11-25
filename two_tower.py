"""A small two-tower model in numpy.

Member tower: features of the member's watch history -> hidden layer -> vector.
Title tower: catalog features -> hidden layer -> vector, plus a small vector of its own per title.
Score: dot product of the two vectors plus one bias per title. Loss: softmax over the titles the
member has not watched yet.
"""
import numpy as np


def init_params(n_member, n_title_features, n_titles, dim, hidden, log_pop, seed):
    rng = np.random.default_rng(seed)

    def weights(rows, cols):
        return rng.normal(0, 1 / np.sqrt(rows), (rows, cols))

    return {
        "mw1": weights(n_member, hidden), "mb1": np.zeros(hidden), "mw2": weights(hidden, dim),
        "tw1": weights(n_title_features, hidden), "tb1": np.zeros(hidden), "tw2": weights(hidden, dim),
        "own": rng.normal(0, 0.05, (n_titles, dim)), "bias": log_pop.copy(),
    }


def forward(p, members, titles):
    mh = np.tanh(members @ p["mw1"] + p["mb1"])
    th = np.tanh(titles @ p["tw1"] + p["tb1"])
    mv, tv = mh @ p["mw2"], th @ p["tw2"] + p["own"]
    return mv @ tv.T + p["bias"], (mh, th, mv, tv)


def mask_seen(scores, seen):
    return np.where(seen, -np.inf, scores)


def loss_and_grads(p, members, titles, seen, answer):
    scores, (mh, th, mv, tv) = forward(p, members, titles)
    scores = mask_seen(scores, seen)
    scores = scores - scores.max(axis=1, keepdims=True)
    prob = np.exp(scores)
    prob /= prob.sum(axis=1, keepdims=True)
    rows, n = np.arange(len(answer)), len(answer)
    loss = -np.log(prob[rows, answer] + 1e-12).mean()

    if not np.isfinite(loss) or not np.isfinite(prob).all():
        raise FloatingPointError("the model is no longer finite")
    d = prob.copy()
    d[rows, answer] -= 1
    d /= n
    dmv, dtv = d @ tv, d.T @ mv
    dmh = (dmv @ p["mw2"].T) * (1 - mh ** 2)
    dth = (dtv @ p["tw2"].T) * (1 - th ** 2)
    return loss, {
        "mw1": members.T @ dmh, "mb1": dmh.sum(0), "mw2": mh.T @ dmv,
        "tw1": titles.T @ dth, "tb1": dth.sum(0), "tw2": th.T @ dtv,
        "own": dtv, "bias": d.sum(0),
    }


def train(p, data, titles, epochs, lr, decay, seed, batch=128, on_epoch=None):
    """Adam with weight decay. The bias decays towards its starting value, log popularity."""
    rng = np.random.default_rng(seed + 1000)
    start = p["bias"].copy()
    m = {k: np.zeros_like(v) for k, v in p.items()}
    s = {k: np.zeros_like(v) for k, v in p.items()}
    step = 0
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(data["answer"]))
        for i in range(0, len(order), batch):
            b = order[i:i + batch]
            _, grads = loss_and_grads(p, data["members"][b], titles, data["seen"][b], data["answer"][b])
            step += 1
            for k, g in grads.items():
                m[k] = 0.9 * m[k] + 0.1 * g
                s[k] = 0.999 * s[k] + 0.001 * g * g
                p[k] -= lr * (m[k] / (1 - 0.9 ** step)) / (np.sqrt(s[k] / (1 - 0.999 ** step)) + 1e-8)
                p[k] -= lr * decay * (p[k] - start if k == "bias" else p[k])
        if on_epoch:
            on_epoch(epoch, p)
    return p


def top_k(scores, seen, k):
    """Indices of the k best unseen titles per row. Equal scores: the lower index first."""
    return np.argsort(-mask_seen(scores, seen), axis=1, kind="stable")[:, :k]
