#!/usr/bin/env python3
"""
Stage 3 — every measured result for one task, written to JSON.

Produces, for a single task (triple-riding or helmet):
  * probe ablation      : standardisation on/off, regularisation sweep
  * centralized baseline: the reference the federated model is judged against
  * federated runs      : {3,4,6} clients x {balanced, videoset} partitions,
                          with per-round AP / F1 / weight-drift traces
  * per-client analysis : training alone vs participating in federation
  * precision-recall    : curves for centralized and federated
  * differential privacy: utility as a function of noise multiplier
  * communication cost  : payload per round against a full-network baseline

PROTOCOL. Model, hyper-parameters and decision threshold are chosen using only
train+val, with GroupKFold grouped by source video so no video spans a fold.
The test split is scored once, at the end. Nothing is tuned on it.

UNIT OF EVALUATION. Scores are pooled per unit -- an association for
triple-riding, a rider track for helmet -- because a tracked object contributes
several near-identical crops. Treating those as independent samples overstates
the effective sample size, and a deployed system judges an object once.
"""

import argparse
import json
import os
import time

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

SEED = 42
FED_ROUNDS = 15
LOCAL_ITERS = 5
CLIENT_TEST_FRAC = 0.2
CLIENT_COUNTS = (3, 4, 6)
PARTITION_MODES = ("balanced", "videoset")
DP_SIGMAS = (0.0, 0.001, 0.005, 0.01, 0.05, 0.1)
DP_CLIP = 1.0


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------ io -----
def load(feat_dir, split):
    z = np.load(os.path.join(feat_dir, f"{split}.npz"), allow_pickle=True)
    X = z["X"]
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-8, None)
    return X, z["y"], z["key"].astype(str), z["vid"].astype(str)


def pool(scores, y, key):
    """Mean score per unit; a unit is positive if any of its crops is."""
    u, inv = np.unique(key, return_inverse=True)
    tot = np.zeros(len(u))
    cnt = np.zeros(len(u))
    lab = np.zeros(len(u), np.int64)
    np.add.at(tot, inv, scores)
    np.add.at(cnt, inv, 1)
    np.maximum.at(lab, inv, y)
    return u, tot / np.clip(cnt, 1, None), lab


def thr_balanced(y, s):
    """Threshold maximising min(precision, recall) -- both must clear the bar,
    and at a few-percent positive rate accuracy is not informative."""
    p, r, t = precision_recall_curve(y, s)
    m = np.minimum(p[:-1], r[:-1])
    return (float(t[int(np.argmax(m))]), float(m.max())) if len(m) else (0.0, 0.0)


def prf(y, s, thr):
    pred = (s >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    return dict(precision=P, recall=R,
                f1=2 * P * R / max(P + R, 1e-9), tp=tp, fp=fp, fn=fn, tn=tn)


def pr_points(y, s, n=200):
    p, r, _ = precision_recall_curve(y, s)
    idx = np.linspace(0, len(p) - 1, min(n, len(p))).astype(int)
    return dict(precision=p[idx].tolist(), recall=r[idx].tolist(),
                ap=float(average_precision_score(y, s)),
                baseline=float(np.mean(y)))


# ------------------------------------------------------------ partition ----
def partition_balanced(videos, pos_of, n):
    """Positives are dealt first so no client is starved of the rare class;
    zero-positive videos are dealt separately, because they never raise a
    client's positive count and a single-phase greedy would give them all to
    whichever client happened to be behind."""
    parts = [[] for _ in range(n)]
    load_ = [0] * n
    for v in sorted([v for v in videos if pos_of.get(v, 0) > 0],
                    key=lambda v: -pos_of[v]):
        i = min(range(n), key=lambda j: (load_[j], len(parts[j])))
        parts[i].append(v)
        load_[i] += pos_of[v]
    for v in [v for v in videos if pos_of.get(v, 0) == 0]:
        i = min(range(n), key=lambda j: len(parts[j]))
        parts[i].append(v)
    return parts


def partition_videoset(videos, n):
    """One videoset per client: different cameras and locations, so genuinely
    non-IID and closer to a real deployment than a random split."""
    import re
    groups = {}
    for v in videos:
        m = re.match(r"^vs(\d+)_", v)
        groups.setdefault(int(m.group(1)) if m else 0, []).append(v)
    keys = sorted(groups)
    parts = [[] for _ in range(min(n, len(keys)))]
    for i, k in enumerate(keys):
        parts[i % len(parts)].extend(groups[k])
    return parts


# ----------------------------------------------------------- federation ----
def _seed_model(m, coef, inter):
    m.classes_ = np.array([0, 1])
    m.coef_ = coef.copy()
    m.intercept_ = inter.copy()
    return m


def local_fit(Xs, y, coef, inter, C):
    m = LogisticRegression(C=C, max_iter=LOCAL_ITERS, warm_start=True,
                           class_weight="balanced")
    _seed_model(m, coef, inter)
    m.fit(Xs, y)
    return m.coef_.copy(), m.intercept_.copy(), len(y)


def make_aggregator():
    """Flower's own FedAvg aggregation, driven directly.

    Flower's simulation engine runs clients as Ray actors; Ray is heavy, does
    not always install, and is pointless for models this small. Using the
    strategy object directly keeps Flower's aggregation while dropping that
    dependency. Falls back to the equivalent weighted mean if the import fails.
    """
    try:
        from flwr.common import (Code, FitRes, Status, ndarrays_to_parameters,
                                 parameters_to_ndarrays)
        from flwr.server.strategy import FedAvg
        strat = FedAvg()

        def agg(updates):
            res = []
            for coef, inter, n in updates:
                res.append((None, FitRes(
                    status=Status(code=Code.OK, message=""),
                    parameters=ndarrays_to_parameters([coef, inter]),
                    num_examples=n, metrics={})))
            params, _ = strat.aggregate_fit(1, res, [])
            nd = parameters_to_ndarrays(params)
            return nd[0], nd[1]
        return agg, "flower-fedavg"
    except Exception as e:
        def agg(updates):
            t = sum(n for _, _, n in updates)
            return (sum(c * n for c, _, n in updates) / t,
                    sum(i * n for _, i, n in updates) / t)
        return agg, f"numpy-fedavg (flower unavailable: {type(e).__name__})"


def dp_noise(coef, inter, sigma, rng):
    """Gaussian mechanism: clip the update then add calibrated noise, so one
    client's contribution is bounded before it reaches the server."""
    if sigma <= 0:
        return coef, inter
    v = np.concatenate([coef.ravel(), inter.ravel()])
    norm = np.linalg.norm(v)
    scale = min(1.0, DP_CLIP / max(norm, 1e-12))
    coef, inter = coef * scale, inter * scale
    return (coef + rng.normal(0, sigma * DP_CLIP, coef.shape),
            inter + rng.normal(0, sigma * DP_CLIP, inter.shape))


def run_federated(clients, dim, C, eval_fn, agg, sigma=0.0, rounds=FED_ROUNDS):
    rng = np.random.default_rng(SEED)
    coef, inter = np.zeros((1, dim)), np.zeros(1)
    trace = []
    for rnd in range(1, rounds + 1):
        prev = coef.copy()
        updates = []
        for c in clients:
            uc, ui, n = local_fit(c["Xtr_s"], c["ytr"], coef, inter, C)
            uc, ui = dp_noise(uc, ui, sigma, rng)
            updates.append((uc, ui, n))
        coef, inter = agg(updates)
        m = eval_fn(coef, inter)
        m.update(round=rnd, drift=float(np.linalg.norm(coef - prev)))
        trace.append(m)
    return coef, inter, trace


# ------------------------------------------------------------------ main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pos-name", default="positive")
    a = ap.parse_args()

    R = {"task": a.task, "positive_class": a.pos_name,
         "config": dict(fed_rounds=FED_ROUNDS, local_iters=LOCAL_ITERS,
                        client_test_frac=CLIENT_TEST_FRAC, seed=SEED,
                        dp_clip=DP_CLIP)}

    Xtr, ytr, ktr, vtr = load(a.feats, "train")
    Xva, yva, kva, vva = load(a.feats, "val")
    Xte, yte, kte, vte = load(a.feats, "test")
    X = np.vstack([Xtr, Xva]); y = np.concatenate([ytr, yva])
    k = np.concatenate([ktr, kva]); g = np.concatenate([vtr, vva])
    dim = X.shape[1]

    _, _, yte_u = pool(np.zeros(len(yte)), yte, kte)
    R["data"] = dict(
        dim=dim, pool_crops=int(len(y)), pool_pos=int(y.sum()),
        pool_units=int(len(np.unique(k))), pool_videos=int(len(np.unique(g))),
        test_crops=int(len(yte)), test_pos=int(yte.sum()),
        test_units=int(len(yte_u)), test_pos_units=int(yte_u.sum()),
        test_videos=int(len(np.unique(vte))),
        train_crops=int(len(ytr)), val_crops=int(len(yva)))
    log(f"{a.task}: pool {len(y)} crops / {len(np.unique(k))} units, "
        f"test {len(yte)} crops / {len(yte_u)} units ({int(yte_u.sum())} positive)")

    sc = StandardScaler().fit(X)
    Xs, Xte_s = sc.transform(X), sc.transform(Xte)

    def eval_units(coef, inter, Xmat, yy, kk):
        s = Xmat @ coef.ravel() + inter[0]
        _, sp, yl = pool(s, yy, kk)
        return s, sp, yl

    # ---------------- 1. probe ablation ----------------
    log("ablation: standardisation x regularisation")
    R["ablation"] = []
    for scaled in (False, True):
        A, B = (Xs, Xte_s) if scaled else (X, Xte)
        for C in (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 1.0):
            m = LogisticRegression(C=C, max_iter=3000,
                                   class_weight="balanced").fit(A, y)
            s, sp, yl = eval_units(m.coef_, m.intercept_, B, yte, kte)
            R["ablation"].append(dict(
                scaled=scaled, C=C, model="logreg",
                ap_unit=float(average_precision_score(yl, sp)),
                ap_crop=float(average_precision_score(yte, s))))
    for C in (0.0003, 0.001, 0.01):
        m = CalibratedClassifierCV(
            LinearSVC(C=C, class_weight="balanced", max_iter=5000), cv=3).fit(Xs, y)
        s = m.predict_proba(Xte_s)[:, 1]
        _, sp, yl = pool(s, yte, kte)
        R["ablation"].append(dict(scaled=True, C=C, model="linsvc",
                                  ap_unit=float(average_precision_score(yl, sp)),
                                  ap_crop=float(average_precision_score(yte, s))))

    # ---------------- 2. hyper-parameter selection on train+val only --------
    log("selecting C by GroupKFold (grouped by video)")
    folds = list(GroupKFold(n_splits=5).split(Xs, y, groups=g))
    R["cv_selection"] = []
    best = None
    for C in (0.0003, 0.001, 0.003, 0.01, 0.03):
        oof = np.zeros(len(y))
        for tr_i, te_i in folds:
            m = LogisticRegression(C=C, max_iter=3000,
                                   class_weight="balanced").fit(Xs[tr_i], y[tr_i])
            oof[te_i] = m.decision_function(Xs[te_i])
        _, sp, yl = pool(oof, y, k)
        ap = float(average_precision_score(yl, sp))
        R["cv_selection"].append(dict(C=C, oof_ap_unit=ap))
        log(f"   C={C:<7} out-of-fold unit AP = {ap:.4f}")
        if best is None or ap > best[0]:
            best = (ap, C, oof)
    _, C_BEST, oof_best = best
    R["config"]["probe_C"] = C_BEST
    log(f"   selected C={C_BEST}")

    # threshold from out-of-fold predictions -- never from test
    _, sp_oof, yl_oof = pool(oof_best, y, k)
    THR, oof_bal = thr_balanced(yl_oof, sp_oof)
    R["config"]["threshold"] = THR
    R["config"]["oof_min_pr"] = oof_bal

    # ---------------- 3. centralized reference ----------------
    cen = LogisticRegression(C=C_BEST, max_iter=3000,
                             class_weight="balanced").fit(Xs, y)
    s_c, sp_c, yl_c = eval_units(cen.coef_, cen.intercept_, Xte_s, yte, kte)
    R["centralized"] = dict(
        **prf(yl_c, sp_c, THR),
        ap_unit=float(average_precision_score(yl_c, sp_c)),
        ap_crop=float(average_precision_score(yte, s_c)),
        pr_unit=pr_points(yl_c, sp_c), pr_crop=pr_points(yte, s_c))
    log(f"centralized: AP(unit)={R['centralized']['ap_unit']:.4f} "
        f"F1={R['centralized']['f1']:.4f}")

    # per-crop test predictions, so the qualitative figure can show real
    # successes and failures rather than hand-picked examples
    _tp = np.load(os.path.join(a.feats, "test.npz"), allow_pickle=True)["paths"]
    np.savez(a.out.replace(".json", "_testpred.npz"),
             paths=_tp, y=yte, score=s_c, key=kte, threshold=THR)

    # ---------------- 4. federated ----------------
    agg, agg_name = make_aggregator()
    R["config"]["aggregator"] = agg_name
    log(f"aggregator: {agg_name}")

    pos_units_of = {}
    for v in np.unique(g):
        m = g == v
        pos_units_of[v] = len({kk for kk, yy in zip(k[m], y[m]) if yy == 1})

    R["federated"] = []
    rng = np.random.default_rng(SEED)
    for mode in PARTITION_MODES:
        for n_cl in CLIENT_COUNTS:
            parts = (partition_videoset(list(np.unique(g)), n_cl) if mode == "videoset"
                     else partition_balanced(list(np.unique(g)), pos_units_of, n_cl))
            n_act = len(parts)

            # videoset mode cannot make more clients than there are videosets,
            # so a 6-client request silently becomes 4. Without this guard the
            # results table gets two rows both labelled "videoset / 4 clients"
            # carrying different numbers, and the 6-client entry disappears
            # from the figures.
            if any(e["mode"] == mode and e["n_clients"] == n_act
                   for e in R["federated"]):
                log(f"  {mode:<9} n={n_cl}: only {n_act} groups available "
                    f"-- duplicate of the n={n_act} run, skipping")
                continue

            flat = [v for p in parts for v in p]
            assert len(flat) == len(set(flat)), "LEAK: video on two clients"
            assert not (set(flat) & set(vte)), "LEAK: test video reached a client"

            clients, pinfo = [], []
            for ci, p in enumerate(parts):
                vids = sorted(p)
                rng.shuffle(vids)
                nh = max(1, int(len(vids) * CLIENT_TEST_FRAC))
                lte, ltr = set(vids[:nh]), set(vids[nh:])
                assert not (lte & ltr), "LEAK: video in client's own train and test"
                mtr, mte = np.isin(g, list(ltr)), np.isin(g, list(lte))
                # RAW features, not the centrally standardised Xs. If clients
                # were given Xs the "federated" scaler below would be computed
                # over already-whitened data -- a no-op -- while the test set
                # got scaled from raw, so the model would train in one feature
                # space and be scored in another. That silently manufactures a
                # large fake "cost of federation".
                clients.append(dict(id=ci, Xtr=X[mtr], ytr=y[mtr], ktr=k[mtr],
                                    Xte=X[mte], yte=y[mte], kte=k[mte]))
                pinfo.append(dict(client=ci, videos=len(p), crops=int(mtr.sum()),
                                  positives=int(y[mtr].sum()),
                                  pos_units=sum(pos_units_of[v] for v in p),
                                  local_test_crops=int(mte.sum())))

            # federated standardisation: clients share only per-feature count,
            # sum and sum-of-squares; the server reconstructs the exact global
            # mean/variance. Identical to a central scaler, no raw data shared.
            usable = [c for c in clients if len(np.unique(c["ytr"])) == 2]
            n_tot = sum(c["Xtr"].shape[0] for c in usable)
            s1 = sum(c["Xtr"].sum(0) for c in usable)
            s2 = sum((c["Xtr"] ** 2).sum(0) for c in usable)
            gmu = s1 / n_tot
            gsd = np.sqrt(np.clip(s2 / n_tot - gmu ** 2, 1e-12, None))
            for c in clients:
                c["Xtr_s"] = (c["Xtr"] - gmu) / gsd
                c["Xte_s"] = ((c["Xte"] - gmu) / gsd) if c["Xte"].shape[0] else c["Xte"]
            # test features standardised with the FEDERATED scaler, which is
            # derived from client aggregates and differs slightly from the
            # centralized one (clients hold train+val minus their own holdouts)
            Xte_fed = (Xte - gmu) / gsd

            # Guard the failure this replaced: training and evaluation features
            # must live in the same space, and a genuinely federated scaler must
            # approximately reproduce the centrally fitted one.
            _tr_mu = float(np.mean([c["Xtr_s"].mean() for c in usable]))
            assert abs(_tr_mu) < 0.5 and abs(float(Xte_fed.mean()) - _tr_mu) < 0.5, \
                (f"feature-space mismatch: client train mean {_tr_mu:.3f} vs "
                 f"test mean {float(Xte_fed.mean()):.3f}")
            assert np.allclose(gmu, sc.mean_, atol=5e-2), \
                "federated scaler does not match the centrally fitted one"

            def ev(coef, inter):
                s, sp, yl = eval_units(coef, inter, Xte_fed, yte, kte)
                d = prf(yl, sp, THR)
                d["ap_unit"] = float(average_precision_score(yl, sp))
                d["ap_crop"] = float(average_precision_score(yte, s))
                return d

            fc, fi, trace = run_federated(usable, dim, C_BEST, ev, agg)
            s_f, sp_f, yl_f = eval_units(fc, fi, Xte_fed, yte, kte)

            # The federated arm needs its OWN threshold. Reusing the
            # centralized one compares two models on different score scales and
            # turns any calibration difference into a fake F1 gap. It is derived
            # here from the clients' OWN held-out videos -- data the federated
            # model never trained on -- which is also what a real deployment
            # would have available. The central test set is still untouched.
            _hs, _hy = [], []
            for c in clients:
                if c["Xte"].shape[0] == 0:
                    continue
                _s, _sp, _yl = eval_units(fc, fi, c["Xte_s"], c["yte"], c["kte"])
                _hs.append(_sp)
                _hy.append(_yl)
            if _hs and len(np.unique(np.concatenate(_hy))) == 2:
                THR_FED, _ = thr_balanced(np.concatenate(_hy), np.concatenate(_hs))
            else:
                THR_FED = THR

            # per-client: alone vs federated, on the client's own held-out videos
            for c, info in zip(clients, pinfo):
                if c["Xte"].shape[0] == 0 or len(np.unique(c["yte"])) < 2:
                    info["local_ap"] = None
                    info["fed_ap"] = None
                    continue
                _, spf, ylf = eval_units(fc, fi, c["Xte_s"], c["yte"], c["kte"])
                info["fed_ap"] = float(average_precision_score(ylf, spf))
                if len(np.unique(c["ytr"])) == 2:
                    lm = LogisticRegression(C=C_BEST, max_iter=3000,
                                            class_weight="balanced")
                    lm.fit(c["Xtr_s"], c["ytr"])
                    _, spl, yll = eval_units(lm.coef_, lm.intercept_,
                                             c["Xte_s"], c["yte"], c["kte"])
                    info["local_ap"] = float(average_precision_score(yll, spl))
                else:
                    info["local_ap"] = None

            entry = dict(mode=mode, n_clients_requested=n_cl, n_clients=n_act,
                         usable_clients=len(usable), partition=pinfo, trace=trace,
                         threshold=THR_FED, threshold_source="client holdouts",
                         **prf(yl_f, sp_f, THR_FED),
                         ap_unit=float(average_precision_score(yl_f, sp_f)),
                         ap_crop=float(average_precision_score(yte, s_f)),
                         pr_unit=pr_points(yl_f, sp_f))
            entry["gap_ap_unit"] = entry["ap_unit"] - R["centralized"]["ap_unit"]
            entry["gap_f1"] = entry["f1"] - R["centralized"]["f1"]
            R["federated"].append(entry)
            log(f"  {mode:<9} n={n_act}: AP(unit)={entry['ap_unit']:.4f} "
                f"F1={entry['f1']:.4f} gap={entry['gap_ap_unit']:+.4f}")

            # ---- differential privacy sweep, on one representative config ----
            if mode == "balanced" and n_cl == 4:
                R["dp"] = []
                for sg in DP_SIGMAS:
                    dc, di, dtrace = run_federated(usable, dim, C_BEST, ev, agg,
                                                   sigma=sg)
                    _, spd, yld = eval_units(dc, di, Xte_fed, yte, kte)
                    R["dp"].append(dict(
                        sigma=sg, ap_unit=float(average_precision_score(yld, spd)),
                        **prf(yld, spd, THR)))
                    log(f"     DP sigma={sg}: AP={R['dp'][-1]['ap_unit']:.4f}")

    # ---------------- 5. communication cost ----------------
    probe_floats = dim + 1
    R["communication"] = dict(
        probe_params=probe_floats,
        bytes_per_round_per_client=probe_floats * 4,
        rounds=FED_ROUNDS,
        comparison={
            "linear probe (this work)": probe_floats,
            "ConvNeXt-Tiny fine-tuned": 28_000_000,
            "SSD MobileNetV2": 3_400_000,
            "YOLOv8s-cls": 5_083_298,
        })

    json.dump(R, open(a.out, "w"), indent=1)
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
