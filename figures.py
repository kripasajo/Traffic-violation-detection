#!/usr/bin/env python3
"""
Stage 4 — publication figures and tables from the measured metrics.

Reads the JSON written by experiments.py and emits vector PDFs (for LaTeX)
plus PNGs, and CSVs of every plotted value so any number in the paper can be
traced back to its source.
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.autolayout": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})
C_CEN, C_FED, C_ALT = "#1f4e79", "#c1121f", "#2a9d8f"
TASK_LABEL = {"triple": "Triple riding", "helmet": "Helmet violation"}
# explicit (negative, positive) per task -- never infer this from dict order
TASK_CLASSES = {"triple": ("normal", "triple"),
                "helmet": ("helmet", "no_helmet")}


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png")


def csv(df, out, name):
    df.to_csv(os.path.join(out, f"{name}.csv"), index=False)


# --------------------------------------------------------------- figures ---
def fig_convergence(R, out):
    """Fig 1 — AP and weight drift per communication round."""
    fig, axes = plt.subplots(2, len(R), figsize=(4.2 * len(R), 5.2), squeeze=False)
    rows = []
    for j, (task, r) in enumerate(R.items()):
        ax, ax2 = axes[0][j], axes[1][j]
        for e in r["federated"]:
            if e["mode"] != "balanced":
                continue
            rd = [t["round"] for t in e["trace"]]
            ap = [t["ap_unit"] for t in e["trace"]]
            dr = [max(t["drift"], 1e-12) for t in e["trace"]]
            ax.plot(rd, ap, marker="o", ms=3, lw=1.4,
                    label=f"{e['n_clients']} clients")
            ax2.semilogy(rd, dr, marker="o", ms=3, lw=1.4,
                         label=f"{e['n_clients']} clients")
            for a_, d_, r_ in zip(ap, dr, rd):
                rows.append(dict(task=task, clients=e["n_clients"], round=r_,
                                 ap_unit=a_, drift=d_))
        ax.axhline(r["centralized"]["ap_unit"], color=C_CEN, ls="--", lw=1.2,
                   label="centralized")
        ax.set_title(f"{TASK_LABEL.get(task, task)} — convergence")
        ax.set_xlabel("communication round")
        ax.set_ylabel("average precision (per unit)")
        ax.legend(frameon=False)
        ax2.set_title("global weight change per round")
        ax2.set_xlabel("communication round")
        ax2.set_ylabel(r"$\|\Delta w\|_2$")
        ax2.legend(frameon=False)
        ax.ticklabel_format(useOffset=False, axis="y")
    fig.tight_layout(h_pad=2.2)
    csv(pd.DataFrame(rows), out, "fig1_convergence")
    save(fig, out, "fig1_convergence")


def fig_cen_vs_fed(R, out):
    """Fig 2 — centralized vs federated across client counts and partitions."""
    fig, axes = plt.subplots(1, len(R), figsize=(5.0 * len(R), 3.4), squeeze=False)
    rows = []
    for j, (task, r) in enumerate(R.items()):
        ax = axes[0][j]
        modes = sorted({e["mode"] for e in r["federated"]})
        counts = sorted({e["n_clients"] for e in r["federated"]})
        w = 0.8 / len(modes)
        for mi, mode in enumerate(modes):
            vals, xs = [], []
            for ci, n in enumerate(counts):
                hit = [e for e in r["federated"]
                       if e["mode"] == mode and e["n_clients"] == n]
                if not hit:
                    continue
                vals.append(hit[0]["ap_unit"])
                xs.append(ci + (mi - (len(modes) - 1) / 2) * w)
                rows.append(dict(task=task, mode=mode, clients=n,
                                 federated_ap=hit[0]["ap_unit"],
                                 centralized_ap=r["centralized"]["ap_unit"],
                                 gap=hit[0]["gap_ap_unit"],
                                 federated_f1=hit[0]["f1"],
                                 centralized_f1=r["centralized"]["f1"]))
            ax.bar(xs, vals, w * 0.92, label=f"federated ({mode})",
                   color=C_FED if mi == 0 else C_ALT, alpha=0.85)
        ax.axhline(r["centralized"]["ap_unit"], color=C_CEN, ls="--", lw=1.4,
                   label="centralized")
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels([f"{c}" for c in counts])
        ax.set_xlabel("number of clients")
        ax.set_ylabel("average precision (per unit)")
        ax.set_title(f"{TASK_LABEL.get(task, task)} — cost of federating")
        ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    csv(pd.DataFrame(rows), out, "fig2_centralized_vs_federated")
    save(fig, out, "fig2_centralized_vs_federated")


def fig_ablation(R, out):
    """Fig 3 — feature standardisation and regularisation."""
    fig, axes = plt.subplots(1, len(R), figsize=(4.4 * len(R), 3.2), squeeze=False)
    rows = []
    for j, (task, r) in enumerate(R.items()):
        ax = axes[0][j]
        for scaled, col, mk in ((True, C_CEN, "o"), (False, C_FED, "s")):
            pts = [(e["C"], e["ap_unit"]) for e in r["ablation"]
                   if e["model"] == "logreg" and e["scaled"] == scaled]
            pts.sort()
            ax.semilogx([p[0] for p in pts], [p[1] for p in pts], marker=mk,
                        ms=4, lw=1.4, color=col,
                        label="standardised" if scaled else "no standardisation")
        for e in r["ablation"]:
            rows.append(dict(task=task, **e))
        ax.set_xlabel("inverse regularisation strength  $C$")
        ax.set_ylabel("average precision (per unit)")
        ax.set_title(f"{TASK_LABEL.get(task, task)} — probe ablation")
        ax.legend(frameon=False)
    fig.tight_layout()
    csv(pd.DataFrame(rows), out, "fig3_ablation")
    save(fig, out, "fig3_ablation")


def fig_dataset(R, out):
    """Fig 4 — dataset composition and class imbalance."""
    fig, axes = plt.subplots(1, len(R), figsize=(4.2 * len(R), 3.2), squeeze=False)
    rows = []
    for j, (task, r) in enumerate(R.items()):
        ax = axes[0][j]
        d = r["data"]
        splits = ["train", "val", "test"]
        comp = r.get("composition", {})
        neg, pos, labels = [], [], []
        for s in splits:
            if s not in comp:
                continue
            vals = comp[s]
            kneg, kpos = TASK_CLASSES.get(task, tuple(vals.keys())[:2])
            labels.append(s)
            neg.append(vals[kneg])
            pos.append(vals[kpos])
            rows.append(dict(task=task, split=s, negative_class=kneg,
                             positive_class=kpos, negative=vals[kneg],
                             positive=vals[kpos],
                             positive_rate=vals[kpos] / max(sum(vals.values()), 1)))
        x = np.arange(len(labels))
        ax.bar(x, neg, 0.62, label=TASK_CLASSES.get(task, ("negative",))[0],
               color="#adb5bd")
        ax.bar(x, pos, 0.62, bottom=neg,
               label=TASK_CLASSES.get(task, (None, "positive"))[1],
               color=C_FED)
        for i, (n_, p_) in enumerate(zip(neg, pos)):
            ax.text(i, n_ + p_, f"{p_/(n_+p_)*100:.1f}%", ha="center",
                    va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("crops")
        ax.set_title(f"{TASK_LABEL.get(task, task)} — composition\n"
                     f"(percentages = positive rate)")
        ax.set_ylim(0, max(n_ + p_ for n_, p_ in zip(neg, pos)) * 1.22)
        ax.legend(frameon=False, loc="upper right", ncol=2)
    fig.tight_layout()
    csv(pd.DataFrame(rows), out, "fig4_dataset_composition")
    save(fig, out, "fig4_dataset_composition")


def fig_partition(R, out):
    """Fig 5 — how videos and positives are distributed over clients."""
    tasks = list(R)
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.6 * len(tasks), 3.2),
                             squeeze=False)
    rows = []
    for j, task in enumerate(tasks):
        ax = axes[0][j]
        r = R[task]
        sel = [e for e in r["federated"] if e["mode"] == "balanced"]
        sel.sort(key=lambda e: e["n_clients"])
        off = 0
        for e in sel:
            xs = [off + i for i in range(len(e["partition"]))]
            ax.bar(xs, [p["pos_units"] for p in e["partition"]], 0.8,
                   color=C_CEN, alpha=0.85)
            ax.text(np.mean(xs), -0.06 * max(1, max(p["pos_units"]
                    for p in e["partition"])), f"{e['n_clients']} clients",
                    ha="center", va="top", fontsize=8)
            for p in e["partition"]:
                rows.append(dict(task=task, mode=e["mode"],
                                 n_clients=e["n_clients"], **p))
            off += len(e["partition"]) + 1
        ax.set_ylabel("positive units per client")
        ax.set_xticks([])
        ax.set_title(f"{TASK_LABEL.get(task, task)} — partition balance")
    # include videoset-mode rows in the CSV even though the plot shows balanced
    for task, r in R.items():
        for e in r["federated"]:
            if e["mode"] != "balanced":
                for p in e["partition"]:
                    rows.append(dict(task=task, mode=e["mode"],
                                     n_clients=e["n_clients"], **p))
    fig.tight_layout()
    csv(pd.DataFrame(rows), out, "fig5_partition_balance")
    save(fig, out, "fig5_partition_balance")


def fig_per_client(R, out):
    """Fig 6 — training alone versus participating in federation."""
    fig, axes = plt.subplots(1, len(R), figsize=(4.6 * len(R), 3.2), squeeze=False)
    rows = []
    for j, (task, r) in enumerate(R.items()):
        ax = axes[0][j]
        sel = [e for e in r["federated"]
               if e["mode"] == "balanced" and e["n_clients"] == 4]
        if not sel:
            sel = [r["federated"][0]]
        e = sel[0]
        ids = [p["client"] for p in e["partition"] if p.get("local_ap") is not None]
        loc = [p["local_ap"] for p in e["partition"] if p.get("local_ap") is not None]
        fed = [p["fed_ap"] for p in e["partition"] if p.get("local_ap") is not None]
        x = np.arange(len(ids))
        ax.bar(x - 0.2, loc, 0.4, label="trained alone", color="#adb5bd")
        ax.bar(x + 0.2, fed, 0.4, label="federated", color=C_FED)
        for i, (l_, f_) in enumerate(zip(loc, fed)):
            ax.annotate(f"{f_-l_:+.3f}", (i, max(l_, f_)), ha="center",
                        va="bottom", fontsize=7)
        for cid, l_, f_ in zip(ids, loc, fed):
            rows.append(dict(task=task, mode=e["mode"], n_clients=e["n_clients"],
                             client=cid, local_ap=l_, fed_ap=f_, delta=f_ - l_))
        ax.set_xticks(x)
        ax.set_xticklabels([f"client {i}" for i in ids])
        ax.set_ylabel("average precision (own held-out videos)")
        ax.set_title(f"{TASK_LABEL.get(task, task)} — benefit of collaboration")
        ax.set_ylim(0, max(max(loc), max(fed)) * 1.20)
        ax.legend(frameon=False, loc="lower right", ncol=2)
    fig.tight_layout()
    csv(pd.DataFrame(rows), out, "fig6_per_client")
    save(fig, out, "fig6_per_client")


def fig_pr(R, out):
    """Fig 7 — precision-recall, centralized vs federated."""
    fig, axes = plt.subplots(1, len(R), figsize=(4.2 * len(R), 3.4), squeeze=False)
    rows = []
    for j, (task, r) in enumerate(R.items()):
        ax = axes[0][j]
        c = r["centralized"]["pr_unit"]
        ax.plot(c["recall"], c["precision"], color=C_CEN, lw=1.6,
                label=f"centralized (AP={c['ap']:.3f})")
        sel = [e for e in r["federated"]
               if e["mode"] == "balanced" and e["n_clients"] == 4]
        if sel:
            f = sel[0]["pr_unit"]
            ax.plot(f["recall"], f["precision"], color=C_FED, lw=1.6,
                    label=f"federated, 4 clients (AP={f['ap']:.3f})")
            for p_, r_ in zip(f["precision"], f["recall"]):
                rows.append(dict(task=task, model="federated", precision=p_, recall=r_))
        ax.axhline(c["baseline"], color="grey", ls=":", lw=1,
                   label=f"chance ({c['baseline']:.3f})")
        for p_, r_ in zip(c["precision"], c["recall"]):
            rows.append(dict(task=task, model="centralized", precision=p_, recall=r_))
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{TASK_LABEL.get(task, task)} — precision/recall")
        ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    csv(pd.DataFrame(rows), out, "fig7_pr_curves")
    save(fig, out, "fig7_pr_curves")


def fig_comm(R, out):
    """Fig 8 — cumulative uplink traffic against full-model baselines."""
    task = list(R)[0]
    comp = R[task]["communication"]["comparison"]
    # start at round 1: round 0 has zero traffic and log(0) plots as a cliff
    rounds = np.arange(1, R[task]["communication"]["rounds"] + 1)
    n_clients = 4
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    rows = []
    for name, params in comp.items():
        mb = rounds * n_clients * params * 4 / 1e6
        ax.plot(rounds, mb, lw=1.6, marker="o", ms=3,
                label=f"{name} ({params:,} params)")
        for r_, m_ in zip(rounds, mb):
            rows.append(dict(model=name, params=params, round=int(r_),
                             cumulative_MB=m_))
    ax.set_yscale("log")
    ax.set_xlabel("communication round")
    ax.set_ylabel("cumulative uplink (MB, 4 clients)")
    ax.set_title("Communication cost of federated training")
    ax.legend(frameon=False, fontsize=7)
    csv(pd.DataFrame(rows), out, "fig8_communication")
    save(fig, out, "fig8_communication")


def fig_dp(R, out):
    """Fig 9 — utility under differential-privacy noise."""
    have = {t: r for t, r in R.items() if r.get("dp")}
    if not have:
        return
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    rows = []
    for (task, r), col in zip(have.items(), (C_CEN, C_FED)):
        sg = [d["sigma"] for d in r["dp"]]
        ap = [d["ap_unit"] for d in r["dp"]]
        ax.plot(range(len(sg)), ap, marker="o", ms=4, lw=1.5, color=col,
                label=TASK_LABEL.get(task, task))
        ax.axhline(r["centralized"]["ap_unit"], color=col, ls="--", lw=1,
                   alpha=0.6)
        for s_, a_ in zip(sg, ap):
            rows.append(dict(task=task, sigma=s_, ap_unit=a_))
        ax.set_xticks(range(len(sg)))
        ax.set_xticklabels([str(s) for s in sg])
    ax.set_xlabel(r"noise multiplier $\sigma$ (dashed = centralized)")
    ax.set_ylabel("average precision (per unit)")
    ax.set_title("Privacy / utility trade-off under the Gaussian mechanism")
    ax.legend(frameon=False)
    csv(pd.DataFrame(rows), out, "fig9_dp_tradeoff")
    save(fig, out, "fig9_dp_tradeoff")


# ---------------------------------------------------------------- tables ---
def tables(R, out):
    rows = []
    for task, r in R.items():
        c = r["centralized"]
        rows.append(dict(task=task, setting="centralized", clients=0, mode="-",
                         precision=c["precision"], recall=c["recall"], f1=c["f1"],
                         ap_unit=c["ap_unit"], ap_crop=c["ap_crop"], gap_ap="-"))
        for e in r["federated"]:
            rows.append(dict(task=task, setting="federated",
                             clients=e["n_clients"],
                             clients_requested=e.get("n_clients_requested"),
                             mode=e["mode"], threshold=e.get("threshold"),
                             precision=e["precision"], recall=e["recall"],
                             f1=e["f1"], ap_unit=e["ap_unit"],
                             ap_crop=e["ap_crop"],
                             gap_ap=round(e["gap_ap_unit"], 4)))
    main = pd.DataFrame(rows)
    csv(main, out, "table2_main_results")

    par = []
    for task, r in R.items():
        cfg, d = r["config"], r["data"]
        par.append(dict(task=task, backbone="DINOv2 ViT-B/14 (frozen)",
                        input_resolution="336x336 letterboxed",
                        feature_dim=d["dim"], probe="logistic regression",
                        regularisation_C=cfg.get("probe_C"),
                        aggregation=cfg.get("aggregator"),
                        global_rounds=cfg["fed_rounds"],
                        local_steps=cfg["local_iters"],
                        clients=", ".join(str(c) for c in sorted(
                            {e["n_clients"] for e in r["federated"]})),
                        partitioning="balanced / per-videoset",
                        client_holdout=cfg["client_test_frac"],
                        decision_threshold=round(cfg.get("threshold", 0), 4),
                        model_selection="GroupKFold(5) by source video",
                        evaluation_unit=("association" if task == "triple"
                                         else "rider track"),
                        metrics="AP, precision, recall, F1",
                        dp_clip=cfg["dp_clip"], seed=cfg["seed"]))
    csv(pd.DataFrame(par), out, "table1_experimental_setup")

    dat = []
    for task, r in R.items():
        d = r["data"]
        dat.append(dict(task=task, **{k: d[k] for k in sorted(d)}))
    csv(pd.DataFrame(dat), out, "table3_dataset_summary")

    # Built in memory then written once: a half-written summary at the end of a
    # multi-hour run is worse than none. to_markdown() needs `tabulate`, which
    # may be absent, so fall back to plain text rather than lose the file.
    def _md(df, transpose=False):
        d = df.T if transpose else df
        try:
            return d.to_markdown(**({} if transpose else
                                    dict(index=False, floatfmt=".4f")))
        except ImportError:
            return "```\n" + d.to_string(index=not transpose is False) + "\n```"

    body = ("# Results summary\n\n## Main results\n\n" + _md(main)
            + "\n\n## Experimental setup\n\n" + _md(pd.DataFrame(par), True)
            + "\n\n## Dataset\n\n" + _md(pd.DataFrame(dat), True) + "\n")
    with open(os.path.join(out, "results_summary.md"), "w") as f:
        f.write(body)
    print("  wrote tables + results_summary.md")


def fig_qualitative(R, out, metrics_paths):
    """Fig 10 — real successes and failures on held-out test data.

    Examples are drawn by rank from the model's own scores (most confident
    correct, most confident wrong, worst miss) rather than hand-picked, so the
    figure cannot flatter the model.
    """
    import cv2
    for task, r in R.items():
        p = [q for q in metrics_paths if json.load(open(q))["task"] == task]
        if not p:
            continue
        pred_file = p[0].replace(".json", "_testpred.npz")
        if not os.path.exists(pred_file):
            print(f"  qualitative[{task}]: no predictions, skipping")
            continue
        z = np.load(pred_file, allow_pickle=True)
        paths, y, s = z["paths"].astype(str), z["y"], z["score"]
        thr = float(z["threshold"])
        pred = (s >= thr).astype(int)

        groups = [
            ("true positive", np.where((pred == 1) & (y == 1))[0], -s),
            ("true negative", np.where((pred == 0) & (y == 0))[0], s),
            ("false positive", np.where((pred == 1) & (y == 0))[0], -s),
            ("false negative", np.where((pred == 0) & (y == 1))[0], s),
        ]
        n_col = 5
        fig, axes = plt.subplots(len(groups), n_col,
                                 figsize=(1.65 * n_col, 1.85 * len(groups)))
        for gi, (name, idx, key) in enumerate(groups):
            idx = idx[np.argsort(key[idx])][:n_col] if len(idx) else idx
            for ci in range(n_col):
                ax = axes[gi][ci]
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if ci < len(idx):
                    im = cv2.imread(paths[idx[ci]])
                    if im is not None:
                        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                        ax.set_title(f"p={1/(1+np.exp(-s[idx[ci]])):.2f}",
                                     fontsize=6, pad=1.5)
                if ci == 0:
                    ax.set_ylabel(name, fontsize=7.5, rotation=0,
                                  ha="right", va="center", labelpad=32)
        fig.suptitle(f"{TASK_LABEL.get(task, task)} — held-out examples "
                     f"(ranked by model confidence, not hand-picked)",
                     fontsize=9)
        save(fig, out, f"fig10_qualitative_{task}")


def fig_architecture(out):
    """Fig 11 — schematic of the federated setup (illustrative, no data)."""
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    ax.grid(False)

    n = 4
    for i in range(n):
        x = 0.5 + i * 2.35
        ax.add_patch(plt.Rectangle((x, 3.4), 1.9, 2.1, fc="#eef2f6",
                                   ec=C_CEN, lw=1.2))
        ax.text(x + .95, 5.22, f"Client {i+1}", ha="center", fontsize=8.5,
                weight="bold")
        ax.text(x + .95, 4.78, "local video", ha="center", fontsize=7)
        ax.text(x + .95, 4.44, "crops (stay local)", ha="center", fontsize=6.6,
                style="italic", color="#6c757d")
        ax.text(x + .95, 4.06, "frozen DINOv2\n(never trained)", ha="center",
                fontsize=6.6)
        ax.text(x + .95, 3.58, "train probe", ha="center", fontsize=7,
                color=C_FED)
        ax.annotate("", xy=(5.0, 2.15), xytext=(x + .95, 3.35),
                    arrowprops=dict(arrowstyle="->", lw=1.1, color=C_FED))
        if i == n - 1:      # label the payload once; the arrows are identical
            ax.text(x + 1.72, 2.85, "769 floats\nper round", fontsize=6.2,
                    color=C_FED, ha="center")

    ax.add_patch(plt.Rectangle((3.3, 0.9), 3.4, 1.25, fc="#fdf0ee",
                               ec=C_FED, lw=1.4))
    ax.text(5.0, 1.78, "Aggregation server", ha="center", fontsize=9,
            weight="bold")
    ax.text(5.0, 1.40, "FedAvg over probe weights", ha="center", fontsize=7.5)
    ax.text(5.0, 1.08, "no raw video, no crops, no features", ha="center",
            fontsize=6.6, style="italic", color="#6c757d")
    ax.annotate("", xy=(0.62, 3.35), xytext=(3.35, 1.95),
                arrowprops=dict(arrowstyle="->", lw=1.1, ls=":", color=C_CEN,
                                connectionstyle="arc3,rad=0.25"))
    ax.text(0.30, 2.55, "broadcast\nglobal weights", fontsize=6.4, color=C_CEN,
            ha="left")
    ax.text(5.0, 0.42, "Only the linear probe is exchanged; the backbone is "
            "identical on every client and never transmitted.",
            ha="center", fontsize=7, color="#343a40")
    save(fig, out, "fig11_architecture")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", nargs="+", required=True)
    ap.add_argument("--crops-triple", default=None)
    ap.add_argument("--crops-helmet", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    R = {}
    for p in a.metrics:
        r = json.load(open(p))
        comp_path = os.path.join(os.path.dirname(p), "dataset_composition.json")
        if os.path.exists(comp_path):
            r["composition"] = json.load(open(comp_path)).get(r["task"], {})
        R[r["task"]] = r
    print(f"tasks: {list(R)}")

    failures = []
    for fn in (fig_convergence, fig_cen_vs_fed, fig_ablation, fig_dataset,
               fig_partition, fig_per_client, fig_pr, fig_comm, fig_dp):
        try:
            fn(R, a.out)
        except Exception as e:
            import traceback
            print(f"  !! {fn.__name__} failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            failures.append(fn.__name__)
    for fn, args in ((fig_qualitative, (R, a.out, a.metrics)),
                     (fig_architecture, (a.out,))):
        try:
            fn(*args)
        except Exception as e:
            import traceback
            print(f"  !! {fn.__name__} failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            failures.append(fn.__name__)
    try:
        tables(R, a.out)
    except Exception as e:
        print(f"  !! tables failed: {type(e).__name__}: {e}")
        failures.append("tables")

    # A silently-missing figure is the failure mode to avoid here: the run takes
    # hours and nobody re-reads the log line by line.
    if failures:
        print(f"\n!!! {len(failures)} OUTPUT(S) FAILED: {', '.join(failures)}")
        print("FIGURES INCOMPLETE")
        sys.exit(1)
    else:
        print("\nall figures and tables produced")
    print("FIGURES COMPLETE")


if __name__ == "__main__":
    main()
