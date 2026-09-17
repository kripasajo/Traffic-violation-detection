#!/usr/bin/env python3
"""
Stage 2 — frozen DINOv2 features for both crop sets.

The backbone is pretrained and never updated, so it is identical on every
federated client and never needs transmitting; only a small probe is trained.

Crops are letterboxed rather than squashed to a square: the aspect ratio of a
crop carries real signal (three riders make a visibly different box than one)
and a plain resize destroys it. Horizontal-flip test-time augmentation is
averaged into every embedding.

Emits, per split:  X (features), y (labels), key (unit id), vid (source video)
where the unit is an association for triple-riding and a rider track for helmet.
"""

import argparse
import glob
import os
import re
import sys
import time

import cv2
import numpy as np
import torch
import timm
from torch.utils.data import Dataset, DataLoader

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
DEV = torch.device("cuda" if torch.cuda.is_available()
                   else "mps" if torch.backends.mps.is_available() else "cpu")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def letterbox(img, S, pad=114):
    h, w = img.shape[:2]
    s = S / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    im = cv2.resize(img, (nw, nh),
                    interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    t, l = (S - nh) // 2, (S - nw) // 2
    return cv2.copyMakeBorder(im, t, S - nh - t, l, S - nw - l,
                              cv2.BORDER_CONSTANT, value=(pad, pad, pad))


class Crops(Dataset):
    """Module-level so DataLoader workers can pickle it on any start method."""

    def __init__(self, paths, size):
        self.paths, self.size = paths, size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = cv2.imread(self.paths[i])
        if im is None:
            im = np.full((self.size, self.size, 3), 114, np.uint8)
        im = letterbox(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), self.size)
        x = (im.astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1))


def unit_key(path):
    """(video, unit) -- association for triple-riding, rider track for helmet."""
    b = os.path.basename(path)[:-4]
    m = re.match(r"^(.*)_(?:assoc|track)([^_]+)_f(\d+)$", b)
    return (m.group(1), m.group(2)) if m else (b, "?")


@torch.no_grad()
def extract(model, paths, size, batch, workers):
    for w in (workers, 0):
        try:
            dl = DataLoader(Crops(paths, size), batch_size=batch,
                            num_workers=w, shuffle=False)
            feats, t0, seen = [], time.time(), 0
            for xb in dl:
                xb = xb.to(DEV)
                f = model(xb).float()
                f = (f + model(torch.flip(xb, dims=[3])).float()) / 2
                feats.append(f.cpu().numpy())
                seen += xb.size(0)
                if seen % (batch * 40) < batch:
                    log(f"    {seen}/{len(paths)} ({seen/max(time.time()-t0,1e-9):.1f}/s)")
            return np.concatenate(feats).astype(np.float32)
        except Exception as e:
            if w == 0:
                raise
            log(f"    parallel loading failed ({type(e).__name__}); single-process")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", nargs=2, required=True,
                    help="negative then positive class directory name")
    ap.add_argument("--size", type=int, default=336)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default="vit_base_patch14_reg4_dinov2.lvd142m")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    log(f"backbone={a.model} size={a.size} device={DEV}")
    try:
        model = timm.create_model(a.model, pretrained=True, num_classes=0,
                                  img_size=a.size)
    except TypeError:
        model = timm.create_model(a.model, pretrained=True, num_classes=0)
    model = model.to(DEV).eval()

    for split in ("train", "val", "test"):
        paths, y = [], []
        for lab, cls in enumerate(a.classes):     # classes[0]=0, classes[1]=1
            p = sorted(glob.glob(os.path.join(a.crops, split, cls, "*.jpg")))
            paths += p
            y += [lab] * len(p)
        if not paths:
            log(f"  {split}: EMPTY, skipping")
            continue
        log(f"  {split}: {len(paths)} crops ({sum(y)} positive)")
        t0 = time.time()
        X = extract(model, paths, a.size, a.batch, a.workers)
        keys = [unit_key(p) for p in paths]
        np.savez(os.path.join(a.out, f"{split}.npz"),
                 X=X, y=np.array(y),
                 key=np.array([f"{v}||{u}" for v, u in keys]),
                 vid=np.array([v for v, _ in keys]),
                 paths=np.array(paths))
        log(f"  {split}: {X.shape} {len(set(keys))} units "
            f"[{(time.time()-t0)/60:.1f} min]")
    log("FEATURES COMPLETE")


if __name__ == "__main__":
    main()
