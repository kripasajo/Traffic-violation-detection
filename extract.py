#!/usr/bin/env python3
"""
Stage 1 — download RideSafe-400 once and extract crops for BOTH tasks.

Two detection tasks share the same source video, so the videos are downloaded
and decoded once and both crop sets are produced in the same pass.

  triple-riding : unit = ASSOCIATION (one motorcycle + its riders)
                  filename  vs{N}_{video}_assoc{id}_f{frame}.jpg
  helmet        : unit = RIDER TRACK (one rider followed across frames)
                  filename  vs{N}_{video}_track{id}_f{frame}.jpg

Both filenames carry the source video, so every downstream split can be made
at VIDEO level and verified. This is the defect being fixed on the helmet
side: the original helmet notebook wrote crops as 0.jpg, 1.jpg, ... and then
split them randomly, which puts frames of the SAME rider — taken a third of a
second apart — into both train and test. That inflates the score and cannot be
repaired after the fact, because the provenance was never recorded.

The train/val/test split is computed ONCE over all videos and shared by both
tasks, so a video that trains one model never tests the other either.
"""

import json
import os
import random
import shutil
import time
import zipfile
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict, Counter
from pathlib import Path

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------- config ---
SEED = 42
REPO = "DeepBug/RideSafe-400"
VIDEOSETS = [1, 2, 3, 4]
ROOT = Path(__file__).resolve().parent
DL = ROOT / "dl"
ANN = ROOT / "annots"
TRIPLE_ROOT = ROOT / "crops_triple"
HELMET_ROOT = ROOT / "crops_helmet"
METRICS = ROOT / "metrics"

# --- triple-riding settings (validated in the centralized pipeline) ---
MAX_CANDIDATES_PER_ASSOC = 15
FRAMES_TO_KEEP_PER_ASSOC = 8
MIN_CROP_SIZE = 150
BORDER_MARGIN = 3
BLUR_VAR_FLOOR = 30.0
CONTAM_IOU_MAX = 0.35
PAD_RATIO = 0.20
TRAIN_NORMAL_TO_TRIPLE_RATIO = 4

# --- helmet settings ---
HELMET_FRAME_STRIDE = 10        # sample every Nth annotated frame of a track
HELMET_FRAMES_PER_TRACK = 8     # cap per rider, mirrors the triple-riding cap
HELMET_MAX_PER_VIDEO_PER_CLASS = 10   # spread coverage over all videos
HELMET_MIN_CROP = 64            # rider boxes are much smaller than a vehicle
HELMET_PAD = 0.12

# decoder cache: unbounded caching exhausts the platform decoder limit and
# silently yields empty splits
MAX_OPEN_CAPS = 4
SEQ_GRAB_LIMIT = 60

random.seed(SEED)
np.random.seed(SEED)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fetch_unzip(filename, dest: Path):
    if dest.exists() and any(dest.iterdir()):
        log(f"  {dest.name} present, skip")
        return
    log(f"  downloading {filename}")
    t0 = time.time()
    p = hf_hub_download(repo_id=REPO, filename=filename, repo_type="dataset",
                        local_dir=str(DL))
    log(f"  unzip ({time.time()-t0:.0f}s dl)")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(p) as z:
        z.extractall(dest)
    os.remove(p)


# ------------------------------------------------------------ decoder ------
_caps = OrderedDict()
_last = {}


def get_cap(path):
    if path in _caps:
        _caps.move_to_end(path)
        return _caps[path]
    while len(_caps) >= MAX_OPEN_CAPS:
        old, cap = _caps.popitem(last=False)
        cap.release()
        _last.pop(old, None)
    _caps[path] = cv2.VideoCapture(path)
    return _caps[path]


def release_caps():
    for c in _caps.values():
        c.release()
    _caps.clear()
    _last.clear()


def read_frame(cap, path, n):
    """Read frame `n`, skipping forward cheaply when it is just ahead.

    Requires STRICT forward progress: with `0 <= n - last`, re-requesting the
    frame just read (n == last) skips the grab loop entirely and returns frame
    n+1, after which every subsequent sequential read is silently off by one.
    A failed grab also has to reset the cursor, or the position drifts.
    """
    last = _last.get(path, -1)
    if last != -1 and 1 <= n - last <= SEQ_GRAB_LIMIT:
        for _ in range(n - last - 1):
            if not cap.grab():
                _last[path] = -1
                return False, None
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ok, fr = cap.read()
    _last[path] = n if ok else -1
    return ok, fr


def sharp(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


# ------------------------------------------------- triple-riding parsing ----
def get_attr(box, name):
    for a in box.findall("attribute"):
        if a.attrib.get("name") == name:
            return a.text
    return None


def build_associations(xml_path):
    root = ET.parse(xml_path).getroot()
    riders, motors = defaultdict(list), {}
    for tr in root.findall("track"):
        lab, b = tr.attrib.get("label"), tr.find("box")
        if b is None:
            continue
        if lab == "rider":
            aid = get_attr(b, "association_id")
            if aid and aid != "-1":
                riders[aid].append(tr)
        elif lab == "motorcycle":
            mid = get_attr(b, "motor_track_id")
            if mid and mid != "-1":
                motors[mid] = tr

    out = []
    for aid, rtracks in riders.items():
        if aid not in motors:
            continue
        mtrack = motors[aid]
        frames = defaultdict(list)
        for rt in rtracks:
            for b in rt.findall("box"):
                if b.attrib.get("outside", "0") == "1":
                    continue
                frames[int(b.attrib["frame"])].append(
                    {"box": b, "occluded": b.attrib.get("occluded", "0") == "1"})
        mframes = {int(b.attrib["frame"]): b for b in mtrack.findall("box")
                   if b.attrib.get("outside", "0") == "0"}
        valid = {f: r for f, r in frames.items() if f in mframes and r}
        if not valid:
            continue
        label = "triple" if max(len(v) for v in valid.values()) >= 3 else "normal"
        pure = {f: r for f, r in valid.items() if (len(r) >= 3) == (label == "triple")}
        if not pure:
            continue
        fs = sorted(pure)
        if len(fs) > MAX_CANDIDATES_PER_ASSOC:
            step = len(fs) / MAX_CANDIDATES_PER_ASSOC
            fs = [fs[int(i * step)] for i in range(MAX_CANDIDATES_PER_ASSOC)]
        out.append({"xml": str(xml_path), "aid": aid, "label": label,
                    "motor_track": mtrack, "motor_frames": mframes,
                    "pure": pure, "cands": fs})
    return out


_moto_idx = {}


def moto_index(xml_path):
    if xml_path in _moto_idx:
        return _moto_idx[xml_path]
    idx = defaultdict(list)
    for tr in ET.parse(xml_path).getroot().findall("track"):
        if tr.attrib.get("label") != "motorcycle":
            continue
        tid = tr.attrib.get("id")
        for b in tr.findall("box"):
            if b.attrib.get("outside", "0") == "1":
                continue
            idx[int(b.attrib["frame"])].append((tid, (
                float(b.attrib["xtl"]), float(b.attrib["ytl"]),
                float(b.attrib["xbr"]), float(b.attrib["ybr"]))))
    _moto_idx[xml_path] = idx
    return idx


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def triple_crop(cap, vpath, n, mbox, riders, own_id, xml_path, W, H):
    cs = [(float(mbox.attrib["xtl"]), float(mbox.attrib["ytl"]),
           float(mbox.attrib["xbr"]), float(mbox.attrib["ybr"]))]
    occ = 0
    for r in riders:
        b = r["box"]
        cs.append((float(b.attrib["xtl"]), float(b.attrib["ytl"]),
                   float(b.attrib["xbr"]), float(b.attrib["ybr"])))
        occ += r["occluded"]
    x1 = min(c[0] for c in cs); y1 = min(c[1] for c in cs)
    x2 = max(c[2] for c in cs); y2 = max(c[3] for c in cs)
    trunc = (x1 <= BORDER_MARGIN or y1 <= BORDER_MARGIN or
             x2 >= W - BORDER_MARGIN or y2 >= H - BORDER_MARGIN)
    px, py = (x2-x1)*PAD_RATIO, (y2-y1)*PAD_RATIO
    cx1, cy1 = max(0, int(x1-px)), max(0, int(y1-py))
    cx2, cy2 = min(W, int(x2+px)), min(H, int(y2+py))
    if cx2-cx1 < MIN_CROP_SIZE or cy2-cy1 < MIN_CROP_SIZE:
        return None, -1e9
    ok, fr = read_frame(cap, vpath, n)
    if not ok or fr is None:
        return None, -1e9
    crop = fr[cy1:cy2, cx1:cx2].copy()
    del fr
    if crop.size == 0:
        return None, -1e9
    blur = sharp(crop)
    if blur < BLUR_VAR_FLOOR:
        return None, -1e9
    contam = 0.0
    for tid, ob in moto_index(xml_path).get(n, []):
        if tid != own_id:
            contam = max(contam, iou((cx1, cy1, cx2, cy2), ob))
    if contam > CONTAM_IOU_MAX:
        return None, -1e9
    q = (min(blur/100.0, 3.0) - (occ/max(1, len(riders)))*1.5
         - contam*3.0 - (1.5 if trunc else 0.0))
    return crop, q


def best_triple_crops(a, vpath):
    cap = get_cap(vpath)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    own = a["motor_track"].attrib.get("id")
    scored = []
    for n in a["cands"]:
        c, q = triple_crop(cap, vpath, n, a["motor_frames"][n], a["pure"][n],
                           own, a["xml"], W, H)
        if c is not None:
            scored.append((q, n, c))
    scored.sort(key=lambda t: -t[0])
    return [(n, c) for _, n, c in scored[:FRAMES_TO_KEEP_PER_ASSOC]]


# -------------------------------------------------------- helmet parsing ----
def build_rider_tracks(xml_path):
    """Rider tracks labelled helmet / no_helmet.

    A rider's helmet status is decided per frame by testing whether the centre
    of a helmet (or no_helmet) box falls inside the rider box -- the same rule
    the original notebook used. The TRACK is then labelled by majority vote and
    only the frames agreeing with that verdict are kept, so a single mislabelled
    frame cannot flip a rider's identity mid-track.
    """
    root = ET.parse(xml_path).getroot()

    # heads (helmet / no_helmet) per frame
    heads = defaultdict(list)
    for tr in root.findall("track"):
        lab = tr.attrib.get("label")
        if lab not in ("helmet", "no_helmet"):
            continue
        for b in tr.findall("box"):
            if b.attrib.get("outside", "0") == "1":
                continue
            cx = (float(b.attrib["xtl"]) + float(b.attrib["xbr"])) / 2
            cy = (float(b.attrib["ytl"]) + float(b.attrib["ybr"])) / 2
            heads[int(b.attrib["frame"])].append((cx, cy, lab))

    # rider boxes per frame
    riders = defaultdict(list)
    for tr in root.findall("track"):
        if tr.attrib.get("label") != "rider":
            continue
        tid = tr.attrib.get("id")
        for b in tr.findall("box"):
            if b.attrib.get("outside", "0") == "1":
                continue
            riders[int(b.attrib["frame"])].append((tid, (
                float(b.attrib["xtl"]), float(b.attrib["ytl"]),
                float(b.attrib["xbr"]), float(b.attrib["ybr"]))))

    # Assign heads to riders ONE-TO-ONE per frame.
    #
    # Riders sharing a motorcycle overlap heavily, so a naive "is this head
    # centre inside that rider box" test matches the driver's head to the
    # pillion too. Scanning helmets before no_helmets then makes "helmet" win
    # every tie, which systematically relabels bare-headed pillion riders as
    # helmeted -- destroying the minority class this task exists to detect.
    #
    # Two constraints fix it: a head must sit in the upper part of the rider
    # box (heads are above torsos), and each head may be consumed by only one
    # rider, best match first.
    HEAD_TOP_FRAC = 0.35
    assign = {}
    for n, rlist in riders.items():
        pairs = []
        for hi, (cx, cy, hlab) in enumerate(heads.get(n, [])):
            for ri, (tid, (x1, y1, x2, y2)) in enumerate(rlist):
                if not (x1 <= cx <= x2 and y1 <= cy <= y1 + HEAD_TOP_FRAC * (y2 - y1)):
                    continue
                # closeness to the top-centre of the rider box
                tcx, tcy = (x1 + x2) / 2, y1
                diag = max(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5, 1e-6)
                score = 1.0 - (((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5) / diag
                pairs.append((score, hi, ri, hlab))
        pairs.sort(key=lambda p: -p[0])
        used_h, used_r = set(), set()
        for score, hi, ri, hlab in pairs:
            if hi in used_h or ri in used_r:
                continue
            used_h.add(hi)
            used_r.add(ri)
            assign[(n, rlist[ri][0])] = (hlab, rlist[ri][1])

    tracks = []
    for tr in root.findall("track"):
        if tr.attrib.get("label") != "rider":
            continue
        tid = tr.attrib.get("id")
        frames = {}
        for b in tr.findall("box"):
            if b.attrib.get("outside", "0") == "1":
                continue
            n = int(b.attrib["frame"])
            got = assign.get((n, tid))
            if got is not None:
                frames[n] = (got[1], got[0])
        if not frames:
            continue
        votes = Counter(l for _, l in frames.values())
        label = votes.most_common(1)[0][0]
        keep = sorted(n for n, (_, l) in frames.items() if l == label)
        keep = keep[::HELMET_FRAME_STRIDE][:HELMET_FRAMES_PER_TRACK]
        if not keep:
            continue
        tracks.append({"tid": tid, "label": label,
                       "boxes": {n: frames[n][0] for n in keep},
                       "frames": keep})
    return tracks


def helmet_crops_for_video(xml_path, vpath, per_class_cap):
    """One sequential pass over the video, oldest frame first."""
    tracks = build_rider_tracks(xml_path)
    if not tracks:
        return []
    cap = get_cap(vpath)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    wanted = defaultdict(list)   # frame -> [(tid, box, label)]
    for t in tracks:
        for n in t["frames"]:
            wanted[n].append((t["tid"], t["boxes"][n], t["label"]))

    made = Counter()
    out = []
    for n in sorted(wanted):
        if all(made[c] >= per_class_cap for c in ("helmet", "no_helmet")):
            break
        if all(made[l] >= per_class_cap for _, _, l in wanted[n]):
            continue
        ok, fr = read_frame(cap, vpath, n)
        if not ok or fr is None:
            continue
        for tid, (x1, y1, x2, y2), lab in wanted[n]:
            if made[lab] >= per_class_cap:
                continue
            pw, ph = (x2-x1)*HELMET_PAD, (y2-y1)*HELMET_PAD
            cx1, cy1 = max(0, int(x1-pw)), max(0, int(y1-ph))
            cx2, cy2 = min(W, int(x2+pw)), min(H, int(y2+ph))
            if cx2-cx1 < HELMET_MIN_CROP or cy2-cy1 < HELMET_MIN_CROP:
                continue
            crop = fr[cy1:cy2, cx1:cx2].copy()
            if crop.size == 0 or sharp(crop) < BLUR_VAR_FLOOR:
                continue
            out.append((tid, n, lab, crop))
            made[lab] += 1
        del fr
    return out


# ------------------------------------------------------------------ main ---
def main():
    for d in (TRIPLE_ROOT, HELMET_ROOT, METRICS):
        d.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        for c in ("normal", "triple"):
            (TRIPLE_ROOT / split / c).mkdir(parents=True, exist_ok=True)
        for c in ("helmet", "no_helmet"):
            (HELMET_ROOT / split / c).mkdir(parents=True, exist_ok=True)

    log("=== annotations ===")
    ANN.mkdir(parents=True, exist_ok=True)
    xml_by_vs = {}
    for vs in VIDEOSETS:
        fetch_unzip(f"videoset{vs}_xml_annots_with_rider_motor_poly.zip",
                    ANN / f"vs{vs}")
        xml_by_vs[vs] = sorted((ANN / f"vs{vs}").rglob("*.xml"))
        log(f"  videoset{vs}: {len(xml_by_vs[vs])} xml")

    # one split shared by both tasks, at video level
    all_xml = [(vs, x) for vs in VIDEOSETS for x in xml_by_vs[vs]]
    tr, tmp = train_test_split(all_xml, test_size=0.30, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, random_state=SEED)
    SPLIT = {}
    for name, grp in (("train", tr), ("val", va), ("test", te)):
        for vs, x in grp:
            SPLIT[x.stem] = name
    log(f"video split: train={len(tr)} val={len(va)} test={len(te)}")
    json.dump(SPLIT, open(METRICS / "video_split.json", "w"), indent=1)

    stats = {"triple": defaultdict(Counter), "helmet": defaultdict(Counter)}

    for vs in VIDEOSETS:
        log(f"=== VIDEOSET {vs} ===")
        vdirs = []
        for part in (1, 2):
            d = ROOT / f"vid{vs}_{part}"
            fetch_unzip(f"videoset{vs}_videos_part{part}.zip", d)
            vdirs.append(d)
        lookup = {}
        for d in vdirs:
            for v in d.rglob("*.mp4"):
                lookup.setdefault(v.stem, str(v))
        log(f"  videos: {len(lookup)}")

        pairs = [(x, lookup[x.stem]) for x in xml_by_vs[vs] if x.stem in lookup]

        # ---------------- triple riding ----------------
        by_split = defaultdict(list)
        for x, vp in pairs:
            for a in build_associations(x):
                a["video"] = vp
                by_split[SPLIT[x.stem]].append(a)

        for split, assocs in by_split.items():
            assocs = assocs[:]
            random.Random(SEED).shuffle(assocs)
            order = lambda L: sorted(L, key=lambda a: (a["video"], a["cands"][0]))
            n_t = 0
            for a in order([a for a in assocs if a["label"] == "triple"]):
                for n, crop in best_triple_crops(a, a["video"]):
                    base = Path(a["xml"]).stem
                    cv2.imwrite(str(TRIPLE_ROOT / split / "triple" /
                                    f"vs{vs}_{base}_assoc{a['aid']}_f{n}.jpg"), crop)
                    n_t += 1
            cap_n = n_t * TRAIN_NORMAL_TO_TRIPLE_RATIO if split == "train" else float("inf")
            normals = [a for a in assocs if a["label"] == "normal"]
            # `normals` is already randomly shuffled; that shuffle IS the
            # sampling. Sorting by video is only a decoder-locality trick, so it
            # must be applied inside a small window that is consumed in full --
            # sorting a large batch and then stopping at the crop cap would
            # discard the shuffle and take every negative from the videos that
            # happen to sort earliest. The window counts ASSOCIATIONS; cap_n
            # counts CROPS, and each association yields several.
            CHUNK_ASSOCS = 32
            chunk = CHUNK_ASSOCS if cap_n != float("inf") else len(normals)
            n_n, i = 0, 0
            while i < len(normals) and n_n < cap_n:
                batch = normals[i:i+chunk]
                i += chunk
                for a in order(batch):
                    if n_n >= cap_n:
                        break
                    for n, crop in best_triple_crops(a, a["video"]):
                        if n_n >= cap_n:
                            break
                        base = Path(a["xml"]).stem
                        cv2.imwrite(str(TRIPLE_ROOT / split / "normal" /
                                        f"vs{vs}_{base}_assoc{a['aid']}_f{n}.jpg"), crop)
                        n_n += 1
            release_caps()
            stats["triple"][split].update({"normal": n_n, "triple": n_t})
            log(f"  triple[{split}] normal={n_n} triple={n_t}")

        # ---------------- helmet ----------------
        hcount = Counter()
        for x, vp in pairs:
            split = SPLIT[x.stem]
            for tid, n, lab, crop in helmet_crops_for_video(
                    x, vp, HELMET_MAX_PER_VIDEO_PER_CLASS):
                cv2.imwrite(str(HELMET_ROOT / split / lab /
                                f"vs{vs}_{x.stem}_track{tid}_f{n}.jpg"), crop)
                hcount[(split, lab)] += 1
            release_caps()
        for (split, lab), c in hcount.items():
            stats["helmet"][split][lab] += c
        log("  helmet " + " ".join(f"{s}/{l}={c}" for (s, l), c in sorted(hcount.items())))

        _moto_idx.clear()
        for d in vdirs:
            shutil.rmtree(d, ignore_errors=True)
        log(f"  removed videoset{vs} video")

    log("=== on-disk counts ===")
    summary = {}
    for task, rootdir, classes in (("triple", TRIPLE_ROOT, ("normal", "triple")),
                                   ("helmet", HELMET_ROOT, ("helmet", "no_helmet"))):
        summary[task] = {}
        for split in ("train", "val", "test"):
            summary[task][split] = {c: len(list((rootdir/split/c).glob("*.jpg")))
                                    for c in classes}
            log(f"  {task}/{split}: {summary[task][split]}")
            assert sum(summary[task][split].values()) > 0, \
                f"{task}/{split} EMPTY -- refusing to continue"
            for c in classes:
                assert summary[task][split][c] > 0, \
                    f"{task}/{split}/{c} has no examples"
    json.dump(summary, open(METRICS / "dataset_composition.json", "w"), indent=1)

    # leakage: no source video may appear in two splits, for either task
    import re
    for task, rootdir in (("triple", TRIPLE_ROOT), ("helmet", HELMET_ROOT)):
        sets = {}
        for split in ("train", "val", "test"):
            s = set()
            for p in rootdir.rglob(f"{split}/*/*.jpg"):
                s.add(re.sub(r"_(assoc|track).*$", "", p.name))
            sets[split] = s
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = sets[a] & sets[b]
            assert not overlap, f"LEAK in {task}: {len(overlap)} videos in {a} and {b}"
        log(f"  {task}: no video-level leakage "
            f"(train={len(sets['train'])} val={len(sets['val'])} test={len(sets['test'])} videos)")

    shutil.rmtree(DL, ignore_errors=True)
    shutil.rmtree(ANN, ignore_errors=True)
    log("EXTRACTION COMPLETE")


if __name__ == "__main__":
    main()
