#!/usr/bin/env python3
"""Group extracted GoPro frames into bursts, coregister them, and write a
training-ready LR/HR dataset for the Option-B (RGB-native) Burstormer.

Pipeline per burst window:
  1. Take `burst_size` consecutive frames; the CENTER frame is the reference
     (minimizes the max temporal/geometric gap any neighbor has to cover,
     vs. picking the first frame).
  2. Align every other frame to the reference. Alignment is estimated on a
     DOWNSCALED copy of each frame pair (--align_max_dim, default 640px on
     the long side) -- cv2.findTransformECC (homography) first, ORB+RANSAC
     homography as a fallback if ECC doesn't converge -- and the resulting
     homography is rescaled back to full resolution before warping. This is
     the single biggest lever on runtime: ECC's cost scales with pixel
     count, so running it at native 4K/5.3K resolution is 20-50x slower
     than necessary for no real accuracy gain -- the final LR frames get
     downsampled by --sr_factor anyway, so registration only needs to be
     accurate to well within a native pixel. (Benchmarked: ~22s/frame-pair
     at native 3840x2160 vs. ~0.5s/frame-pair at 640px, same resulting
     alignment accuracy.) Frames that fail both methods are dropped from
     that burst -- Burstormer's adaptive burst pooling tolerates a variable
     burst size, so this is safe rather than padding with garbage.
  3. Intersect the valid (non-black-border) regions of all warped frames,
     shrink by --border_margin, and sample one or more --patch_size crops
     from that common region. Each candidate crop is screened before it's
     accepted:
       - --min_laplacian_var rejects crops that are too FLAT/low-detail
         (sky, still water, blank walls, out-of-focus background). This is
         a texture/focus measure, not a color one.
       - --max_vegetation_frac rejects crops where more than that fraction
         of pixels look green/foliage-colored (grass, leaves, trees), via
         a rough HSV heuristic. This is a SEPARATE filter from the texture
         one on purpose: grass/foliage is usually HIGH local variance (lots
         of fine edges), so a flatness filter alone won't catch it -- what
         makes it a poor MFSR target is that the fine detail tends to be
         repetitive/aliased rather than genuinely recoverable structure,
         and it's cheap to flag by color instead of trying to distinguish
         "good" high-frequency detail from "bad" high-frequency detail with
         a single texture number. Note this heuristic will also flag other
         green content (painted walls, green fabric, etc.) -- it's a proxy,
         not semantic segmentation.
     Each candidate location is retried up to --texture_max_attempts times
     before that patch slot is given up on (skipped, not padded).
  4. The reference crop (native resolution) becomes the HR ground truth.
     Every frame in the aligned burst (reference included, as burst[0]) is
     downsampled by --sr_factor to build the LR input burst.
  5. Everything is written to disk as
         out/<split>/<clip>_w<window>_p<patch>/hr.png
         out/<split>/<clip>_w<window>_p<patch>/lr/im_00.png ... im_{k-1}.png
         out/<split>/<clip>_w<window>_p<patch>/meta.json
     which `gopro_burst_dataset.py` reads directly. Scene names are derived
     from (clip, window, patch) rather than a shared running counter, so
     windows can be processed in parallel across worker processes without
     any coordination. meta.json also records each patch's laplacian_var
     and vegetation_frac so you can audit what got kept/dropped later.

Usage:
    python build_bursts.py --frames_dir frames/GX010001 --outdir dataset \
        --burst_size 14 --sr_factor 4 --patch_size 384 --patches_per_window 2

    # multiple clips at once (recommended -- keeps scene naming from colliding)
    python build_bursts.py --frames_dir frames --multi_clip --outdir dataset \
        --burst_size 14 --sr_factor 4 --patch_size 384 --patches_per_window 2

    # use all CPU cores (each window is independent work)
    python build_bursts.py --frames_dir frames --multi_clip --outdir dataset \
        --workers 8

    # skip flat sky/water AND grass/foliage-dominated patches
    python build_bursts.py --frames_dir frames --multi_clip --outdir dataset \
        --min_laplacian_var 15 --max_vegetation_frac 0.6 --texture_max_attempts 25
"""
import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #

def resize_for_alignment(img, max_dim):
    """Downscale `img` so its longest side is <= max_dim. Returns (small_img, scale)
    where scale maps full-res coordinates -> small-res coordinates."""
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale >= 1.0:
        return img, 1.0
    small = cv2.resize(img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                        interpolation=cv2.INTER_AREA)
    return small, scale


def estimate_homography_ecc(ref_gray_f32, mov_gray_f32, iterations, eps):
    """Returns H (in the SAME pixel coordinates as the inputs) such that
    cv2.warpPerspective(mov, H, size) aligns mov -> ref, or None if ECC fails
    to converge."""
    warp_matrix = np.eye(3, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, eps)
    try:
        _, warp_matrix = cv2.findTransformECC(
            ref_gray_f32, mov_gray_f32, warp_matrix, cv2.MOTION_HOMOGRAPHY, criteria, None, 5
        )
    except cv2.error:
        return None
    # findTransformECC(template=ref, input=mov, ...) returns the warp that
    # needs WARP_INVERSE_MAP to send mov -> ref. Invert once here so every
    # caller downstream can just do a plain forward warpPerspective(mov, H).
    try:
        return np.linalg.inv(warp_matrix).astype(np.float64)
    except np.linalg.LinAlgError:
        return None


def estimate_homography_orb(ref_gray_u8, mov_gray_u8, min_matches=30):
    """Returns H (in the SAME pixel coordinates as the inputs), or None."""
    orb = cv2.ORB_create(4000)
    kp_ref, des_ref = orb.detectAndCompute(ref_gray_u8, None)
    kp_mov, des_mov = orb.detectAndCompute(mov_gray_u8, None)
    if des_ref is None or des_mov is None or len(kp_ref) < 4 or len(kp_mov) < 4:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des_mov, des_ref, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < min_matches:
        return None
    src_pts = np.float32([kp_mov[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    if H is None:
        return None
    return H.astype(np.float64)  # already forward: mov -> ref


def rescale_homography(H_small, scale):
    """Converts a homography estimated on images downscaled by `scale`
    (small = full * scale) into one that operates on full-resolution
    coordinates directly."""
    if scale >= 1.0:
        return H_small.astype(np.float32)
    T = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)
    H_full = np.linalg.inv(T) @ H_small @ T
    return H_full.astype(np.float32)


def align_frame(ref_bgr, mov_bgr, ecc_iterations, ecc_eps, align_max_dim):
    """Returns (warped_bgr, valid_mask_u8, method) or (None, None, None) on failure.
    Homography is ESTIMATED on downscaled copies (fast) and APPLIED at full
    resolution (so the output is still native-res)."""
    h, w = ref_bgr.shape[:2]

    ref_small, scale = resize_for_alignment(ref_bgr, align_max_dim)
    mov_small, _ = resize_for_alignment(mov_bgr, align_max_dim)  # same input size -> same scale

    ref_gray_small = cv2.cvtColor(ref_small, cv2.COLOR_BGR2GRAY)
    mov_gray_small = cv2.cvtColor(mov_small, cv2.COLOR_BGR2GRAY)

    H_small = estimate_homography_ecc(
        ref_gray_small.astype(np.float32) / 255.0, mov_gray_small.astype(np.float32) / 255.0,
        ecc_iterations, ecc_eps,
    )
    method = "ecc"
    if H_small is None:
        H_small = estimate_homography_orb(ref_gray_small, mov_gray_small)
        method = "orb"
    if H_small is None:
        return None, None, None

    H_full = rescale_homography(H_small, scale)

    warped = cv2.warpPerspective(
        mov_bgr, H_full, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    mask = cv2.warpPerspective(
        np.full((h, w), 255, dtype=np.uint8), H_full, (w, h), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return warped, mask, method


# --------------------------------------------------------------------------- #
# Patch content filtering (texture / vegetation)
# --------------------------------------------------------------------------- #

def laplacian_variance(gray_u8):
    """Focus/texture measure: variance of the Laplacian. Low values mean a
    flat, low-detail region (sky, still water, blank walls, out-of-focus
    background); values well above the noise floor indicate genuine edges
    or texture worth training on. Exact useful threshold depends on your
    footage's exposure/compression -- audit meta.json's laplacian_var
    values on a sample run before locking in --min_laplacian_var."""
    lap = cv2.Laplacian(gray_u8, cv2.CV_64F)
    return float(lap.var())


def vegetation_fraction(bgr_u8):
    """Rough heuristic fraction of pixels that look like foliage/grass:
    green-dominant hue in HSV, with enough saturation/value to exclude
    near-gray or near-black/white pixels. This is a COLOR heuristic, not a
    texture one -- grass and leaves are often high local variance (lots of
    fine edges), so laplacian_variance() alone won't catch them; this
    filter targets them by color instead. It will also flag other green
    content (painted walls, green fabric, astroturf, etc.) -- it's a cheap
    proxy, not semantic segmentation."""
    hsv = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # OpenCV hue range is 0-179; green foliage typically falls ~35-85.
    green_mask = (h >= 35) & (h <= 85) & (s > 40) & (v > 30)
    return float(green_mask.mean())


def evaluate_patch_content(ref_crop_bgr, min_laplacian_var, max_vegetation_frac):
    """Returns (passes, laplacian_var, vegetation_frac). Only computes the
    vegetation heuristic when it's actually active (max_vegetation_frac < 1.0),
    since it's the pricier of the two checks."""
    gray = cv2.cvtColor(ref_crop_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = laplacian_variance(gray)

    if lap_var < min_laplacian_var:
        return False, lap_var, 0.0

    if max_vegetation_frac < 1.0:
        veg_frac = vegetation_fraction(ref_crop_bgr)
        if veg_frac > max_vegetation_frac:
            return False, lap_var, veg_frac
    else:
        veg_frac = 0.0

    return True, lap_var, veg_frac


# --------------------------------------------------------------------------- #
# Burst assembly
# --------------------------------------------------------------------------- #

def common_valid_bbox(masks, border_margin):
    """Intersection of a list of 0/255 masks, eroded by border_margin, as (x, y, w, h)."""
    inter = masks[0].copy()
    for m in masks[1:]:
        inter = cv2.bitwise_and(inter, m)
    if border_margin > 0:
        kernel = np.ones((border_margin * 2 + 1, border_margin * 2 + 1), np.uint8)
        inter = cv2.erode(inter, kernel)
    ys, xs = np.where(inter > 0)
    if len(xs) == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def build_window(frames_bgr, ecc_iterations, ecc_eps, min_valid_frames, align_max_dim):
    """Aligns every frame in the window to the center frame.
    Returns (aligned_bgr_list, masks_list, ref_index_in_window, methods) with
    the reference itself first in the returned lists."""
    ref_idx = len(frames_bgr) // 2
    ref = frames_bgr[ref_idx]
    h, w = ref.shape[:2]

    aligned = [ref]
    masks = [np.full((h, w), 255, dtype=np.uint8)]
    methods = ["reference"]

    for i, frame in enumerate(frames_bgr):
        if i == ref_idx:
            continue
        warped, mask, method = align_frame(ref, frame, ecc_iterations, ecc_eps, align_max_dim)
        if warped is None:
            continue
        aligned.append(warped)
        masks.append(mask)
        methods.append(method)

    if len(aligned) < min_valid_frames:
        return None

    return aligned, masks, ref_idx, methods


def sample_patches(aligned, masks, patch_size, patches_per_window, border_margin, rng,
                    min_laplacian_var=0.0, max_vegetation_frac=1.0, max_attempts=20):
    """Samples up to `patches_per_window` crops from the common valid region.

    If min_laplacian_var > 0 or max_vegetation_frac < 1.0, each candidate
    location is screened via evaluate_patch_content() on the REFERENCE
    frame's crop (aligned[0] -- the frame that becomes hr.png, and the most
    representative single frame for the burst's content). A candidate that
    fails is discarded and a new random location is tried, up to
    max_attempts times, before that patch slot is given up on. This means a
    window can end up with fewer than patches_per_window patches (or zero)
    if it's dominated by flat/vegetation content -- that's intentional,
    same "skip rather than pad with garbage" philosophy as frame alignment.

    Returns a list of (crop_frames, (px, py), metrics_dict) tuples, where
    metrics_dict has "laplacian_var" and "vegetation_frac" for the frame
    that was ACCEPTED (0.0 filler values when filtering is fully disabled).
    """
    bbox = common_valid_bbox(masks, border_margin)
    if bbox is None:
        return []
    x0, y0, bw, bh = bbox
    if bw < patch_size or bh < patch_size:
        return []

    filtering_enabled = (min_laplacian_var > 0.0) or (max_vegetation_frac < 1.0)
    attempts_per_patch = max_attempts if filtering_enabled else 1

    patches = []
    for _ in range(patches_per_window):
        accepted = None
        for _attempt in range(attempts_per_patch):
            px = rng.randint(x0, x0 + bw - patch_size)
            py = rng.randint(y0, y0 + bh - patch_size)

            if not filtering_enabled:
                accepted = (px, py, 0.0, 0.0)
                break

            ref_crop = aligned[0][py:py + patch_size, px:px + patch_size]
            ok, lap_var, veg_frac = evaluate_patch_content(
                ref_crop, min_laplacian_var, max_vegetation_frac
            )
            if ok:
                accepted = (px, py, lap_var, veg_frac)
                break

        if accepted is None:
            # Exhausted attempts -- every candidate location in this window
            # looked flat (sky/water) or vegetation-dominated. Skip this
            # slot rather than writing a low-value patch.
            continue

        px, py, lap_var, veg_frac = accepted
        crop = [frame[py:py + patch_size, px:px + patch_size] for frame in aligned]
        metrics = {"laplacian_var": round(float(lap_var), 2),
                   "vegetation_frac": round(float(veg_frac), 3)}
        patches.append((crop, (px, py), metrics))

    return patches


# --------------------------------------------------------------------------- #
# Per-window worker (top-level function so it's picklable for multiprocessing)
# --------------------------------------------------------------------------- #

def process_window(task):
    (clip_name, w_idx, window_paths, split, args_dict, seed) = task
    rng = random.Random(seed)

    frames_bgr = [cv2.imread(str(p)) for p in window_paths]
    if any(f is None for f in frames_bgr):
        return f"  [{clip_name}] window {w_idx}: unreadable frame(s), skipping"

    result = build_window(
        frames_bgr, args_dict["ecc_iterations"], args_dict["ecc_eps"],
        args_dict["min_valid_frames"], args_dict["align_max_dim"],
    )
    if result is None:
        return f"  [{clip_name}] window {w_idx}: not enough frames aligned, skipping"
    aligned, masks, ref_idx_in_window, methods = result

    patches = sample_patches(
        aligned, masks, args_dict["patch_size"], args_dict["patches_per_window"],
        args_dict["border_margin"], rng,
        min_laplacian_var=args_dict["min_laplacian_var"],
        max_vegetation_frac=args_dict["max_vegetation_frac"],
        max_attempts=args_dict["texture_max_attempts"],
    )
    if not patches:
        return (f"  [{clip_name}] window {w_idx}: no patches passed content filters "
                f"(or common region smaller than patch_size={args_dict['patch_size']}), skipping")

    for p_idx, (crop_frames, (px, py), metrics) in enumerate(patches):
        scene_name = f"{clip_name}_w{w_idx:05d}_p{p_idx:02d}"
        scene_dir = Path(args_dict["outdir"]) / split / scene_name
        (scene_dir / "lr").mkdir(parents=True, exist_ok=True)

        hr = crop_frames[0]  # reference is always index 0
        cv2.imwrite(str(scene_dir / "hr.png"), hr)

        lr_size = args_dict["patch_size"] // args_dict["sr_factor"]
        for k, frame in enumerate(crop_frames):
            lr_frame = cv2.resize(frame, (lr_size, lr_size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(scene_dir / "lr" / f"im_{k:02d}.png"), lr_frame)

        meta = {
            "clip": clip_name,
            "window_index": w_idx,
            "reference_frame": str(window_paths[ref_idx_in_window].name),
            "source_frames": [p.name for p in window_paths],
            "alignment_methods": methods,
            "burst_size": len(crop_frames),
            "patch_size": args_dict["patch_size"],
            "sr_factor": args_dict["sr_factor"],
            "patch_origin_xy": [int(px), int(py)],
            "reference_index": 0,
            "laplacian_var": metrics["laplacian_var"],
            "vegetation_frac": metrics["vegetation_frac"],
        }
        with open(scene_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    return (f"  [{clip_name}] window {w_idx} [{split}]: {len(patches[0][0])} frames, "
            f"{len(patches)} patch(es) written")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def collect_tasks(frames_dir, clip_name, args):
    frame_paths = sorted(Path(frames_dir).glob("*.png"))
    if len(frame_paths) < args.burst_size:
        print(f"  skip {clip_name}: only {len(frame_paths)} frames, need >= {args.burst_size}")
        return []

    n_windows = 1 + (len(frame_paths) - args.burst_size) // args.stride
    val_start_window = int(n_windows * (1 - args.val_fraction))

    args_dict = {
        "ecc_iterations": args.ecc_iterations, "ecc_eps": args.ecc_eps,
        "min_valid_frames": args.min_valid_frames, "align_max_dim": args.align_max_dim,
        "patch_size": args.patch_size, "patches_per_window": args.patches_per_window,
        "border_margin": args.border_margin, "outdir": args.outdir, "sr_factor": args.sr_factor,
        "min_laplacian_var": args.min_laplacian_var,
        "max_vegetation_frac": args.max_vegetation_frac,
        "texture_max_attempts": args.texture_max_attempts,
    }

    tasks = []
    for w_idx in range(n_windows):
        start = w_idx * args.stride
        window_paths = frame_paths[start:start + args.burst_size]
        split = "val" if w_idx >= val_start_window else "train"
        tasks.append((clip_name, w_idx, window_paths, split, args_dict, args.seed + w_idx))
    return tasks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames_dir", type=str, required=True,
                     help="directory of extracted frames for ONE clip (from extract_frames.py), "
                          "or a parent directory containing multiple clip subfolders with --multi_clip")
    ap.add_argument("--multi_clip", action="store_true",
                     help="treat --frames_dir as a parent of per-clip subfolders")
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--burst_size", type=int, default=14)
    ap.add_argument("--stride", type=int, default=None,
                     help="window step; default = burst_size // 2 (50%% overlap)")
    ap.add_argument("--sr_factor", type=int, default=4)
    ap.add_argument("--patch_size", type=int, default=384, help="HR patch size in pixels")
    ap.add_argument("--patches_per_window", type=int, default=2)
    ap.add_argument("--border_margin", type=int, default=8,
                     help="extra inset (px) from the common valid region, beyond mask intersection")
    ap.add_argument("--min_valid_frames", type=int, default=None,
                     help="drop a window if fewer than this many frames align; default = burst_size // 2")
    ap.add_argument("--val_fraction", type=float, default=0.1,
                     help="fraction of each clip's windows (temporally last) reserved for val")
    ap.add_argument("--align_max_dim", type=int, default=640,
                     help="downscale frames to this many px (long side) for alignment estimation "
                          "only; the homography is rescaled and applied at full resolution. This is "
                          "the main speed knob -- ECC at native 4K/5.3K is ~20-50x slower for "
                          "essentially no accuracy gain, since output is downsampled by sr_factor "
                          "anyway. Raise it only if alignment quality looks poor at the default.")
    ap.add_argument("--ecc_iterations", type=int, default=200)
    ap.add_argument("--ecc_eps", type=float, default=1e-6)
    ap.add_argument("--min_laplacian_var", type=float, default=15.0,
                     help="minimum Laplacian variance (texture/focus measure) a candidate patch's "
                          "reference crop must have to be kept. Patches below this look flat/"
                          "low-detail -- sky, still water, blank walls, out-of-focus background. "
                          "0 (default) disables this filter. Flat regions typically fall well under "
                          "~20-30; textured regions run into the hundreds+, but tune to your footage "
                          "-- audit the laplacian_var values written to meta.json on a trial run "
                          "before locking in a threshold.")
    ap.add_argument("--max_vegetation_frac", type=float, default=0.70,
                     help="reject a candidate patch if more than this fraction of its reference "
                          "crop looks green/foliage-colored (rough HSV heuristic for grass/trees/"
                          "leaves). 1.0 (default) disables this filter. NOTE: grass/foliage is often "
                          "HIGH texture (lots of fine edges), so --min_laplacian_var alone will NOT "
                          "catch it -- this is a separate color-based heuristic, and it will also "
                          "flag other green content (painted walls, green fabric, etc.). Try 0.5-0.7 "
                          "as a starting point.")
    ap.add_argument("--texture_max_attempts", type=int, default=25,
                     help="random resample attempts per patch slot before giving up on that slot, "
                          "when --min_laplacian_var and/or --max_vegetation_frac are active")
    ap.add_argument("--workers", type=int, default=1,
                     help="parallel worker processes (windows are independent). Try os.cpu_count().")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.stride is None:
        args.stride = max(1, args.burst_size // 2)
    if args.min_valid_frames is None:
        args.min_valid_frames = max(2, args.burst_size // 2)

    if args.multi_clip:
        clip_dirs = sorted(d for d in Path(args.frames_dir).iterdir() if d.is_dir())
    else:
        clip_dirs = [Path(args.frames_dir)]

    all_tasks = []
    for clip_dir in clip_dirs:
        all_tasks.extend(collect_tasks(clip_dir, clip_dir.name, args))

    filter_note = ""
    if args.min_laplacian_var > 0.0 or args.max_vegetation_frac < 1.0:
        filter_note = (f", min_laplacian_var={args.min_laplacian_var}, "
                        f"max_vegetation_frac={args.max_vegetation_frac}, "
                        f"texture_max_attempts={args.texture_max_attempts}")

    print(f"{len(all_tasks)} window(s) queued across {len(clip_dirs)} clip(s), "
          f"align_max_dim={args.align_max_dim}, workers={args.workers}{filter_note}")

    n_scenes_written = 0
    if args.workers <= 1:
        for task in all_tasks:
            print(process_window(task))
            n_scenes_written += 1
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(process_window, task) for task in all_tasks]
            for fut in as_completed(futures):
                print(fut.result())
                n_scenes_written += 1

    print(f"\nDone. {n_scenes_written} window(s) processed under {args.outdir}/{{train,val}}/")


if __name__ == "__main__":
    main()