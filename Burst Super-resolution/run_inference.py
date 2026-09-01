#!/usr/bin/env python3
"""Ad-hoc burst inference: pick a specific frame (as burst center) from an
already-extracted frames directory OR directly from a raw video file, then
either run the trained Option-B Burstormer on ONE tile, or on a full GRID of
overlapping tiles covering the center of the frame -- no build_bursts.py
dataset needs to exist first.

Two input modes (mutually exclusive):
  --frames_dir <dir>     Frames already extracted by extract_frames.py
                          (frame_XXXXXX.png). --center_frame selects which
                          one is the burst's reference/center frame, by
                          index (0-based, into the sorted file list) or by
                          exact filename.
  --video <path.mp4>     Raw clip -- burst_size frames are extracted ON THE
                          FLY around --center_time (or --center_frame_number,
                          the raw decoded frame index in the source video),
                          spaced by --every (matches extract_frames.py's
                          convention). Note: the timestamp seek is
                          approximate, not guaranteed frame-exact --
                          --center_frame_number is more precise if you
                          already know the exact frame index.

Two tiling modes (mutually exclusive):
  (default)               ONE tile. --tile_x/--tile_y pick its top-left
                          corner (full-res reference-frame pixel coords);
                          omit both to auto-center it in the common valid
                          (non-black-border) aligned region.
  --tile_grid              ALL tiles covering the center of the frame (the
                          common valid aligned region, optionally narrowed
                          by --region_size), stepped so adjacent tiles
                          overlap by --tile_overlap pixels ("margin
                          padding"), so the stitched result has no hard
                          seams. Writes each tile's outputs individually
                          AND a blended full-region mosaic.

Usage:
    # single tile, from already-extracted frames
    python run_inference_on_frame.py \\
        --frames_dir frames/GX010001 --center_frame 340 \\
        --weights ./logs/Track_1_GoPro/saved_model/last.ckpt

    # full tile grid covering the center of the frame, slight overlap
    python run_inference_on_frame.py \\
        --frames_dir frames/GX010001 --center_frame 340 \\
        --weights ./logs/Track_1_GoPro/saved_model/last.ckpt \\
        --tile_grid --tile_overlap 32

    # tile grid restricted to a 1500x1000 region centered in the frame,
    # from a raw video instead of pre-extracted frames
    python run_inference_on_frame.py \\
        --video raw_clips/GX010001.MP4 --center_time 00:00:12.500 \\
        --weights ./logs/Track_1_GoPro/saved_model/last.ckpt \\
        --tile_grid --tile_overlap 48 --region_size 1500x1000

Output (single-tile mode, written to --result_dir):
    reference_native.png   native-res reference-frame crop (pseudo-GT)
    pred.png                 network's SR output
    bicubic.png               naive baseline for comparison
    comparison.png            bicubic | Burstormer | HR GT, labeled, side by side
    lr/im_00.png ... im_NN.png   the actual LR burst fed to the network
    meta.json                    run parameters + PSNR summary

Output (--tile_grid mode, written to --result_dir):
    mosaic_pred.png          all tiles' SR outputs, overlap-blended together
    mosaic_reference.png     same blending applied to the native reference
    mosaic_bicubic.png       same blending applied to the bicubic baseline
    mosaic_comparison.png    bicubic | Burstormer | HR GT mosaics, side by side
    tiles/tile_r{row}_c{col}_x{px}_y{py}/
        reference_native.png, pred.png, bicubic.png, comparison.png, lr/im_XX.png   (per tile)
    results.csv               per-tile PSNR (net vs. bicubic)
    meta.json                     grid parameters + mean PSNR summary
"""
import argparse
import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from Network_option_b import Burstormer
from utils.metrics import PSNR


# --------------------------------------------------------------------------- #
# Alignment (duplicated from build_bursts.py so this script is self-contained
# and doesn't depend on build_bursts.py being importable from this directory)
# --------------------------------------------------------------------------- #

def resize_for_alignment(img, max_dim):
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale >= 1.0:
        return img, 1.0
    small = cv2.resize(img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                        interpolation=cv2.INTER_AREA)
    return small, scale


def estimate_homography_ecc(ref_gray_f32, mov_gray_f32, iterations, eps):
    warp_matrix = np.eye(3, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, eps)
    try:
        _, warp_matrix = cv2.findTransformECC(
            ref_gray_f32, mov_gray_f32, warp_matrix, cv2.MOTION_HOMOGRAPHY, criteria, None, 5
        )
    except cv2.error:
        return None
    try:
        return np.linalg.inv(warp_matrix).astype(np.float64)
    except np.linalg.LinAlgError:
        return None


def estimate_homography_orb(ref_gray_u8, mov_gray_u8, min_matches=30):
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
    return H.astype(np.float64)


def rescale_homography(H_small, scale):
    if scale >= 1.0:
        return H_small.astype(np.float32)
    T = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)
    H_full = np.linalg.inv(T) @ H_small @ T
    return H_full.astype(np.float32)


def align_frame(ref_bgr, mov_bgr, ecc_iterations, ecc_eps, align_max_dim):
    h, w = ref_bgr.shape[:2]
    ref_small, scale = resize_for_alignment(ref_bgr, align_max_dim)
    mov_small, _ = resize_for_alignment(mov_bgr, align_max_dim)

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
    warped = cv2.warpPerspective(mov_bgr, H_full, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.warpPerspective(np.full((h, w), 255, dtype=np.uint8), H_full, (w, h),
                                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped, mask, method


def common_valid_bbox(masks, border_margin):
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


# --------------------------------------------------------------------------- #
# Frame sourcing
# --------------------------------------------------------------------------- #

def load_burst_from_frames_dir(frames_dir, center_frame, burst_size):
    frame_paths = sorted(Path(frames_dir).glob("*.png"))
    if not frame_paths:
        raise RuntimeError(f"No .png frames found in {frames_dir}")

    if not center_frame.lstrip("-").isdigit():
        matches = [p for p in frame_paths if p.name == center_frame]
        if not matches:
            raise RuntimeError(f"'{center_frame}' not found in {frames_dir}")
        center_idx = frame_paths.index(matches[0])
    else:
        center_idx = int(center_frame)

    half = burst_size // 2
    start = center_idx - half
    end = start + burst_size
    if start < 0 or end > len(frame_paths):
        raise RuntimeError(
            f"Requested burst window [{start}:{end}] falls outside available frames "
            f"[0:{len(frame_paths)}] in {frames_dir}. Pick a --center_frame further "
            f"from the start/end of the clip, or reduce --burst_size."
        )
    window_paths = frame_paths[start:end]
    frames_bgr = [cv2.imread(str(p)) for p in window_paths]
    if any(f is None for f in frames_bgr):
        raise RuntimeError("One or more frames in the window failed to load.")
    return frames_bgr, [p.name for p in window_paths], frame_paths[center_idx].name


def extract_burst_from_video(video_path, center_time, center_frame_number, every, burst_size, tmpdir):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg/ffprobe not found on PATH.")

    if not Path(video_path).exists():
        raise RuntimeError(f"--video path does not exist: {video_path}\n"
                            f"(Reminder: on WSL, paths use forward slashes and start with /mnt/c/... "
                            f"-- not Windows-style \\mnt\\c\\...)")

    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "0", "-select_streams", "v:0", "-show_entries",
             "stream=r_frame_rate", "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed on {video_path}:\n{e.stderr.strip()}") from e
    num, den = probe.stdout.strip().split("/")
    fps = float(num) / float(den)

    if center_frame_number is not None:
        center_frame_idx = center_frame_number
    else:
        h, m, s = center_time.split(":")
        seconds = int(h) * 3600 + int(m) * 60 + float(s)
        center_frame_idx = int(round(seconds * fps))

    half_span = (burst_size // 2) * every
    start_frame = max(0, center_frame_idx - half_span)
    seek_time = max(0.0, start_frame / fps - 1.0)

    n_frames_to_decode = burst_size * every + every
    vf = f"select='between(n\\,0\\,{n_frames_to_decode - 1})'"

    out_pattern = str(Path(tmpdir) / "raw_%06d.png")
    cmd = [
        "ffmpeg", "-y", "-ss", f"{seek_time:.3f}", "-i", str(video_path),
        "-vf", vf, "-vsync", "vfr", "-pix_fmt", "rgb24", out_pattern,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed extracting from {video_path}:\n{e.stderr.strip()}") from e

    decoded = sorted(Path(tmpdir).glob("raw_*.png"))
    if len(decoded) < burst_size * every:
        raise RuntimeError(
            f"Only decoded {len(decoded)} frames near the requested position -- "
            f"too close to the start/end of the clip for burst_size={burst_size}, every={every}."
        )

    picked = decoded[0::every][:burst_size]
    if len(picked) < burst_size:
        raise RuntimeError("Not enough frames decoded to fill the requested burst -- "
                            "try a smaller --every or --burst_size.")

    frames_bgr = [cv2.imread(str(p)) for p in picked]
    names = [f"video_frame_{start_frame + i * every}" for i in range(burst_size)]
    center_name = names[burst_size // 2]
    return frames_bgr, names, center_name


# --------------------------------------------------------------------------- #
# Tiling / stitching helpers (used only in --tile_grid mode)
# --------------------------------------------------------------------------- #

def generate_tile_grid(bbox, patch_size, overlap):
    """Returns (tile_positions, n_cols, n_rows). tile_positions is a flat
    list of (px, py) top-left corners, row-major, guaranteed to cover the
    whole bbox including its right/bottom edges exactly (the last row/col
    is snapped to the edge rather than overshooting)."""
    x0, y0, bw, bh = bbox
    stride = patch_size - overlap
    if stride <= 0:
        raise ValueError("--tile_overlap must be smaller than --patch_size")

    def axis_positions(start, length):
        if length <= patch_size:
            return [start]
        positions = list(range(start, start + length - patch_size + 1, stride))
        if positions[-1] != start + length - patch_size:
            positions.append(start + length - patch_size)
        return positions

    xs = axis_positions(x0, bw)
    ys = axis_positions(y0, bh)
    tile_positions = [(px, py) for py in ys for px in xs]
    return tile_positions, len(xs), len(ys)


def tile_weight_map(patch_size, overlap, taper_left, taper_right, taper_top, taper_bottom):
    """2D blending weight for ONE tile: ramps 0->1 over `overlap` px, but
    ONLY on the edges flagged True (i.e. edges that actually overlap a
    neighboring tile). Edges that coincide with the outer boundary of the
    whole tiled region get full weight (1) all the way to the edge --
    otherwise that boundary would taper to zero with no neighbor there to
    make up the difference, leaving a darkened/black border on the mosaic."""
    row = np.ones(patch_size, dtype=np.float32)
    if overlap > 0:
        taper = np.linspace(0.0, 1.0, overlap, endpoint=False, dtype=np.float32)
        if taper_left:
            row[:overlap] = taper
        if taper_right:
            row[-overlap:] = taper[::-1]
    col = np.ones(patch_size, dtype=np.float32)
    if overlap > 0:
        taper = np.linspace(0.0, 1.0, overlap, endpoint=False, dtype=np.float32)
        if taper_top:
            col[:overlap] = taper
        if taper_bottom:
            col[-overlap:] = taper[::-1]
    return np.outer(col, row)


class Mosaic:
    """Accumulates overlap-blended tile images into one canvas. Each tile's
    weight map is supplied per-call (it depends on that tile's position in
    the grid -- see tile_weight_map), not shared across all tiles."""
    def __init__(self, bbox, patch_size):
        _, _, bw, bh = bbox
        self.x0, self.y0 = bbox[0], bbox[1]
        self.canvas = np.zeros((bh, bw, 3), dtype=np.float64)
        self.weight_sum = np.zeros((bh, bw), dtype=np.float64)
        self.patch_size = patch_size

    def add(self, px, py, tile_bgr_u8, w2d):
        yy0, xx0 = py - self.y0, px - self.x0
        region = tile_bgr_u8.astype(np.float64) * w2d[..., None]
        self.canvas[yy0:yy0 + self.patch_size, xx0:xx0 + self.patch_size] += region
        self.weight_sum[yy0:yy0 + self.patch_size, xx0:xx0 + self.patch_size] += w2d

    def finalize(self):
        safe_weight = np.maximum(self.weight_sum, 1e-6)[..., None]
        return np.clip(self.canvas / safe_weight, 0, 255).astype(np.uint8)


def make_comparison_image(panels, labels, label_height=32, separator_width=4,
                           separator_color=(200, 200, 200), label_bg=(0, 0, 0), label_fg=(255, 255, 255)):
    """Horizontally concatenates same-size BGR images into one comparison
    PNG, each with a labeled banner on top, separated by a thin divider."""
    h, w = panels[0].shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, w / 700.0)
    thickness = 1 if font_scale < 1.0 else 2

    labeled_panels = []
    for img, label in zip(panels, labels):
        banner = np.full((label_height, w, 3), label_bg, dtype=np.uint8)
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        tx = max(0, (w - text_w) // 2)
        ty = (label_height + text_h) // 2
        cv2.putText(banner, label, (tx, ty), font, font_scale, label_fg, thickness, cv2.LINE_AA)
        labeled_panels.append(np.vstack([banner, img]))

    sep = np.full((labeled_panels[0].shape[0], separator_width, 3), separator_color, dtype=np.uint8)
    pieces = []
    for i, panel in enumerate(labeled_panels):
        pieces.append(panel)
        if i < len(labeled_panels) - 1:
            pieces.append(sep)
    return np.hstack(pieces)


# --------------------------------------------------------------------------- #
# Shared inference step (one tile in, pred/reference/bicubic + PSNR out)
# --------------------------------------------------------------------------- #

def run_one_tile(model, psnr_fn, aligned, px, py, patch_size, sr_factor):
    crop_frames = [frame[py:py + patch_size, px:px + patch_size] for frame in aligned]
    reference_native = crop_frames[0]  # aligned[0] is always the reference/center frame

    lr_size = patch_size // sr_factor
    lr_frames = [cv2.resize(f, (lr_size, lr_size), interpolation=cv2.INTER_AREA) for f in crop_frames]

    def to_tensor01(bgr_u8):
        rgb = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(rgb).permute(2, 0, 1)

    burst_t = torch.stack([to_tensor01(f) for f in lr_frames], dim=0).unsqueeze(0).cuda()

    with torch.no_grad():
        pred = model(burst_t).clamp(0.0, 1.0)

    ref_native_t = to_tensor01(reference_native).unsqueeze(0).cuda()
    net_psnr = psnr_fn(pred, ref_native_t).item()

    ref_lr_t = burst_t[:, 0]
    bicubic = F.interpolate(ref_lr_t, size=ref_native_t.shape[-2:], mode="bicubic", align_corners=False).clamp(0.0, 1.0)
    bicubic_psnr = psnr_fn(bicubic, ref_native_t).item()

    def to_bgr_u8(t):
        arr = (t.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0) * 255.0).cpu().numpy().astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    return {
        "reference_native": reference_native,
        "pred_bgr": to_bgr_u8(pred),
        "bicubic_bgr": to_bgr_u8(bicubic),
        "lr_frames": lr_frames,
        "net_psnr": net_psnr,
        "bicubic_psnr": bicubic_psnr,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames_dir", type=str, help="directory of already-extracted frames (extract_frames.py output)")
    src.add_argument("--video", type=str, help="raw video file -- burst frames extracted on the fly")

    ap.add_argument("--center_frame", type=str, default=None,
                     help="[--frames_dir mode] index (0-based) or exact filename of the frame to use as burst center")
    ap.add_argument("--center_time", type=str, default=None, help="[--video mode] HH:MM:SS[.ms] burst center timestamp")
    ap.add_argument("--center_frame_number", type=int, default=None,
                     help="[--video mode] raw decoded frame number instead of --center_time")
    ap.add_argument("--every", type=int, default=2, help="[--video mode] frame spacing, matches extract_frames.py's --every")

    ap.add_argument("--weights", required=True, type=str, help="trained .ckpt path")
    ap.add_argument("--burst_size", type=int, default=14)
    ap.add_argument("--sr_factor", type=int, default=4)
    ap.add_argument("--patch_size", type=int, default=384)

    # single-tile options (ignored when --tile_grid is set)
    ap.add_argument("--tile_x", type=int, default=None, help="[single-tile mode] patch top-left x, full-res coords")
    ap.add_argument("--tile_y", type=int, default=None, help="[single-tile mode] patch top-left y, full-res coords")

    # tile-grid options
    ap.add_argument("--tile_grid", action="store_true",
                     help="tile the whole center of the frame with overlapping patches instead of just one")
    ap.add_argument("--tile_overlap", type=int, default=32,
                     help="[--tile_grid mode] overlap in pixels between adjacent tiles -- the 'margin padding' "
                          "used both to guarantee full coverage with no gaps and to blend away seams in the "
                          "stitched mosaic. Must be smaller than --patch_size.")
    ap.add_argument("--region_size", type=str, default=None,
                     help="[--tile_grid mode] optional 'WxH' -- restrict the grid to a WxH region centered "
                          "within the valid aligned area, instead of tiling the whole thing (useful to bound "
                          "runtime/disk on a large frame). Omit to tile the entire valid center region.")

    ap.add_argument("--border_margin", type=int, default=8)
    ap.add_argument("--align_max_dim", type=int, default=640)
    ap.add_argument("--ecc_iterations", type=int, default=200)
    ap.add_argument("--ecc_eps", type=float, default=1e-6)
    ap.add_argument("--min_valid_frames", type=int, default=None)
    ap.add_argument("--result_dir", type=str, default="./frame_inference_result")
    args = ap.parse_args()

    if args.min_valid_frames is None:
        args.min_valid_frames = max(2, args.burst_size // 2)

    # ---- load raw burst frames ----
    tmpdir_ctx = None
    if args.frames_dir:
        if args.center_frame is None:
            raise SystemExit("--center_frame is required with --frames_dir")
        frames_bgr, source_names, center_name = load_burst_from_frames_dir(
            args.frames_dir, args.center_frame, args.burst_size
        )
        clip_name = Path(args.frames_dir).name
    else:
        if args.center_time is None and args.center_frame_number is None:
            raise SystemExit("--center_time or --center_frame_number is required with --video")
        tmpdir_ctx = tempfile.TemporaryDirectory()
        frames_bgr, source_names, center_name = extract_burst_from_video(
            args.video, args.center_time, args.center_frame_number, args.every,
            args.burst_size, tmpdir_ctx.name,
        )
        clip_name = Path(args.video).stem

    print(f"Loaded {len(frames_bgr)} frame(s) from {'--frames_dir' if args.frames_dir else '--video'}, "
          f"center frame: {center_name}")

    # ---- align burst to its center frame (shared across all tiles) ----
    result = build_window(frames_bgr, args.ecc_iterations, args.ecc_eps, args.min_valid_frames, args.align_max_dim)
    if result is None:
        raise SystemExit(f"Not enough frames aligned (need >= {args.min_valid_frames}). "
                          f"Try a smaller --align_max_dim, a burst from steadier footage, or a smaller --burst_size.")
    aligned, masks, ref_idx_in_window, methods = result
    print(f"Aligned {len(aligned)}/{len(frames_bgr)} frames "
          f"({sum(1 for m in methods if m == 'ecc')} ecc, {sum(1 for m in methods if m == 'orb')} orb, "
          f"1 reference, {len(frames_bgr) - len(aligned)} dropped)")

    if tmpdir_ctx is not None:
        tmpdir_ctx.cleanup()

    bbox = common_valid_bbox(masks, args.border_margin)
    if bbox is None:
        raise SystemExit("No common valid region across the aligned burst -- alignment likely failed badly.")
    x0, y0, bw, bh = bbox
    if bw < args.patch_size or bh < args.patch_size:
        raise SystemExit(f"Common valid region ({bw}x{bh}) is smaller than --patch_size={args.patch_size}.")

    # ---- load model once, reused for every tile ----
    model = Burstormer.load_from_checkpoint(args.weights)
    model.cuda()
    model.eval()
    psnr_fn = PSNR(boundary_ignore=40)

    out_dir = Path(args.result_dir)

    if args.tile_grid:
        # -------------------- tile-grid mode -------------------- #
        if args.region_size:
            rw, rh = (int(v) for v in args.region_size.lower().split("x"))
            rw, rh = min(rw, bw), min(rh, bh)
            x0, y0, bw, bh = x0 + (bw - rw) // 2, y0 + (bh - rh) // 2, rw, rh
            if bw < args.patch_size or bh < args.patch_size:
                raise SystemExit(f"--region_size {args.region_size} is smaller than --patch_size={args.patch_size}.")

        bbox = (x0, y0, bw, bh)
        tile_positions, n_cols, n_rows = generate_tile_grid(bbox, args.patch_size, args.tile_overlap)
        print(f"Tile grid: {n_cols} x {n_rows} = {len(tile_positions)} tile(s), "
              f"covering region ({x0},{y0}) {bw}x{bh}, overlap={args.tile_overlap}px")

        pred_mosaic = Mosaic(bbox, args.patch_size)
        ref_mosaic = Mosaic(bbox, args.patch_size)
        bicubic_mosaic = Mosaic(bbox, args.patch_size)

        tiles_dir = out_dir / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        rows = []

        for i, (px, py) in enumerate(tile_positions):
            row, col = i // n_cols, i % n_cols
            # Only taper edges that actually border a neighboring tile --
            # edges on the outer boundary of the whole region stay full weight.
            w2d = tile_weight_map(
                args.patch_size, args.tile_overlap,
                taper_left=(col > 0), taper_right=(col < n_cols - 1),
                taper_top=(row > 0), taper_bottom=(row < n_rows - 1),
            )
            r = run_one_tile(model, psnr_fn, aligned, px, py, args.patch_size, args.sr_factor)

            pred_mosaic.add(px, py, r["pred_bgr"], w2d)
            ref_mosaic.add(px, py, r["reference_native"], w2d)
            bicubic_mosaic.add(px, py, r["bicubic_bgr"], w2d)

            tile_name = f"tile_r{row:02d}_c{col:02d}_x{px}_y{py}"
            tile_dir = tiles_dir / tile_name
            (tile_dir / "lr").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(tile_dir / "reference_native.png"), r["reference_native"])
            cv2.imwrite(str(tile_dir / "pred.png"), r["pred_bgr"])
            cv2.imwrite(str(tile_dir / "bicubic.png"), r["bicubic_bgr"])
            comparison = make_comparison_image(
                [r["bicubic_bgr"], r["pred_bgr"], r["reference_native"]],
                ["Bicubic", "Burstormer", "HR GT"],
            )
            cv2.imwrite(str(tile_dir / "comparison.png"), comparison)
            for k, f in enumerate(r["lr_frames"]):
                cv2.imwrite(str(tile_dir / "lr" / f"im_{k:02d}.png"), f)

            print(f"  [{i + 1}/{len(tile_positions)}] {tile_name}: "
                  f"net_psnr={r['net_psnr']:.2f} dB, bicubic_psnr={r['bicubic_psnr']:.2f} dB")

            rows.append({
                "tile": tile_name, "x": px, "y": py,
                "net_psnr": round(r["net_psnr"], 3), "bicubic_psnr": round(r["bicubic_psnr"], 3),
            })

        out_dir.mkdir(parents=True, exist_ok=True)
        pred_final = pred_mosaic.finalize()
        ref_final = ref_mosaic.finalize()
        bicubic_final = bicubic_mosaic.finalize()
        cv2.imwrite(str(out_dir / "mosaic_pred.png"), pred_final)
        cv2.imwrite(str(out_dir / "mosaic_reference.png"), ref_final)
        cv2.imwrite(str(out_dir / "mosaic_bicubic.png"), bicubic_final)
        mosaic_comparison = make_comparison_image(
            [bicubic_final, pred_final, ref_final], ["Bicubic", "Burstormer", "HR GT"]
        )
        cv2.imwrite(str(out_dir / "mosaic_comparison.png"), mosaic_comparison)

        with open(out_dir / "results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["tile", "x", "y", "net_psnr", "bicubic_psnr"])
            writer.writeheader()
            writer.writerows(rows)

        mean_net = float(np.mean([r["net_psnr"] for r in rows]))
        mean_bicubic = float(np.mean([r["bicubic_psnr"] for r in rows]))

        meta = {
            "clip": clip_name, "center_frame": center_name, "source_frames": source_names,
            "alignment_methods": methods, "burst_size": len(aligned),
            "patch_size": args.patch_size, "sr_factor": args.sr_factor, "tile_overlap": args.tile_overlap,
            "region_xywh": [int(x0), int(y0), int(bw), int(bh)],
            "grid_cols": n_cols, "grid_rows": n_rows, "n_tiles": len(tile_positions),
            "mean_net_psnr": round(mean_net, 3), "mean_bicubic_psnr": round(mean_bicubic, 3),
        }
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\nDone. {len(tile_positions)} tile(s) processed.")
        print(f"Mean net PSNR: {mean_net:.2f} dB | Mean bicubic PSNR: {mean_bicubic:.2f} dB")
        print(f"Mosaics:    {out_dir}/mosaic_{{pred,reference,bicubic}}.png")
        print(f"Comparison: {out_dir}/mosaic_comparison.png")
        print(f"Tiles:      {tiles_dir}/tile_rXX_cXX_x*_y*/ (each with its own comparison.png)")
        print(f"Summary:    {out_dir}/results.csv, {out_dir}/meta.json")

    else:
        # -------------------- single-tile mode (unchanged behavior) -------------------- #
        if args.tile_x is not None and args.tile_y is not None:
            px, py = args.tile_x, args.tile_y
            if px < x0 or py < y0 or px + args.patch_size > x0 + bw or py + args.patch_size > y0 + bh:
                print(f"WARNING: requested tile ({px},{py}) falls partly/fully outside the common "
                      f"valid region ({x0},{y0},{bw}x{bh}) -- clamping to the nearest valid position.")
                px = min(max(px, x0), x0 + bw - args.patch_size)
                py = min(max(py, y0), y0 + bh - args.patch_size)
        else:
            px = x0 + (bw - args.patch_size) // 2
            py = y0 + (bh - args.patch_size) // 2

        print(f"Tile origin: ({px}, {py}), size {args.patch_size}x{args.patch_size}")
        r = run_one_tile(model, psnr_fn, aligned, px, py, args.patch_size, args.sr_factor)

        print(f"net PSNR (vs. native reference crop):     {r['net_psnr']:.2f} dB")
        print(f"bicubic PSNR (vs. native reference crop): {r['bicubic_psnr']:.2f} dB")
        print(f"delta:                                    {r['net_psnr'] - r['bicubic_psnr']:+.2f} dB")

        (out_dir / "lr").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "reference_native.png"), r["reference_native"])
        cv2.imwrite(str(out_dir / "pred.png"), r["pred_bgr"])
        cv2.imwrite(str(out_dir / "bicubic.png"), r["bicubic_bgr"])
        comparison = make_comparison_image(
            [r["bicubic_bgr"], r["pred_bgr"], r["reference_native"]],
            ["Bicubic", "Burstormer", "HR GT"],
        )
        cv2.imwrite(str(out_dir / "comparison.png"), comparison)
        for k, f in enumerate(r["lr_frames"]):
            cv2.imwrite(str(out_dir / "lr" / f"im_{k:02d}.png"), f)

        meta = {
            "clip": clip_name, "center_frame": center_name, "source_frames": source_names,
            "alignment_methods": methods, "burst_size": len(aligned),
            "patch_size": args.patch_size, "sr_factor": args.sr_factor,
            "tile_origin_xy": [int(px), int(py)],
            "net_psnr_vs_native_ref": round(r["net_psnr"], 3),
            "bicubic_psnr_vs_native_ref": round(r["bicubic_psnr"], 3),
        }
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\nDone. Outputs written to {out_dir}/")


if __name__ == "__main__":
    main()