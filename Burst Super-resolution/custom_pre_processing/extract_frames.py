#!/usr/bin/env python3
"""Extract frames from GoPro footage (4K/5.3K) for MFSR/MISR burst training.

Uses ffmpeg's `select` filter (not `fps=`) so you get exact, deterministic
source frames -- every Nth *decoded* frame -- rather than ffmpeg's fps filter
retiming/duplicating frames to hit a target rate. That matters here: the
"burst diversity" you're relying on downstream is the real motion between
true consecutive frames, and a resampling filter can quietly hand you
duplicated or interpolated frames instead.

Usage:
    # single clip
    python extract_frames.py --input GX010001.MP4 --outdir frames --every 2

    # every clip in a folder
    python extract_frames.py --input_dir raw_clips/ --outdir frames --every 2

    # trim to a window within a clip
    python extract_frames.py --input GX010001.MP4 --outdir frames \
        --every 2 --start 00:00:05 --end 00:00:25
"""
import argparse
import shutil
import subprocess
from pathlib import Path


def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it first "
            "(e.g. `sudo apt install ffmpeg` or `brew install ffmpeg`)."
        )


def extract_clip(input_path: Path, outdir: Path, every: int, start: str, end: str):
    outdir.mkdir(parents=True, exist_ok=True)

    # keep frame indices n where n % every == 0
    vf = f"select=not(mod(n\\,{every}))"

    cmd = ["ffmpeg", "-y"]
    if start:
        cmd += ["-ss", start]
    cmd += ["-i", str(input_path)]
    if end:
        cmd += ["-to", end]
    cmd += [
        "-vf", vf,
        "-vsync", "vfr",       # don't let ffmpeg pad/duplicate frames to match a rate
        "-pix_fmt", "rgb24",
        str(outdir / "frame_%06d.png"),  # lossless -- avoid re-compressing on top of H.264/H.265
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=str, help="single GoPro clip (e.g. GX010001.MP4)")
    src.add_argument("--input_dir", type=str, help="directory of GoPro clips")
    ap.add_argument("--outdir", type=str, required=True, help="root output directory")
    ap.add_argument("--every", type=int, default=12,
                     help="keep every Nth source frame (default: 2). Lower = more motion "
                          "blur risk between neighbors; higher = larger displacement.")
    ap.add_argument("--start", type=str, default=None, help="trim start, e.g. 00:00:05")
    ap.add_argument("--end", type=str, default=None, help="trim end, e.g. 00:00:25")
    ap.add_argument("--ext", type=str, default=".MP4", help="clip extension when using --input_dir")
    args = ap.parse_args()

    check_ffmpeg()
    outroot = Path(args.outdir)

    if args.input:
        clips = [Path(args.input)]
    else:
        clips = sorted(Path(args.input_dir).glob(f"*{args.ext}"))
        if not clips:
            raise RuntimeError(f"No clips matching *{args.ext} found in {args.input_dir}")

    for clip in clips:
        clip_out = outroot / clip.stem
        print(f"[{clip.name}] -> {clip_out}")
        extract_clip(clip, clip_out, args.every, args.start, args.end)

    print(f"\nDone. Frames are under {outroot}/<clip_name>/frame_XXXXXX.png")


if __name__ == "__main__":
    main()