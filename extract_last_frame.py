"""Pull the last (or first) frame of a rendered clip into ComfyUI\\input\\.

For chaining scenes: clip B's `first_frame` should be clip A's final frame, which
gives pixel continuity instead of hoping the prompt reproduces the character.

Usage
  python extract_last_frame.py H3_ORIGIN_clipA_00001_.mp4
  python extract_last_frame.py clipA.mp4 --name gator_shot3_end
  python extract_last_frame.py clipA.mp4 --first          # grab frame 0 instead

Input may be a bare filename (looked up in ComfyUI\\output\\video\\) or a path.
Output always lands in ComfyUI\\input\\ because that is the only place LoadImage
reads -- output\\ is NOT on its search path.
"""
import argparse
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
OUT_VIDEO = ROOT / "ComfyUI" / "output" / "video"
INPUT_DIR = ROOT / "ComfyUI" / "input"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="clip filename (searched in output/video) or full path")
    ap.add_argument("--name", help="output stem (default: <video stem>_lastframe)")
    ap.add_argument("--first", action="store_true", help="extract frame 0 instead of the last")
    args = ap.parse_args()

    src = pathlib.Path(args.video)
    if not src.exists():
        cand = OUT_VIDEO / args.video
        if not cand.exists():
            sys.exit(f"not found: {args.video}\n  also tried: {cand}")
        src = cand

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not on PATH -- install it or extract the frame manually")

    stem = args.name or f"{src.stem}_{'firstframe' if args.first else 'lastframe'}"
    dst = INPUT_DIR / f"{stem}.png"
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.first:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(src),
               "-frames:v", "1", "-q:v", "1", str(dst), "-y"]
    else:
        # -sseof seeks from the end; 0.15 s back is enough to land on the final frame
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-sseof", "-0.15",
               "-i", str(src), "-update", "1", "-frames:v", "1", "-q:v", "1",
               str(dst), "-y"]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not dst.exists():
        sys.exit(f"ffmpeg failed:\n{r.stderr[:600]}")

    print(f"wrote {dst}")
    print(f"\nIn ComfyUI: un-mute the 'first_frame' LoadImage node and pick:\n"
          f"  {dst.name}\n"
          f"(Ctrl+M toggles mute. You may need to refresh the browser tab for a\n"
          f" newly written file to appear in the LoadImage dropdown.)")


if __name__ == "__main__":
    main()
