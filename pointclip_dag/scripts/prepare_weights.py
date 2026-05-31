from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-model", default="ViT-L/14")
    parser.add_argument("--download-root", default="weights/clip")
    parser.add_argument("--skip-clip", action="store_true")
    parser.add_argument("--depth-anything-v2", action="store_true")
    parser.add_argument("--depth-anything-encoder", default="vits", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--depth-anything-dir", default="weights/depth_anything_v2")
    parser.add_argument("--dinov2", action="store_true")
    parser.add_argument("--dinov2-encoder", default="vitl14", choices=["vits14", "vitb14", "vitl14"])
    parser.add_argument("--dinov2-dir", default="weights/dinov2")
    args = parser.parse_args()

    if not args.skip_clip:
        root = Path(args.download_root)
        root.mkdir(parents=True, exist_ok=True)
        try:
            import clip
        except Exception as exc:
            raise SystemExit("OpenAI CLIP is not installed. Install with `pip install git+https://github.com/openai/CLIP.git`.") from exc

        model, _ = clip.load(args.clip_model, device="cpu", jit=False, download_root=str(root))
        del model
        print(f"CLIP {args.clip_model} is ready under {root}")

    if args.depth_anything_v2:
        path = _download_depth_anything_v2(args.depth_anything_encoder, Path(args.depth_anything_dir))
        print(f"DepthAnythingV2 {args.depth_anything_encoder} checkpoint is ready: {path}")

    if args.dinov2:
        path = _download_dinov2(args.dinov2_encoder, Path(args.dinov2_dir))
        print(f"DINOv2 {args.dinov2_encoder} image encoder checkpoint is ready: {path}")


def _download_depth_anything_v2(encoder: str, out_dir: Path) -> Path:
    repos = {
        "vits": "depth-anything/Depth-Anything-V2-Small",
        "vitb": "depth-anything/Depth-Anything-V2-Base",
        "vitl": "depth-anything/Depth-Anything-V2-Large",
    }
    filename = f"depth_anything_v2_{encoder}.pth"
    url = f"https://huggingface.co/{repos[encoder]}/resolve/main/{filename}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    print(f"Downloading {url} -> {out_path}")
    urllib.request.urlretrieve(url, out_path)
    return out_path


def _download_dinov2(encoder: str, out_dir: Path) -> Path:
    urls = {
        "vits14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
        "vitb14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
        "vitl14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
    }
    filename = f"dinov2_{encoder}_pretrain.pth"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    print(f"Downloading {urls[encoder]} -> {out_path}")
    urllib.request.urlretrieve(urls[encoder], out_path)
    return out_path


if __name__ == "__main__":
    run()
