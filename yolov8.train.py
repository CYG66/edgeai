from pathlib import Path
import platform
import argparse

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
DATA_CFG = BASE_DIR / "yolo-bvn.yaml"
DEFAULT_WEIGHTS = BASE_DIR / "yolov8n.pt"
DEFAULT_PROJECT_DIR = BASE_DIR / "runs" / "detect"


def is_arm_edge_device() -> bool:
    machine = platform.machine().lower()
    return any(token in machine for token in ("arm", "aarch64"))


def resolve_run_name(project_dir: Path, requested_name: str | None = None) -> str:
    if requested_name:
        return requested_name

    base_name = "train"
    candidate = project_dir / base_name
    if not candidate.exists():
        return base_name

    suffix = 1
    while (project_dir / f"{base_name}{suffix}").exists():
        suffix += 1
    return f"{base_name}{suffix}"


def build_train_args(data_cfg: Path, run_name: str | None = None) -> dict:
    # The device in the competition slide is an 8 GB ARM edge platform with NPU.
    # Ultralytics training does not use the NPU directly, so we use CPU-safe settings
    # on edge hardware and keep a stronger config for desktop development machines.
    edge_mode = is_arm_edge_device()
    project_dir = DEFAULT_PROJECT_DIR
    project_dir.mkdir(parents=True, exist_ok=True)
    resolved_name = resolve_run_name(project_dir, run_name)

    # Desktop tuning is allowed to spend more compute to improve final accuracy,
    # while edge settings stay conservative so they can still run on ARM devices.
    desktop_workers = 6
    desktop_epochs = 280
    desktop_batch = 4
    desktop_imgsz = 768
    desktop_patience = 90
    desktop_lr0 = 4e-5
    desktop_translate = 0.02
    desktop_scale = 0.20
    desktop_hsv_s = 0.28
    desktop_hsv_v = 0.18
    desktop_fliplr = 0.20
    desktop_mosaic = 0.08
    desktop_close_mosaic = 30
    desktop_nbs = 64
    desktop_freeze = 8
    desktop_box = 8.0
    desktop_cls = 0.7
    desktop_dfl = 1.7
    desktop_cache = False
    desktop_multi_scale = False

    args = {
        "data": str(data_cfg),
        "workers": 1 if edge_mode else desktop_workers,
        "epochs": 150 if edge_mode else desktop_epochs,
        "batch": 2 if edge_mode else desktop_batch,
        "imgsz": 512 if edge_mode else desktop_imgsz,
        "device": "cpu" if edge_mode else "cuda:0",
        "amp": False if edge_mode else True,
        "patience": 40 if edge_mode else desktop_patience,
        "lr0": 6e-5 if edge_mode else desktop_lr0,
        "resume": False,
        "optimizer": "AdamW",
        "seed": 0,
        "exist_ok": False,
        "pretrained": True,
        "single_cls": False,
        "deterministic": True,
        "lrf": 0.1,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        # For small datasets, keep augmentation conservative to avoid damaging defect patterns.
        "hsv_h": 0.01,
        "hsv_s": 0.25 if edge_mode else desktop_hsv_s,
        "hsv_v": 0.15 if edge_mode else desktop_hsv_v,
        "degrees": 0.0,
        "translate": 0.03 if edge_mode else desktop_translate,
        "scale": 0.12 if edge_mode else desktop_scale,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.15 if edge_mode else desktop_fliplr,
        "mosaic": 0.05 if edge_mode else desktop_mosaic,
        "close_mosaic": 20 if edge_mode else desktop_close_mosaic,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "cos_lr": True,
        "rect": True if edge_mode else False,
        "val": True,
        "save": True,
        "save_period": -1,
        "plots": True,
        "cache": False if edge_mode else desktop_cache,
        "fraction": 1.0,
        "nbs": 8 if edge_mode else desktop_nbs,
        "dropout": 0.0,
        # Freeze more backbone layers on small datasets to improve transfer stability.
        "freeze": 16 if edge_mode else desktop_freeze,
        "profile": False,
        "box": 7.5 if edge_mode else desktop_box,
        "cls": 0.5 if edge_mode else desktop_cls,
        "dfl": 1.5 if edge_mode else desktop_dfl,
        "iou": 0.7,
        "conf": 0.25,
        "max_det": 100 if edge_mode else 300,
        "multi_scale": False if edge_mode else desktop_multi_scale,
        "project": str(project_dir),
        "name": resolved_name,
    }

    return args


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on the BVN defect dataset.")
    parser.add_argument("--data", type=Path, default=DATA_CFG)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    train_args = build_train_args(args.data, args.name)
    print("Training configuration:")
    for key in ("data", "device", "imgsz", "batch", "epochs", "workers", "optimizer", "freeze", "name"):
        print(f"  {key}: {train_args[key]}")
    model.train(**train_args)


if __name__ == "__main__":
    main()
