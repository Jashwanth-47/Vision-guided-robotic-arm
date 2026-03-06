import os
import random
import shutil
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO

# =====================================================
# SETTINGS (EDIT ONLY SOURCE_DIR IF NEEDED)
# =====================================================

SOURCE_DIR = r"C:\Users\Jashwanth\Downloads\Garbage_Dataset_Classification"
OUT_DIR = r"C:\Users\Jashwanth\Downloads\WasteYOLO_FRESH"

TRAIN_RATIO = 0.8
SEED = 42
BOX_MARGIN = 0.05

EPOCHS = 20
IMGSZ = 640
BATCH = 16

# Keep all classes for context but robot acts only on plastic
CLASS_NAMES = ["trash", "plastic", "paper", "metal", "glass", "cardboard"]

# =====================================================
# UTILITIES
# =====================================================

def ensure(p):
    os.makedirs(p, exist_ok=True)

def list_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(exts)
    ]

def write_label(path, cls, margin):
    cx, cy = 0.5, 0.5
    w, h = 1 - 2 * margin, 1 - 2 * margin
    with open(path, "w") as f:
        f.write(f"{cls} {cx} {cy} {w} {h}\n")

# =====================================================
# BUILD DATASET
# =====================================================

def build_dataset():
    # Remove old dataset folder
    if os.path.exists(OUT_DIR):
        print("⚠️ Removing old dataset folder...")
        shutil.rmtree(OUT_DIR)

    for split in ["train", "val"]:
        ensure(os.path.join(OUT_DIR, "images", split))
        ensure(os.path.join(OUT_DIR, "labels", split))

    random.seed(SEED)
    class_to_id = {name: i for i, name in enumerate(CLASS_NAMES)}
    all_items = []

    for cname in CLASS_NAMES:
        class_folder = os.path.join(SOURCE_DIR, cname)
        if not os.path.exists(class_folder):
            raise FileNotFoundError(f"{class_folder} not found")
        images = list_images(class_folder)
        if not images:
            raise RuntimeError(f"No images in {class_folder}")
        for img in images:
            all_items.append((img, class_to_id[cname]))

    random.shuffle(all_items)

    split_index = int(len(all_items) * TRAIN_RATIO)
    train_items = all_items[:split_index]
    val_items = all_items[split_index:]

    def copy_split(items, split):
        img_out = os.path.join(OUT_DIR, "images", split)
        lbl_out = os.path.join(OUT_DIR, "labels", split)
        for img_path, cls in items:
            new_name = f"{cls}_{Path(img_path).name}"
            dst_img = os.path.join(img_out, new_name)
            shutil.copy2(img_path, dst_img)
            label_path = os.path.join(lbl_out, Path(new_name).stem + ".txt")
            write_label(label_path, cls, BOX_MARGIN)

    copy_split(train_items, "train")
    copy_split(val_items, "val")

    # Create YOLO data.yaml
    data_yaml = {
        "path": OUT_DIR.replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }

    yaml_path = os.path.join(OUT_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    print("✅ Dataset ready at:", OUT_DIR)
    return yaml_path

# =====================================================
# TRAINING
# =====================================================

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("🔥 GPU:", torch.cuda.get_device_name(0))
        device = 0
    else:
        print("⚠️ No GPU detected.")
        device = "cpu"

    yaml_path = build_dataset()

    # Use YOLOv8n for fast training
    model = YOLO("yolov8n.pt")

    model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=device,
        workers=4,
        patience=10,
        pretrained=True,
        project=OUT_DIR,
        name="plastic_focus_run",
        exist_ok=True,
        verbose=True
    )

    print("\n✅ Training finished.")
    print("Best model at:")
    print(os.path.join(OUT_DIR, "plastic_focus_run", "weights", "best.pt"))
    print("\n💡 Remember: In your robot code, only act on detections with class == 'plastic'.")