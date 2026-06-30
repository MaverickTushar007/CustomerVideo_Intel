import pathlib, shutil, yaml

# Base directories (relative to this script)
BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "roboflow_data"
OUT_DIR = BASE_DIR / "roboflow_filtered"

# Clean or create output directory
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True)

# Load original data.yaml to discover class indices
with open(DATA_DIR / "data.yaml", "r") as f:
    meta = yaml.safe_load(f)
orig_names = meta.get("names", [])
# Target class names (must match exactly the names in data.yaml)
TARGET_NAMES = {"staff", "customer"}
# Build mapping: original index -> new index (or -1 to discard)
orig_to_new = {}
new_names = []
for idx, name in enumerate(orig_names):
    if name in TARGET_NAMES:
        new_idx = len(new_names)
        orig_to_new[idx] = new_idx
        new_names.append(name)
    else:
        orig_to_new[idx] = -1
if not new_names:
    raise RuntimeError("Dataset does not contain 'staff' or 'customer' classes")

# Helper to process a split (train / val / test)
def process_split(split: str):
    img_src = DATA_DIR / split / "images"
    lbl_src = DATA_DIR / split / "labels"
    img_dst = OUT_DIR / split / "images"
    lbl_dst = OUT_DIR / split / "labels"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)
    for lbl_file in lbl_src.glob("*.txt"):
        # Read original label lines
        with open(lbl_file, "r") as lf:
            lines = lf.readlines()
        kept = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            orig_idx = int(parts[0])
            new_idx = orig_to_new.get(orig_idx, -1)
            if new_idx != -1:
                kept.append(f"{new_idx} {' '.join(parts[1:])}\n")
        if kept:
            # Copy associated image (try common extensions)
            img_path = img_src / (lbl_file.stem + ".jpg")
            if not img_path.is_file():
                for ext in [".jpeg", ".png"]:
                    alt = img_src / (lbl_file.stem + ext)
                    if alt.is_file():
                        img_path = alt
                        break
            if img_path.is_file():
                shutil.copy2(img_path, img_dst / img_path.name)
                with open(lbl_dst / lbl_file.name, "w") as out_f:
                    out_f.writelines(kept)
        # If no kept labels, we skip the image – it contains only irrelevant classes.

for s in ["train", "val", "test"]:
    process_split(s)

# Write new data.yaml for the filtered subset
filtered_yaml = {
    "train": "../train/images",
    "val": "../val/images",
    "test": "../test/images",
    "nc": len(new_names),
    "names": new_names,
    "roboflow": meta.get("roboflow", {}),
}
with open(OUT_DIR / "data.yaml", "w") as f:
    yaml.safe_dump(filtered_yaml, f)

print(f"Filtered dataset created at {OUT_DIR}")
