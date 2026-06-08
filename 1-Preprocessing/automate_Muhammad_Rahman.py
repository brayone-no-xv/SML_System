"""
automate_Muhammad_Rahman.py
============================
Automated preprocessing pipeline for image classification
Dataset: Sampah Daur Ulang (Kaggle)

Konversi dari proses eksperimen pada notebook Eksperimen_SML_Muhammad-Rahman.py
menjadi fungsi-fungsi otomatis yang mengembalikan data siap dilatih.

Tahapan:
  1. Download dataset dari Kaggle
  2. Ekstraksi file ZIP
  3. Penemuan struktur kelas secara otomatis
  4. Split dataset menjadi Train / Validation / Test (70/15/15)
  5. Membangun tf.data.Dataset pipeline siap latih

Usage:
    from automate_Muhammad_Rahman import preprocess
    train_ds, val_ds, test_ds, class_names, config = preprocess()
"""

import os
import sys
import shutil
import subprocess
import zipfile
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

# ======================== CONFIGURATION ========================
SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
IMG_EXTS = {".jpg", ".jpeg", ".png"}
DATASET_SLUG = "fathurrahmanalfarizy/sampah-daur-ulang"
ZIP_NAME = "sampah-daur-ulang.zip"
PREPROCESS_DIR = Path(__file__).resolve().parent

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ======================== KAGGLE & DOWNLOAD ========================

def setup_kaggle():
    """Locate and setup kaggle.json credentials."""
    kaggle_dir = Path.home() / ".kaggle"
    target = kaggle_dir / "kaggle.json"

    candidates = [Path("kaggle.json"), Path.cwd() / "kaggle.json", target]
    found = None
    for c in candidates:
        if c.exists():
            found = c
            break

    if found is None:
        return False

    kaggle_dir.mkdir(parents=True, exist_ok=True)
    if found.resolve() != target.resolve():
        shutil.copy2(found, target)

    os.chmod(target, 0o600)
    os.environ["KAGGLE_CONFIG_DIR"] = str(kaggle_dir)
    return True


def download_dataset(output_dir=None):
    """Download dataset ZIP from Kaggle."""
    if output_dir is None:
        output_dir = PREPROCESS_DIR
    if not setup_kaggle():
        raise FileNotFoundError("kaggle.json tidak ditemukan.")

    try:
        import kaggle  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"])

    kaggle_cli = Path(sys.executable).with_name("kaggle")
    if not kaggle_cli.exists():
        kaggle_cli = Path(shutil.which("kaggle") or "kaggle")

    zip_path = Path(output_dir) / ZIP_NAME
    subprocess.check_call([
        str(kaggle_cli), "datasets", "download",
        "-d", DATASET_SLUG, "-p", str(output_dir), "--force",
    ])

    if not zip_path.exists():
        raise FileNotFoundError(f"Download gagal: {zip_path} tidak ditemukan.")
    return zip_path


def extract_dataset(zip_path, extract_dir=None):
    """Extract downloaded ZIP file."""
    if extract_dir is None:
        extract_dir = Path(zip_path).parent / "sampah-daur-ulang"
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print(f"Ekstraksi selesai ke: {extract_dir}")
    return extract_dir.resolve()


# ======================== STRUCTURE DISCOVERY ========================

def has_images(path):
    """Check if directory contains image files."""
    return any(p.is_file() and p.suffix.lower() in IMG_EXTS for p in path.iterdir())


def is_class_root(path):
    """Check if directory is a class root (has >=2 subdirs with images)."""
    subdirs = [d for d in path.iterdir() if d.is_dir()]
    if len(subdirs) < 2:
        return False
    return sum(1 for d in subdirs if has_images(d)) >= 2


def find_class_root(root):
    """Find the directory containing class subdirectories."""
    root = Path(root)
    if is_class_root(root):
        return root
    for p in root.rglob("*"):
        if p.is_dir() and is_class_root(p):
            return p
    return None


def find_named_dir(root, names):
    """Find a directory by name within root."""
    for name in names:
        for p in root.rglob(name):
            if p.is_dir():
                return p
    return None


# ======================== PREPROCESSING ========================

def list_images(base_dir, class_names):
    """Kumpulkan semua path gambar dan label dari direktori kelas."""
    paths, labels = [], []
    for idx, class_name in enumerate(class_names):
        class_dir = Path(base_dir) / class_name
        if not class_dir.exists():
            continue
        for p in class_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                paths.append(str(p))
                labels.append(idx)
    return np.array(paths), np.array(labels)


def split_data(paths, labels):
    """Split data menjadi train, validation, dan test."""
    idx = np.arange(len(paths))
    np.random.shuffle(idx)
    paths, labels = paths[idx], labels[idx]

    n = len(paths)
    n_train = int(n * TRAIN_SPLIT)
    n_val = int(n * VAL_SPLIT)

    splits = {
        "train": (paths[:n_train], labels[:n_train]),
        "val": (paths[n_train:n_train + n_val], labels[n_train:n_train + n_val]),
        "test": (paths[n_train + n_val:], labels[n_train + n_val:]),
    }
    for name, (p, _) in splits.items():
        print(f"  {name}: {len(p)} gambar")
    return splits


def decode_img(path, label):
    """Decode dan resize gambar."""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    return img, label


def build_dataset(paths, labels, shuffle=False, batch_size=BATCH_SIZE):
    """Bangun tf.data.Dataset dari paths dan labels."""
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(min(len(paths), 1000), seed=SEED)
    ds = ds.map(decode_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def make_data_augmentation():
    """Buat layer augmentasi data."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.2),
    ], name="data_augmentation")


def print_dataset_info(class_root, class_names, paths, labels):
    """Print ringkasan informasi dataset."""
    from collections import Counter
    dist = Counter(labels)
    print(f"\nDataset root: {class_root}")
    print(f"Jumlah kelas: {len(class_names)}")
    print(f"Total gambar: {len(paths)}")
    print("Distribusi per kelas:")
    for idx, cls in enumerate(class_names):
        print(f"  {cls}: {dist.get(idx, 0)} gambar")


# ======================== MAIN FUNCTION ========================

def preprocess(data_dir=None, download_data=True):
    """
    Pipeline preprocessing lengkap.

    Parameters
    ----------
    data_dir : str or Path, optional
        Path ke dataset yang sudah diekstrak.
        Jika None dan download_data=True, dataset didownload dari Kaggle.
    download_data : bool
        Apakah perlu download dataset jika data_dir tidak diberikan.

    Returns
    -------
    train_ds : tf.data.Dataset   — Dataset training (sudah di-batch & prefetch)
    val_ds   : tf.data.Dataset   — Dataset validasi
    test_ds  : tf.data.Dataset   — Dataset testing
    class_names : list[str]      — Daftar nama kelas
    config   : dict              — Konfigurasi (IMG_SIZE, BATCH_SIZE, NUM_CLASSES)
    """
    # 1. Data Loading
    if data_dir is None:
        data_dir = PREPROCESS_DIR / "sampah-daur-ulang"
    dataset_root = Path(data_dir).resolve()

    if not dataset_root.exists() or not any(dataset_root.rglob("*.jpg")):
        if download_data:
            zip_path = download_dataset(PREPROCESS_DIR)
            dataset_root = extract_dataset(zip_path, dataset_root)
        else:
            raise ValueError(f"Dataset tidak ditemukan di {dataset_root} dan download_data=False.")

    # 2. Temukan class root
    class_root = find_class_root(dataset_root)
    if class_root is None:
        raise RuntimeError("Tidak dapat menemukan folder kelas. Periksa struktur dataset.")

    # 3. EDA — Discover classes
    class_names = sorted([d.name for d in class_root.iterdir() if d.is_dir()])
    num_classes = len(class_names)
    if num_classes == 0:
        raise RuntimeError("Tidak ada folder kelas ditemukan.")

    # 4. List & split images
    all_paths, all_labels = list_images(class_root, class_names)
    if len(all_paths) == 0:
        raise RuntimeError("Tidak ada gambar ditemukan.")

    print_dataset_info(class_root, class_names, all_paths, all_labels)

    print("Splitting data:")
    splits = split_data(all_paths, all_labels)

    # 5. Build tf.data pipelines
    train_ds = build_dataset(*splits["train"], shuffle=True)
    val_ds = build_dataset(*splits["val"], shuffle=False)
    test_ds = build_dataset(*splits["test"], shuffle=False)

    config = {
        "IMG_SIZE": IMG_SIZE,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_CLASSES": num_classes,
        "SEED": SEED,
    }

    print("\n✅ Preprocessing selesai! Data siap dilatih.")
    return train_ds, val_ds, test_ds, class_names, config


# ======================== ENTRY POINT ========================

if __name__ == "__main__":
    train_ds, val_ds, test_ds, class_names, cfg = preprocess()
    print(f"\nConfig: {cfg}")
    print(f"Classes: {class_names}")
