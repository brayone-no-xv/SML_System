{
 "nbformat": 4,
 "nbformat_minor": 0,
 "metadata": {
  "colab": {
   "provenance": [],
   "gpuType": "T4"
  },
  "kernelspec": {
   "name": "python3",
   "display_name": "Python 3"
  },
  "language_info": {
   "name": "python"
  },
  "accelerator": "GPU"
 },
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "sec1_title"
   },
   "source": [
    "# **1. Perkenalan Dataset**\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "sec1_desc"
   },
   "source": [
    "Tahap pertama, Anda harus mencari dan menggunakan dataset dengan ketentuan sebagai berikut:\n",
    "\n",
    "1. **Sumber Dataset**:  \n",
    "   Dataset dapat diperoleh dari berbagai sumber, seperti public repositories (*Kaggle*, *UCI ML Repository*, *Open Data*) atau data primer yang Anda kumpulkan sendiri.\n",
    "\n",
    "**Dataset yang digunakan**: [Sampah Daur Ulang](https://www.kaggle.com/datasets/fathurrahmanalfarizy/sampah-daur-ulang) dari Kaggle.\n",
    "Dataset ini berisi gambar sampah yang dikategorikan ke dalam beberapa kelas untuk klasifikasi menggunakan deep learning (image classification).\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "sec2_title"
   },
   "source": [
    "# **2. Import Library**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "sec2_desc"
   },
   "source": [
    "Pada tahap ini, Anda perlu mengimpor beberapa pustaka (library) Python yang dibutuhkan untuk analisis data dan pembangunan model machine learning atau deep learning."
   ]
  },
  {
   "cell_type": "code",
   "metadata": {
    "id": "sec2_code"
   },
   "source": [
    "from pathlib import Path\n",
    "import os, shutil, zipfile, random, sys, subprocess, inspect\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from PIL import Image\n",
    "import tensorflow as tf\n",
    "\n",
    "try:\n",
    "    from google.colab import files\n",
    "except Exception:\n",
    "    files = None\n",
    "\n",
    "SEED = 42\n",
    "random.seed(SEED)\n",
    "np.random.seed(SEED)\n",
    "tf.random.set_seed(SEED)\n",
    "\n",
    "print(\"TensorFlow version:\", tf.__version__)"
   ],
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "sec3_title"
   },
   "source": [
    "# **3. Memuat Dataset**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "sec3_desc"
   },
   "source": [
    "Pada tahap ini, Anda perlu memuat dataset ke dalam notebook. Jika dataset dalam format CSV, Anda bisa menggunakan pustaka pandas untuk membacanya. Pastikan untuk mengecek beberapa baris awal dataset untuk memahami strukturnya dan memastikan data telah dimuat dengan benar.\n",
    "\n",
    "Jika dataset berada di Google Drive, pastikan Anda menghubungkan Google Drive ke Colab terlebih dahulu. Setelah dataset berhasil dimuat, langkah berikutnya adalah memeriksa kesesuaian data dan siap untuk dianalisis lebih lanjut.\n",
    "\n",
    "Jika dataset berupa unstructured data, silakan sesuaikan dengan format seperti kelas Machine Learning Pengembangan atau Machine Learning Terapan"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {
    "id": "sec3_code"
   },
   "source": [
    "# === Konfigurasi Kaggle ===\n",
    "KAGGLE_JSON_SOURCE = Path.home() / \".kaggle\" / \"kaggle.json\"\n",
    "local_candidates = [Path(\"kaggle.json\"), Path.cwd() / \"kaggle.json\", KAGGLE_JSON_SOURCE]\n",
    "\n",
    "found = None\n",
    "for c in local_candidates:\n",
    "    if c.exists():\n",
    "        found = c\n",
    "        break\n",
    "if found is None:\n",
    "    raise FileNotFoundError(\"kaggle.json tidak ditemukan.\")\n",
    "\n",
    "kaggle_dir = Path.home() / \".kaggle\"\n",
    "kaggle_dir.mkdir(parents=True, exist_ok=True)\n",
    "target = kaggle_dir / \"kaggle.json\"\n",
    "if found.resolve() != target.resolve():\n",
    "    shutil.copy2(found, target)\n",
    "os.chmod(target, 0o600)\n",
    "os.environ[\"KAGGLE_CONFIG_DIR\"] = str(kaggle_dir)\n",
    "\n",
    "# === Download & Extract Dataset ===\n",
    "DATASET = \"fathurrahmanalfarizy/sampah-daur-ulang\"\n",
    "ZIP_PATH = Path(\"sampah-daur-ulang.zip\")\n",
    "extract_dir = Path(\"sampah-daur-ulang\").expanduser()\n",
    "\n",
    "def ensure_kaggle_installed():\n",
    "    try:\n",
    "        import kaggle\n",
    "    except Exception:\n",
    "        subprocess.check_call([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"kaggle\"])\n",
    "\n",
    "ensure_kaggle_installed()\n",
    "\n",
    "kaggle_cli = Path(sys.executable).with_name(\"kaggle\")\n",
    "if not kaggle_cli.exists():\n",
    "    kaggle_cli = Path(shutil.which(\"kaggle\") or \"kaggle\")\n",
    "\n",
    "subprocess.check_call([str(kaggle_cli), \"datasets\", \"download\", \"-d\", DATASET, \"-p\", \".\", \"--force\"])\n",
    "\n",
    "if not ZIP_PATH.exists():\n",
    "    raise FileNotFoundError(f\"Download gagal: {ZIP_PATH} tidak ditemukan.\")\n",
    "\n",
    "extract_dir.mkdir(parents=True, exist_ok=True)\n",
    "with zipfile.ZipFile(ZIP_PATH, \"r\") as zf:\n",
    "    zf.extractall(extract_dir)\n",
    "print(f\"Ekstraksi selesai ke: {extract_dir}\")\n",
    "\n",
    "EXTRACT_DIR = extract_dir.resolve()\n",
    "\n",
    "# Tampilkan isi folder (ringkas)\n",
    "print(\"\\nIsi folder dataset (max 20 item):\")\n",
    "for i, p in enumerate(sorted(extract_dir.rglob(\"*\"))):\n",
    "    rel = p.relative_to(extract_dir).as_posix()\n",
    "    print((\"  \" + rel + \"/\") if p.is_dir() else (\"  \" + rel))\n",
    "    if i >= 19:\n",
    "        break"
   ],
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "sec4_title"
   },
   "source": [
    "# **4. Exploratory Data Analysis (EDA)**\n",
    "\n",
    "Pada tahap ini, Anda akan melakukan **Exploratory Data Analysis (EDA)** untuk memahami karakteristik dataset.\n",
    "\n",
    "Tujuan dari EDA adalah untuk memperoleh wawasan awal yang mendalam mengenai data dan menentukan langkah selanjutnya dalam analisis atau pemodelan."
   ]
  },
  {
   "cell_type": "code",
   "metadata": {
    "id": "sec4_code"
   },
   "source": [
    "# === Konfigurasi ===\n",
    "IMG_SIZE = (224, 224)\n",
    "BATCH_SIZE = 32\n",
    "TRAIN_SPLIT = 0.70\n",
    "VAL_SPLIT = 0.15\n",
    "TEST_SPLIT = 0.15\n",
    "IMG_EXTS = {\".jpg\", \".jpeg\", \".png\"}\n",
    "\n",
    "# === Helper functions ===\n",
    "def has_images(path):\n",
    "    return any(p.is_file() and p.suffix.lower() in IMG_EXTS for p in path.iterdir())\n",
    "\n",
    "def is_class_root(path):\n",
    "    subdirs = [d for d in path.iterdir() if d.is_dir()]\n",
    "    return len(subdirs) >= 2 and sum(1 for d in subdirs if has_images(d)) >= 2\n",
    "\n",
    "def find_named_dir(root, names):\n",
    "    for name in names:\n",
    "        for p in root.rglob(name):\n",
    "            if p.is_dir():\n",
    "                return p\n",
    "    return None\n",
    "\n",
    "def find_class_root(root):\n",
    "    for p in root.rglob(\"*\"):\n",
    "        if p.is_dir() and is_class_root(p):\n",
    "            return p\n",
    "    return None\n",
    "\n",
    "def list_images(base_dir, class_names):\n",
    "    paths, labels = [], []\n",
    "    for idx, cls in enumerate(class_names):\n",
    "        d = Path(base_dir) / cls\n",
    "        if not d.exists():\n",
    "            continue\n",
    "        for p in d.rglob(\"*\"):\n",
    "            if p.is_file() and p.suffix.lower() in IMG_EXTS:\n",
    "                paths.append(str(p))\n",
    "                labels.append(idx)\n",
    "    return np.array(paths), np.array(labels)\n",
    "\n",
    "# === Deteksi struktur dataset ===\n",
    "dataset_root = Path(EXTRACT_DIR)\n",
    "train_dir = find_named_dir(dataset_root, [\"train\", \"training\", \"Train\", \"Training\"])\n",
    "val_dir = find_named_dir(dataset_root, [\"val\", \"valid\", \"validation\", \"Val\", \"Valid\", \"Validation\"])\n",
    "test_dir = find_named_dir(dataset_root, [\"test\", \"testing\", \"Test\", \"Testing\"])\n",
    "\n",
    "class_root = train_dir if train_dir else find_class_root(dataset_root)\n",
    "if class_root is None:\n",
    "    raise RuntimeError(\"Class folders tidak ditemukan.\")\n",
    "\n",
    "class_names = sorted([d.name for d in class_root.iterdir() if d.is_dir()])\n",
    "NUM_CLASSES = len(class_names)\n",
    "print(f\"Jumlah kelas: {NUM_CLASSES}\")\n",
    "print(f\"Nama kelas: {class_names}\")\n",
    "\n",
    "# === Hitung distribusi per kelas ===\n",
    "base_paths, base_labels = list_images(class_root, class_names)\n",
    "print(f\"\\nTotal gambar: {len(base_paths)}\")\n",
    "\n",
    "from collections import Counter\n",
    "dist = Counter(base_labels)\n",
    "print(\"\\nDistribusi per kelas:\")\n",
    "for idx, cls in enumerate(class_names):\n",
    "    print(f\"  {cls}: {dist.get(idx, 0)} gambar\")\n",
    "\n",
    "# === Visualisasi sampel gambar per kelas ===\n",
    "fig, axes = plt.subplots(2, min(NUM_CLASSES, 5), figsize=(15, 6))\n",
    "if NUM_CLASSES < 5:\n",
    "    axes = axes if NUM_CLASSES > 1 else [[axes[0]], [axes[1]]]\n",
    "for idx, cls in enumerate(class_names[:5]):\n",
    "    cls_imgs = [p for p, l in zip(base_paths, base_labels) if l == idx]\n",
    "    if cls_imgs:\n",
    "        img = Image.open(random.choice(cls_imgs)).convert(\"RGB\")\n",
    "        axes[0][idx].imshow(img)\n",
    "        axes[0][idx].set_title(cls, fontsize=10)\n",
    "        axes[0][idx].axis(\"off\")\n",
    "        # Show second sample\n",
    "        img2 = Image.open(random.choice(cls_imgs)).convert(\"RGB\")\n",
    "        axes[1][idx].imshow(img2)\n",
    "        axes[1][idx].axis(\"off\")\n",
    "plt.suptitle(\"Sampel Gambar per Kelas\", fontsize=14)\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "# === Distribusi ukuran gambar ===\n",
    "sample_sizes = []\n",
    "for p in random.sample(list(base_paths), min(200, len(base_paths))):\n",
    "    try:\n",
    "        img = Image.open(p)\n",
    "        sample_sizes.append(img.size)\n",
    "    except Exception:\n",
    "        pass\n",
    "\n",
    "widths = [s[0] for s in sample_sizes]\n",
    "heights = [s[1] for s in sample_sizes]\n",
    "print(f\"\\nUkuran gambar (sampel {len(sample_sizes)}):\")\n",
    "print(f\"  Width  - min: {min(widths)}, max: {max(widths)}, mean: {np.mean(widths):.0f}\")\n",
    "print(f\"  Height - min: {min(heights)}, max: {max(heights)}, mean: {np.mean(heights):.0f}\")\n",
    "\n",
    "# === Bar chart distribusi kelas ===\n",
    "plt.figure(figsize=(10, 4))\n",
    "counts = [dist.get(i, 0) for i in range(NUM_CLASSES)]\n",
    "plt.bar(class_names, counts, color=\"steelblue\")\n",
    "plt.title(\"Distribusi Jumlah Gambar per Kelas\")\n",
    "plt.xlabel(\"Kelas\")\n",
    "plt.ylabel(\"Jumlah Gambar\")\n",
    "plt.xticks(rotation=45, ha=\"right\")\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ],
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "sec5_title"
   },
   "source": [
    "# **5. Data Preprocessing**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "sec5_desc"
   },
   "source": [
    "Pada tahap ini, data preprocessing adalah langkah penting untuk memastikan kualitas data sebelum digunakan dalam model machine learning.\n",
    "\n",
    "Jika Anda menggunakan data teks, data mentah sering kali mengandung nilai kosong, duplikasi, atau rentang nilai yang tidak konsisten, yang dapat memengaruhi kinerja model. Oleh karena itu, proses ini bertujuan untuk membersihkan dan mempersiapkan data agar analisis berjalan optimal.\n",
    "\n",
    "Berikut adalah tahapan-tahapan yang bisa dilakukan, tetapi **tidak terbatas** pada:\n",
    "1. Menghapus atau Menangani Data Kosong (Missing Values)\n",
    "2. Menghapus Data Duplikat\n",
    "3. Normalisasi atau Standarisasi Fitur\n",
    "4. Deteksi dan Penanganan Outlier\n",
    "5. Encoding Data Kategorikal\n",
    "6. Binning (Pengelompokan Data)\n",
    "\n",
    "Cukup sesuaikan dengan karakteristik data yang kamu gunakan yah. Khususnya ketika kami menggunakan data tidak terstruktur."
   ]
  },
  {
   "cell_type": "code",
   "metadata": {
    "id": "sec5_code"
   },
   "source": [
    "# === Shuffle & Split Data ===\n",
    "idx_arr = np.arange(len(base_paths))\n",
    "np.random.shuffle(idx_arr)\n",
    "base_paths = base_paths[idx_arr]\n",
    "base_labels = base_labels[idx_arr]\n",
    "\n",
    "n_total = len(base_paths)\n",
    "n_train = int(n_total * TRAIN_SPLIT)\n",
    "n_val = int(n_total * VAL_SPLIT)\n",
    "\n",
    "train_paths = base_paths[:n_train]\n",
    "train_labels = base_labels[:n_train]\n",
    "val_paths = base_paths[n_train:n_train + n_val]\n",
    "val_labels = base_labels[n_train:n_train + n_val]\n",
    "test_paths = base_paths[n_train + n_val:]\n",
    "test_labels = base_labels[n_train + n_val:]\n",
    "\n",
    "# Override jika ada folder terpisah\n",
    "if val_dir is not None:\n",
    "    val_paths, val_labels = list_images(val_dir, class_names)\n",
    "if test_dir is not None:\n",
    "    test_paths, test_labels = list_images(test_dir, class_names)\n",
    "\n",
    "print(f\"Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}\")\n",
    "\n",
    "# === Decode & Resize Image ===\n",
    "def decode_img(path, label):\n",
    "    img = tf.io.read_file(path)\n",
    "    img = tf.image.decode_image(img, channels=3, expand_animations=False)\n",
    "    img = tf.image.resize(img, IMG_SIZE)\n",
    "    img = tf.cast(img, tf.float32)\n",
    "    return img, label\n",
    "\n",
    "# === Build tf.data Pipeline ===\n",
    "def build_ds(paths, labels, shuffle=False):\n",
    "    ds = tf.data.Dataset.from_tensor_slices((paths, labels))\n",
    "    if shuffle:\n",
    "        ds = ds.shuffle(min(len(paths), 1000), seed=SEED)\n",
    "    ds = ds.map(decode_img, num_parallel_calls=tf.data.AUTOTUNE)\n",
    "    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)\n",
    "    return ds\n",
    "\n",
    "train_ds = build_ds(train_paths, train_labels, shuffle=True)\n",
    "val_ds = build_ds(val_paths, val_labels, shuffle=False)\n",
    "test_ds = build_ds(test_paths, test_labels, shuffle=False)\n",
    "\n",
    "print(\"\\nDataset pipeline siap:\")\n",
    "print(f\"  class_root : {class_root}\")\n",
    "print(f\"  train_dir  : {train_dir}\")\n",
    "print(f\"  val_dir    : {val_dir}\")\n",
    "print(f\"  test_dir   : {test_dir}\")\n",
    "print(f\"  Classes    : {class_names}\")\n",
    "print(f\"  IMG_SIZE   : {IMG_SIZE}\")\n",
    "print(f\"  BATCH_SIZE : {BATCH_SIZE}\")\n",
    "\n",
    "# === Verifikasi batch pertama ===\n",
    "for images, labels in train_ds.take(1):\n",
    "    print(f\"\\nBatch shape : {images.shape}\")\n",
    "    print(f\"Labels shape: {labels.shape}\")\n",
    "    print(f\"Pixel range : [{images.numpy().min():.1f}, {images.numpy().max():.1f}]\")\n",
    "\n",
    "# === Data Augmentation Layer ===\n",
    "def make_data_augmentation():\n",
    "    return tf.keras.Sequential([\n",
    "        tf.keras.layers.RandomFlip(\"horizontal\"),\n",
    "        tf.keras.layers.RandomRotation(0.15),\n",
    "        tf.keras.layers.RandomZoom(0.15),\n",
    "        tf.keras.layers.RandomContrast(0.2),\n",
    "    ], name=\"data_augmentation\")\n",
    "\n",
    "print(\"\\nPreprocessing selesai. Dataset siap untuk modelling.\")"
   ],
   "execution_count": null,
   "outputs": []
  }
 ]
}