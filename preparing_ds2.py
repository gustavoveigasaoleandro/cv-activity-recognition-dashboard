import tensorflow as tf
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter
from Checkpoint.F1checkpoint import F1Checkpoint

# ======= Caminhos =======
FALL_DATASET_DIR = Path("./fall_dataset")
TRAIN_IMAGES_DIR = FALL_DATASET_DIR / "images" / "train"
TRAIN_LABELS_DIR = FALL_DATASET_DIR / "labels" / "train"
VAL_IMAGES_DIR = FALL_DATASET_DIR / "images" / "val"
VAL_LABELS_DIR = FALL_DATASET_DIR / "labels" / "val"

# ======= Definições das classes =======
CLASS_ID_TO_NAME = {
    0: "falling_person",
    1: "lying_person",
    2: "standing_person",
}
CLASS_PRIORITY = [0, 1, 2]  # prioridade ao mapear multi-caixas para um único rótulo
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

# ======= Hiperparâmetros =======
IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS_HEAD = 8
EPOCHS_FT = 12


def _choose_label(label_path: Path) -> str:
    """
    Converte o formato YOLO (múltiplas caixas) em um único rótulo de classificação.
    Dá prioridade para as classes mais críticas (queda > deitado > em pé) para
    garantir que toda imagem seja mapeada para uma única classe.
    """
    label_ids = set()
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                class_id = int(float(parts[0]))
            except (ValueError, IndexError):
                continue
            label_ids.add(class_id)
    for class_id in CLASS_PRIORITY:
        if class_id in label_ids:
            return CLASS_ID_TO_NAME[class_id]
    # fallback: assume classe menos crítica quando não há anotação
    return CLASS_ID_TO_NAME[CLASS_PRIORITY[-1]]


def _iter_images(directory: Path):
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def build_split_df(images_dir: Path, labels_dir: Path) -> pd.DataFrame:
    rows = []
    missing_labels = []
    for img_path in _iter_images(images_dir):
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            missing_labels.append(img_path.name)
            continue
        rows.append({
            "filepath": img_path.resolve().as_posix(),
            "label": _choose_label(label_path),
        })
    if missing_labels:
        print(f"[WARN] {len(missing_labels)} imagens sem label correspondente: {missing_labels[:5]}")
    return pd.DataFrame(rows)


# ======= Carrega splits =======
train_df = build_split_df(TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR)
holdout_df = build_split_df(VAL_IMAGES_DIR, VAL_LABELS_DIR)  # dataset de validação oficial

if train_df.empty:
    raise RuntimeError("Nenhuma imagem encontrada em fall_dataset/images/train")

class_names = sorted(train_df["label"].unique().tolist())
num_classes = len(class_names)
print(f"Classes detectadas: {class_names}")
str2idx = tf.keras.layers.StringLookup(
    vocabulary=class_names, num_oov_indices=0, mask_token=None
)
idx2str = tf.keras.layers.StringLookup(
    vocabulary=str2idx.get_vocabulary(), invert=True, mask_token=None
)

# ======= Split interno treino/val (80/20 estratificado) =======
train_idx, val_idx = train_test_split(
    train_df.index,
    test_size=0.2,
    random_state=42,
    stratify=train_df["label"],
)
train_split = train_df.loc[train_idx].reset_index(drop=True)
val_split = train_df.loc[val_idx].reset_index(drop=True)

# ======= Data augmentation =======
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomContrast(0.1),
])


def load_image(path, label=None, augment_on=False):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img.set_shape((None, None, 3))
    img = tf.image.resize(img, IMG_SIZE)
    if augment_on:
        img = augment(img, training=True)
    img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
    if label is None:
        return img
    y = str2idx(label)
    return img, tf.cast(y, tf.int32)


def make_ds(df: pd.DataFrame, shuffle=False, augment_on=False):
    paths = df["filepath"].tolist()
    if "label" in df.columns:
        labels = df["label"].tolist()
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(lambda p, y: load_image(p, y, augment_on),
                    num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.map(lambda p: load_image(p, None, False),
                    num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(2048)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)


train_ds = make_ds(train_split, shuffle=True, augment_on=True)
val_ds = make_ds(val_split, shuffle=False, augment_on=False)
holdout_ds = make_ds(holdout_df, shuffle=False, augment_on=False)

# ======= Pesos de classe opcional =======
counts = Counter(train_split["label"])
total = sum(counts.values())
class_weights = {
    int(str2idx(tf.constant(label)).numpy()): total / (num_classes * count)
    for label, count in counts.items()
}

# ======= Backbone =======
base_model = tf.keras.applications.EfficientNetV2B0(
    include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3)
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.4)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

# ======= Etapa 1: cabeça =======
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[
        "accuracy",
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3"),
    ],
)

early = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

print("\n=== ETAPA 1: cabeça congelada ===\n")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    callbacks=[early],
    class_weight=class_weights,
    verbose=1,
)

# ======= Etapa 2: fine-tuning =======
base_model.trainable = True
fine_tune_at = int(len(base_model.layers) * 0.7)
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[
        "accuracy",
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3"),
    ],
)

f1_ckpt = F1Checkpoint(val_ds, idx2str, path="./best_by_f2.keras")

print("\n=== ETAPA 2: fine-tuning ===\n")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FT,
    callbacks=[f1_ckpt],
    class_weight=class_weights,
    verbose=1,
)

# ======= Avaliação e predição no holdout oficial =======
print("\n=== AVALIAÇÃO HOLDOUT (split val do dataset) ===\n")
metrics = model.evaluate(holdout_ds, verbose=1)
print({name: value for name, value in zip(model.metrics_names, metrics)})

probs = model.predict(holdout_ds, verbose=1)
pred_idx = probs.argmax(axis=1)
pred_labels = idx2str(tf.constant(pred_idx)).numpy().astype(str)

pred_df = holdout_df.copy()
pred_df['pred_label'] = pred_labels
pred_df.to_csv('fall_holdout_predictions.csv', index=False)
print("Arquivo salvo: fall_holdout_predictions.csv")
