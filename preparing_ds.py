import tensorflow as tf
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from Checkpoint.F1checkpoint import F1Checkpoint
# ======= Caminhos =======
TRAIN_CSV = Path("./Human_Action_Recognition/Training_set.csv")
TEST_CSV  = Path("./Human_Action_Recognition/Testing_set.csv")
TRAIN_IMAGES_DIR = Path("./Human_Action_Recognition/train")
TEST_IMAGES_DIR  = Path("./Human_Action_Recognition/test")

# ======= Hiperparâmetros =======
IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS_HEAD = 20
EPOCHS_FT   = 40

# ======= Carrega CSVs =======
train_df = pd.read_csv(TRAIN_CSV)        # filename, label (string)
test_df  = pd.read_csv(TEST_CSV)         # filename

# ======= Vocabulário de classes (ordem fixa) =======
class_names = sorted(train_df["label"].unique().tolist())
num_classes = len(class_names)
print(f"Classes ({num_classes}): {class_names}")
str2idx = tf.keras.layers.StringLookup(
    vocabulary=class_names, num_oov_indices=0, mask_token=None
)
idx2str = tf.keras.layers.StringLookup(
    vocabulary=str2idx.get_vocabulary(), invert=True, mask_token=None
)

# ======= Split estratificado treino/val =======
train_idx, val_idx = train_test_split(
    train_df.index, test_size=0.2, random_state=42, stratify=train_df["label"]
)
train_split = train_df.loc[train_idx].reset_index(drop=True)
val_split   = train_df.loc[val_idx].reset_index(drop=True)

# ======= Data augmentation (leve e útil p/ ações) =======
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomContrast(0.1),
])

# ======= Loader de imagem =======
def load_image(path, label=None, augment_on=False):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)  # ou decode_png
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    if augment_on:
        img = augment(img, training=True)
    # PREPROCESS do backbone escolhido:
    if label is None:
        return img
    y = str2idx(label)                         # string -> índice [0..num_classes-1]
    return img, tf.cast(y, tf.int32)

def make_ds(df, base_dir, shuffle=False, augment_on=False):
    paths = [str(base_dir / fn) for fn in df["filename"]]
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

train_ds = make_ds(train_split, TRAIN_IMAGES_DIR, shuffle=True, augment_on=True)
val_ds   = make_ds(val_split,   TRAIN_IMAGES_DIR, shuffle=False, augment_on=False)
test_ds  = make_ds(test_df,     TEST_IMAGES_DIR,  shuffle=False, augment_on=False)

# ======= (Opcional) Pesos de classe se houver desbalanceamento real =======
# from collections import Counter
# counts = Counter(train_split["label"])
# total = sum(counts.values())
# class_weights = { str2idx(tf.constant(k)).numpy(): total/(num_classes*counts[k]) for k in counts }
class_weights = None  # dataset do Kaggle costuma ser balanceado

# ======= Backbone + cabeça multiclasse =======
base_model = tf.keras.applications.EfficientNetV2B0(
    include_top=False,
    weights="imagenet",
    include_preprocessing=True,
    input_shape=(*IMG_SIZE, 3)
)
base_model.trainable = False  # ETAPA 1: congelado

inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.35)(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.45)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

# ======= Compilações =======
# ETAPA 1: treina só a cabeça
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[
        "accuracy",
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3")
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
    verbose=1
)

# ETAPA 2: fine-tuning parcial do backbone
base_model.trainable = True
# Descongela últimas “n” camadas (ajuste fino):
fine_tune_at = int(len(base_model.layers) * 0.6)  # libera ~40% finais
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[
        "accuracy",
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3")
    ],
)

f1_ckpt = F1Checkpoint(val_ds, idx2str, path="./best_by_f1.keras")
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)
early_ft = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

print("\n=== ETAPA 2: fine-tuning ===\n")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FT,
    callbacks=[f1_ckpt, reduce_lr, early_ft],
    class_weight=class_weights,
    verbose=1
)

# ======= Inferência no conjunto de teste (sem rótulos) =======
probs = model.predict(test_ds, verbose=1)
pred_idx = probs.argmax(axis=1)
pred_labels = idx2str(tf.constant(pred_idx)).numpy().astype(str)

# Cria CSV de submissão (filename,label)
sub = pd.DataFrame({
    "filename": test_df["filename"],
    "label": pred_labels
})
sub.to_csv("submission.csv", index=False)
print("Arquivo salvo: submission.csv")
