# make_side_effect_embeddings.py
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# === 1. Load side-effect file ===
path = "C:/Users/Manav/OneDrive/Desktop/MIT/3rd Year/Sem V/ANN/Project/HODDI/Data/dictionary/side_effects_unique.txt"

print("Loading side-effect data...")
df = pd.read_csv(path, sep=",", on_bad_lines="skip", low_memory=False)

# === 2. Extract side effect names ===
side_effects = df["side_effect_name"].dropna().unique().tolist()
print(f"Found {len(side_effects)} unique side effects")

# === 3. Initialize light model (CPU friendly) ===
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# === 4. Encode ===
embeddings = model.encode(
    side_effects,
    batch_size=32,
    convert_to_tensor=True,
    show_progress_bar=True
)

# === 5. Save as .pt ===
save_path = "C:/Users/Manav/OneDrive/Desktop/MIT/3rd Year/Sem V/ANN/Project/HODDI/Data/side_effect_embeddings.pt"
torch.save({"names": side_effects, "embeddings": embeddings}, save_path)

print(f"✅ Saved {len(side_effects)} side-effect embeddings to {save_path}")
