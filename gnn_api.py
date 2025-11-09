from fastapi import FastAPI, Query
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from typing import List  # ✅ use this for compatibility

app = FastAPI()

# === Load side effect embeddings ===
data = torch.load(
    r"C:\Users\Manav\OneDrive\Desktop\MIT\3rd Year\Sem V\ANN\Project\HODDI\Data\side_effect_embeddings.pt",
    weights_only=False  # safe load for now
)
side_effect_names = [name.lower() for name in data["names"]]
side_effect_embeddings = data["embeddings"]

# === Sentence transformer for new symptom encoding ===
model = SentenceTransformer("all-MiniLM-L6-v2")

@app.get("/predict")
def predict(symptoms: List[str] = Query(...)):  # ✅ changed from list[str] to List[str]
    # Convert list of symptoms into one text representation
    symptom_text = " ".join(symptoms)
    symptom_embed = model.encode(symptom_text, convert_to_tensor=True)

    # Compute cosine similarity
    sim = F.cosine_similarity(symptom_embed.unsqueeze(0), side_effect_embeddings)
    topk = torch.topk(sim, k=5)

    results = []
    for idx, score in zip(topk.indices, topk.values):
        name = side_effect_names[idx]
        results.append({"side_effect": name, "similarity": float(score)})

    return {"input_symptoms": symptoms, "predictions": results}
