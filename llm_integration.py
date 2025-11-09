from fastapi import FastAPI, Query
import torch
import torch.nn.functional as F
from typing import List
import ollama
import json

app = FastAPI()

# === Paths ===
BASE_PATH = r"C:\Users\Manav\OneDrive\Desktop\MIT\3rd Year\Sem V\ANN\Project\HODDI\Data"
DICT_PATH = BASE_PATH + r"\dictionary"

# === Load Embeddings ===
drug_embeddings = torch.load(BASE_PATH + r"\drug_embeddings.pt")
side_effect_embeddings = torch.load(BASE_PATH + r"\side_effect_embeddings.pt")

# === Load Drug Names ===
drug_names = []
with open(DICT_PATH + r"\Drugbank_ID_SMILE_all_structure links.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if parts:
            drug_names.append(parts[0])

# === Load Side Effect Names ===
side_effect_names = []
with open(DICT_PATH + r"\Side_effects_unique.txt", "r", encoding="utf-8") as f:
    header = f.readline()  # skip header
    for line in f:
        parts = line.strip().split(",")
        if len(parts) > 1:
            side_effect_names.append(parts[1].lower())

print(f"✅ Loaded {len(drug_names)} drugs and {len(side_effect_names)} side effects.")


# === Core Similarity Search ===
def find_similar_drugs(input_side_effects: List[str], top_k: int = 5):
    matched_indices = [side_effect_names.index(s.lower()) for s in input_side_effects if s.lower() in side_effect_names]
    if not matched_indices:
        return {"message": "No matching side effects found."}

    # Handle dict or tensor
    if isinstance(side_effect_embeddings, dict):
        matched_embeds = [side_effect_embeddings["embeddings"][i] for i in matched_indices]
        matched_embeds = torch.stack(matched_embeds)
    else:
        matched_tensor = torch.tensor(matched_indices, dtype=torch.long)
        matched_embeds = side_effect_embeddings[matched_tensor]

    avg_embedding = torch.mean(matched_embeds, dim=0)

    # === Dimension alignment (project 384 → 128) ===
    side_dim = avg_embedding.shape[0]
    drug_dim = drug_embeddings.shape[1] if drug_embeddings.ndim > 1 else drug_embeddings.shape[0]

    if side_dim != drug_dim:
        # simple linear projection (no training)
        projector = torch.nn.Linear(side_dim, drug_dim, bias=False)
        with torch.no_grad():
            avg_embedding = projector(avg_embedding)

    sims = F.cosine_similarity(avg_embedding.unsqueeze(0), drug_embeddings)
    topk = torch.topk(sims, top_k)

    results = [
        {"drug": drug_names[i], "similarity": float(sims[i])}
        for i in topk.indices
    ]
    return results


# === Step 1: Extract side effects from user text ===
def extract_side_effects(prompt: str):
    ollama_prompt = f"""
You are a medical text parser.
Extract possible side effects or symptoms mentioned in this text.
Return ONLY a JSON array of side effect strings — no labels, no text, just a valid JSON list.

Example:
Input: "I've been feeling dizzy and nauseous."
Output: ["dizziness", "nausea"]

Text: "{prompt}"
"""

    response = ollama.chat(model="llama3.2:3b", messages=[{"role": "user", "content": ollama_prompt}])
    content = response["message"]["content"].strip()

    # === Try multiple parsing strategies ===
    try:
        # Case 1: It's already a clean JSON list
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return [s.lower() for s in parsed]
        # Case 2: It's a JSON object with a key like {"side_effects": [...]}
        elif isinstance(parsed, dict) and "side_effects" in parsed:
            return [s.lower() for s in parsed["side_effects"]]
    except json.JSONDecodeError:
        pass

    # Case 3: Try to extract JSON-like list manually if the model added text
    import re
    match = re.search(r"\[(.*?)\]", content)
    if match:
        raw_list = "[" + match.group(1) + "]"
        try:
            parsed = json.loads(raw_list)
            if isinstance(parsed, list):
                return [s.lower() for s in parsed]
        except json.JSONDecodeError:
            pass

    # Case 4: Fallback — extract symptom-like words
    return [t.strip().lower() for t in content.replace(",", " ").replace("and", " ").split() if t]


# === Step 2: Generate human-readable response ===
def generate_response(symptoms: List[str], predicted_drugs: List[dict]):
    if not predicted_drugs:
        return "I couldn’t find any drugs related to those symptoms."

    drug_text = ", ".join([d["drug"] for d in predicted_drugs])
    symptom_text = ", ".join(symptoms)

    summary_prompt = f"""
You are a helpful medical assistant.
A user reported symptoms: {symptom_text}.
Based on a medical graph model, the most related drugs are: {drug_text}.
Write a short and clear response explaining these results in 2-3 sentences.
"""

    response = ollama.chat(model="llama3.2:3b", messages=[{"role": "user", "content": summary_prompt}])
    return response["message"]["content"]


# === Main Route ===
@app.get("/ask")
def ask_llm(prompt: str = Query(..., description="Describe your symptoms or side effects")):
    # Step 1: Extract symptoms
    symptoms = extract_side_effects(prompt)
    print(f"🧠 Extracted symptoms: {symptoms}")

    # Step 2: Find related drugs
    results = find_similar_drugs(symptoms)
    print(f"💊 Top drugs: {results}")

    # Step 3: Generate LLM response
    answer = generate_response(symptoms, results)
    return {
        "input": prompt,
        "symptoms": symptoms,
        "results": results,
        "response": answer
    }
