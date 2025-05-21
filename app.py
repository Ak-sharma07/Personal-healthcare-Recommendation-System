from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

app = Flask(__name__)
CORS(app) 

df = None
diet_map = {}
med_map = {}
prec_map = {}
work_map = {}
desc_map = {}


biobert_tokenizer = None
biobert_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models_and_data():
    """Load all models and data needed for predictions"""
    global df, diet_map, med_map, prec_map, work_map, desc_map
    global biobert_tokenizer, biobert_model

    # 1) LOAD ALL CSVs
    symptoms_df = pd.read_csv("dataset/symtoms_df.csv")
    symptom_sev = pd.read_csv("dataset/Symptom-severity.csv")
    desc_df     = pd.read_csv("dataset/description.csv")
    diets_df    = pd.read_csv("dataset/diets.csv")
    meds_df     = pd.read_csv("dataset/medications.csv")
    prec_df     = pd.read_csv("dataset/precautions_df.csv")
    workout_df  = pd.read_csv("dataset/workout_df.csv")
    
    # 2) Process symptoms
    symptom_cols = ["Symptom_1", "Symptom_2", "Symptom_3", "Symptom_4"]
    symptoms_df["symptoms_list"] = (
        symptoms_df[symptom_cols]
          .apply(lambda row: [s for s in row if pd.notna(s)], axis=1)
    )
    symptoms_df["symptoms_text"] = symptoms_df["symptoms_list"].apply(lambda L: ", ".join(L))
    
    # 3) Compute total severity
    sev_map = dict(zip(symptom_sev["Symptom"], symptom_sev["weight"]))
    def total_severity(symptoms):
        return sum(sev_map.get(s, 0) for s in symptoms)
    symptoms_df["total_severity"] = symptoms_df["symptoms_list"].apply(total_severity)
    
    # 4) Merge descriptions
    df = symptoms_df.merge(desc_df, on="Disease", how="left")
    df["Description"] = df["Description"].fillna("")
    
    # 5) Build precaution list
    prec_cols = ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]
    prec_df["prec_list"] = (
        prec_df[prec_cols]
          .apply(lambda row: [p.strip() for p in row if pd.notna(p) and p.strip() != ""], axis=1)
    )
    prec_map = prec_df.set_index("Disease")["prec_list"].to_dict()
    
    # 6) Map diseases to recommendations
    def split_list(cell):
        return [x.strip() for x in str(cell).split(",") if x.strip()]
    
    diet_map = diets_df.set_index("Disease")["Diet"].apply(split_list).to_dict()
    med_map  = meds_df.set_index("Disease")["Medication"].apply(split_list).to_dict()
    work_map = workout_df.set_index("disease")["workout"].apply(split_list).to_dict()
    desc_map = desc_df.set_index("Disease")["Description"].to_dict()
    
    df["diet_list"] = df["Disease"].map(diet_map)
    df["med_list"]  = df["Disease"].map(med_map)
    df["prec_list"] = df["Disease"].map(prec_map)
    df["work_list"] = df["Disease"].map(work_map)
    
    # 7) Drop rows with missing targets
    df = df.dropna(subset=["diet_list", "med_list", "prec_list", "work_list"]).reset_index(drop=True)
    
    # 8) Clean and combine text
    def clean_text(s):
        s = s.lower()
        s = re.sub(r"[^a-z0-9, ]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
    
    df["clean_symptoms"] = df["symptoms_text"].apply(clean_text)
    df["clean_desc"]     = df["Description"].apply(clean_text)
    df["text_input"]     = df["clean_symptoms"] + " " + df["clean_desc"]
    
    # Load BioBERT model & tokenizer
    biobert_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    biobert_model     = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1").to(device)
    
    # Compute and store embeddings for every disease entry
    def get_embedding(text: str) -> np.ndarray:
        inputs = biobert_tokenizer(text,
                                   return_tensors="pt",
                                   truncation=True,
                                   padding=True,
                                   max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = biobert_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    
    df["embedding"] = df["text_input"].apply(get_embedding)
    
    print("Models and data loaded successfully!")

def predict_disease_and_recommendations(text_input):
    """Predict disease and get recommendations based on user input"""
    # Clean the input
    def clean_text(s):
        s = s.lower()
        s = re.sub(r"[^a-z0-9, ]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
    
    cleaned = clean_text(text_input)
    
    # Embed user input via BioBERT
    inputs = biobert_tokenizer(cleaned,
                               return_tensors="pt",
                               truncation=True,
                               padding=True,
                               max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = biobert_model(**inputs)
    user_emb = out.last_hidden_state[:, 0, :].squeeze().cpu().numpy().reshape(1, -1)
    
    # Find most similar disease via cosine similarity on embeddings
    similarities = cosine_similarity(user_emb, np.vstack(df["embedding"].values))
    top_idx = similarities[0].argmax()
    predicted_disease = df.iloc[top_idx]["Disease"]
    
    # Get description
    description = desc_map.get(predicted_disease, "No description available")
    
    # Confidence score
    confidence = float(similarities[0][top_idx])
    
    # Fetch recommendations
    recommendations = {
        "Diets":       diet_map.get(predicted_disease, []),
        "Medicines":   med_map.get(predicted_disease, []),
        "Precautions": prec_map.get(predicted_disease, []),
        "Workouts":    work_map.get(predicted_disease, []),
    }
    
    return {
        "disease":        predicted_disease,
        "description":    description,
        "confidence":     confidence,
        "recommendations": recommendations
    }

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "No text input provided"}), 400
    
    text_input = data['text']
    result = predict_disease_and_recommendations(text_input)
    return jsonify(result)

@app.route('/api/diseases', methods=['GET'])
def get_diseases():
    """Return a list of all diseases in the dataset"""
    diseases = sorted(df["Disease"].unique().tolist())
    return jsonify({"diseases": diseases})

if __name__ == '__main__':
    load_models_and_data()
    app.run(debug=True, port=5000)
