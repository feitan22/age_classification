# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import joblib
from ultralytics import YOLO
from skimage.feature import hog

# ---------------------------
# Fonction HOG pour une image
# ---------------------------
def extract_features_from_image(img):
    img = cv2.resize(img, (128, 128))
    img = (img / 255.0).astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16,16),
        cells_per_block=(2,2),
        block_norm='L2-Hys'
    )
    return feat

# ---------------------------
# YOLO pour détection personnes
# ---------------------------
yolo_model = YOLO("yolov8n.pt")  # YOLOv8 Nano pré-entraîné COCO

# ---------------------------
# Charger tous les modèles
# ---------------------------
models = {
    "KNN": joblib.load("models/knn.pkl"),
    "Decision Tree": joblib.load("models/decision_tree.pkl"),
    "Naive Bayes": joblib.load("models/naive_bayes.pkl")
}
age_categories = ["child", "adult", "elderly"]

# ---------------------------
# Streamlit interface
# ---------------------------
st.title("Détection de personnes et classification par âge")
model_choice = st.selectbox("Choisir le modèle d'âge :", ["KNN", "Decision Tree", "Naive Bayes"])
uploaded_file = st.file_uploader("Uploader une image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    # Détection personnes YOLO
    results = yolo_model(img_np)
    person_features = []
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    for box, cls in zip(boxes, classes):
        if int(cls) == 0:  # COCO classe 0 = personne
            x1, y1, x2, y2 = map(int, box)
            person_img = img_np[y1:y2, x1:x2]
            feat = extract_features_from_image(person_img)
            person_features.append(feat)

    if len(person_features) == 0:
        st.warning("Aucune personne détectée !")
    else:
        # Prédiction âge avec le modèle choisi
        person_features = np.array(person_features)
        model_age = models[model_choice]
        preds = model_age.predict(person_features)

        # Compter catégories présentes
        counts = {}
        for cat in age_categories:
            counts[cat] = sum(preds == cat)

        # Affichage des résultats
        display_text = ""
        for cat, cnt in counts.items():
            if cnt > 0:
                display_text += f"{cnt} {cat}{'s' if cnt > 1 else ''}  "
        st.success(display_text.strip())

        # Afficher l'image avec les boxes
        for box, cls in zip(boxes, classes):
            if int(cls) == 0:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img_np, (x1, y1), (x2, y2), (0,255,0), 2)

        st.image(img_np, caption="Image détectée", width=700)
