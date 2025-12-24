# Age Classification App

Cette application permet de détecter les personnes sur une image et de les classifier par âge en trois catégories : **child, adult, elderly**. L’utilisateur peut choisir entre **KNN, Decision Tree et Naive Bayes**.

## Structure du projet

age_classification/
│
├─ app.py                  # Application Streamlit principale
├─ models/                 # Dossier contenant les modèles entraînés
│   ├─ knn.pkl             # Modèle KNN entraîné pour la classification d'âge
│   ├─ decision_tree.pkl   # Modèle Decision Tree entraîné
│   └─ naive_bayes.pkl     # Modèle Naive Bayes entraîné
├─ dataset/                # Dataset pour entraînement et tests
│   ├─ train/              # Images et annotations pour l'entraînement
│   ├─ valid/              # Images et annotations pour la validation
│   └─ test/               # Images et annotations pour les tests
├─ utils/                  # Fonctions utilitaires
│   └─ load_dataset.py     # Chargement des images et annotations
├─ trainning/              # Notebook et scripts pour entraîner les modèles
│   └─ train_models.ipynb  # Notebook pour entraîner KNN, Decision Tree, Naive Bayes
├─ requirements.txt        # Liste des packages Python nécessaires
└─ README.md               # Documentation complète du projet

### Installer les dépendances

pip install -r requirements.txt

#### Lancement de l’application

python -m streamlit run app.py
