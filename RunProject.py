import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import time as t
import sklearn.utils as u
import sklearn.preprocessing as pp
import sklearn.tree as tr
import sklearn.ensemble as es
import sklearn.metrics as m
import sklearn.linear_model as lm
import sklearn.neural_network as nn
import numpy as np
import warnings as w
import os

w.filterwarnings('ignore')

# Create a directory for outputs
if not os.path.exists('outputs'):
    os.makedirs('outputs')

print("Loading data...")
data = pd.read_csv("AI-Data.csv")

print("Generating Correlation Heatmap (saved to outputs/correlation_heatmap.png)...")
plt.figure(figsize=(12, 8))
sb.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("outputs/correlation_heatmap.png")
plt.close()

# Skipping interactive graphs for automated run
print("\nSkipping interactive graphs. Running Machine Learning models...")

# Data Preprocessing
# Note: Dropping some columns as per original script
data_processed = data.copy()
data_processed = data_processed.drop(["gender", "StageID", "GradeID", "NationalITy", "PlaceofBirth", 
                                      "SectionID", "Topic", "Semester", "Relation", 
                                      "ParentschoolSatisfaction", "ParentAnsweringSurvey", 
                                      "AnnouncementsView"], axis=1)

u.shuffle(data_processed)

# Label Encoding
for column in data_processed.columns:
    if data_processed[column].dtype == type(object):
        le = pp.LabelEncoder()
        data_processed[column] = le.fit_transform(data_processed[column])

ind = int(len(data_processed) * 0.70)
feats = data_processed.values[:, 0:4]
lbls = data_processed.values[:, 4]

feats_Train = feats[0:ind]
feats_Test = feats[(ind+1):len(feats)]
lbls_Train = lbls[0:ind]
lbls_Test = lbls[(ind+1):len(lbls)]

def evaluate_model(model, name):
    print(f"\n--- {name} ---")
    model.fit(feats_Train, lbls_Train)
    lbls_pred = model.predict(feats_Test)
    acc = m.accuracy_score(lbls_Test, lbls_pred)
    print(f"Accuracy: {acc:.3f}")
    print("Classification Report:")
    print(m.classification_report(lbls_Test, lbls_pred))
    return acc

# Run Models
models = [
    (tr.DecisionTreeClassifier(), "Decision Tree"),
    (es.RandomForestClassifier(), "Random Forest"),
    (lm.Perceptron(), "Linear Model Perceptron"),
    (lm.LogisticRegression(), "Logistic Regression"),
    (nn.MLPClassifier(activation="logistic"), "Neural Network MLP")
]

results = []
for model, name in models:
    acc = evaluate_model(model, name)
    results.append((name, acc))

print("\nSummary of Accuracies:")
for name, acc in results:
    print(f"{name}: {acc:.3f}")

print("\nTask completed successfully.")
