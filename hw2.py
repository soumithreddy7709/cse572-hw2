import re, json, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

# --- Load ---
df = pd.read_csv("titanic/train.csv")
X, y = df.drop(columns=["Survived"]), df["Survived"].astype(int)
df_test = pd.read_csv("titanic/test.csv")

# --- Feature engineering ---
def add_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["FamilySize"] = X["SibSp"].fillna(0) + X["Parch"].fillna(0) + 1
    X["IsAlone"] = (X["FamilySize"] == 1).astype(int)

    def extract_title(name):
        m = re.search(r",\s*([^\.]*)\.", str(name))
        return m.group(1).strip() if m else "None"
    X["Title"] = X["Name"].apply(extract_title)
    title_map = {
        "Mlle":"Miss","Ms":"Miss","Mme":"Mrs",
        "Lady":"Royalty","Countess":"Royalty","Sir":"Royalty","Don":"Royalty","Dona":"Royalty","Jonkheer":"Royalty",
        "Capt":"Officer","Col":"Officer","Major":"Officer","Dr":"Officer","Rev":"Officer",
    }
    X["Title"] = X["Title"].replace(title_map)
    vc = X["Title"].value_counts()
    X.loc[X["Title"].isin(vc[vc < 10].index), "Title"] = "Rare"

    X["Deck"] = X["Cabin"].astype(str).str[0]
    X.loc[X["Deck"].isin(["n","N"," ", "0"]), "Deck"] = "Unknown"
    X["CabinKnown"] = X["Cabin"].notna().astype(int)
    return X

numeric_features = ["Age","SibSp","Parch","Fare","FamilySize","IsAlone"]
categorical_features = ["Pclass","Sex","Embarked","Title","Deck","CabinKnown"]
numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median"))])
categorical_transformer = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                   ("ohe", OneHotEncoder(handle_unknown="ignore"))])
preprocess = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- Decision Tree (grid search) ---
dt_pipe = Pipeline([
    ("feats", FunctionTransformer(add_features, validate=False)),
    ("preprocess", preprocess),
    ("model", DecisionTreeClassifier(random_state=42)),
])
dt_grid = {
    "model__criterion": ["gini","entropy"],
    "model__max_depth": [None, 4, 6, 8, 10],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
}
dt_search = GridSearchCV(dt_pipe, dt_grid, scoring="accuracy", cv=cv, n_jobs=-1, refit=True)
dt_search.fit(X, y)
dt_best = dt_search.best_estimator_
dt_scores = cross_val_score(dt_best, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

# --- Plot the tuned tree ---
dt_best.fit(X, y)
feature_names = dt_best.named_steps["preprocess"].get_feature_names_out()
feature_names = [f.split("__", 1)[-1] for f in feature_names]
plt.figure(figsize=(22, 12))
plot_tree(dt_best.named_steps["model"], feature_names=feature_names, class_names=["Died","Survived"], filled=False)
plt.title("Fine‑tuned Decision Tree (Titanic)")
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=220)
plt.close()

# --- Random Forest (grid search) ---
rf_pipe = Pipeline([
    ("feats", FunctionTransformer(add_features, validate=False)),
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(random_state=42, n_jobs=-1)),
])
rf_grid = {
    "model__n_estimators": [200],
    "model__max_depth": [None, 8, 14],
    "model__min_samples_leaf": [1, 2],
    "model__max_features": ["sqrt", 0.7],
}
rf_search = GridSearchCV(rf_pipe, rf_grid, scoring="accuracy", cv=cv, n_jobs=-1, refit=True)
rf_search.fit(X, y)
rf_best = rf_search.best_estimator_
rf_scores = cross_val_score(rf_best, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

print("Decision Tree best params:", dt_search.best_params_)
print("Decision Tree 5-fold CV accuracy: mean=%.4f std=%.4f" % (dt_scores.mean(), dt_scores.std()))
print("Random Forest best params:", rf_search.best_params_)
print("Random Forest 5-fold CV accuracy: mean=%.4f std=%.4f" % (rf_scores.mean(), rf_scores.std()))
print("Better algorithm:", "Random Forest" if rf_scores.mean() >= dt_scores.mean() else "Decision Tree")
