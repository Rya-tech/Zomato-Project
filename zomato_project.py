
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# --------------------------------
# 1. LOAD THE TWO DATASETS
# --------------------------------

reviews = pd.read_csv("Zomato Restaurant reviews.csv")
metadata = pd.read_csv("Zomato Restaurant names and Metadata.csv")

print("Reviews Dataset:", reviews.shape)
print("Metadata Dataset:", metadata.shape)

# --------------------------------
# 2. MERGE THE DATASETS
# --------------------------------

data = pd.merge(
    reviews,
    metadata,
    left_on="Restaurant",
    right_on="Name",
    how="left"
)

print("\nMerged Dataset:", data.shape)

# Save merged dataset
data.to_csv("zomato_merged_dataset.csv", index=False)

print("\nDatasets merged successfully!")

# --------------------------------
# 3. DATA PREPROCESSING
# --------------------------------

# Convert ratings into numeric values
data["Rating"] = pd.to_numeric(
    data["Rating"], errors="coerce"
)

# Remove missing reviews and ratings
data = data.dropna(
    subset=["Review", "Rating"]
).copy()

data["Review"] = data["Review"].astype(str).str.strip()

data = data[data["Review"] != ""]

# Convert restaurant cost into numeric values
data["Cost_numeric"] = pd.to_numeric(
    data["Cost"].astype(str).str.replace(",", "", regex=False),
    errors="coerce"
)

# Handle missing cuisines
data["Cuisines"] = data["Cuisines"].fillna("Unknown")

# --------------------------------
# 4. CREATE SENTIMENT LABELS
# --------------------------------

def get_sentiment(rating):

    if rating >= 4:
        return "Positive"

    elif rating <= 2:
        return "Negative"

    else:
        return "Neutral"


data["Sentiment"] = data["Rating"].apply(get_sentiment)

print("\nSentiment Distribution:")
print(data["Sentiment"].value_counts())

# --------------------------------
# 5. SELECT FEATURES AND TARGET
# --------------------------------

X = data[[
    "Review",
    "Cuisines",
    "Cost_numeric"
]]

y = data["Sentiment"]

# --------------------------------
# 6. SPLIT DATASET
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# --------------------------------
# 7. FEATURE EXTRACTION
# --------------------------------

preprocessor = ColumnTransformer(

    transformers=[

        ("text",
         TfidfVectorizer(
             lowercase=True,
             stop_words="english",
             ngram_range=(1, 2),
             max_features=30000,
             min_df=2
         ),
         "Review"),

        ("cuisine",
         OneHotEncoder(handle_unknown="ignore"),
         ["Cuisines"]),

        ("cost",
         Pipeline([
             ("imputer", SimpleImputer(strategy="median")),
             ("scaler", StandardScaler())
         ]),
         ["Cost_numeric"])
    ]
)

# --------------------------------
# 8. CREATE MACHINE LEARNING MODEL
# --------------------------------

model = Pipeline([

    ("preprocessing", preprocessor),

    ("classifier", LogisticRegression(
        max_iter=1200,
        class_weight="balanced"
    ))

])

# --------------------------------
# 9. TRAIN THE MODEL
# --------------------------------

print("\nTraining the model...")

model.fit(X_train, y_train)

print("Model training completed!")

# --------------------------------
# 10. TEST THE MODEL
# --------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:",
      round(accuracy * 100, 2), "%")

print("\nClassification Report:")

print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

# --------------------------------
# 11. SAVE THE MODEL
# --------------------------------

joblib.dump(model, "zomato_sentiment_model.joblib")

print("\nModel saved successfully!")

# --------------------------------
# 12. PREDICT NEW CUSTOMER REVIEWS
# --------------------------------

print("\n--- NEW REVIEW PREDICTION ---")

while True:

    review = input("\nEnter restaurant review (or exit): ")

    if review.lower() == "exit":
        break

    cuisine = input("Enter restaurant cuisine: ")

    cost = input("Enter restaurant cost: ")

    if cost.strip() == "":
        cost = np.nan
    else:
        cost = float(cost)

    new_data = pd.DataFrame([{
        "Review": review,
        "Cuisines": cuisine,
        "Cost_numeric": cost
    }])

    result = model.predict(new_data)

    print("\nPredicted Sentiment:", result[0])

    print("----------------------------")
