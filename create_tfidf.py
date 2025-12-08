import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your clustered dataset
df = pd.read_csv("clustered_df.csv")

# Fill missing overviews
df["Overview"] = df["Overview"].fillna("")

# Create TF-IDF matrix from Overview
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
overview_tfidf = tfidf.fit_transform(df["Overview"])

# Save the TF-IDF matrix
with open("overview_tfidf.pkl", "wb") as f:
    pickle.dump(overview_tfidf, f)

print("✅ overview_tfidf.pkl created successfully!")
