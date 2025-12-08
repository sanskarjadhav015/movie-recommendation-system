import pickle
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("clustered_df.csv")
overview_tfidf = pickle.load(open("overview_tfidf.pkl", "rb"))

# ---------------- RECOMMENDATION FUNCTION ----------------
def recommend_movies(title, df, num_recommendations=6):
    if title not in df['Series_Title'].values:
        return None

    cluster_label = df[df['Series_Title'] == title]['Cluster'].values[0]
    cluster_movies = df[df['Cluster'] == cluster_label]

    movie_vector = overview_tfidf[df[df['Series_Title'] == title].index[0]]
    similarities = cosine_similarity(
        movie_vector,
        overview_tfidf[cluster_movies.index]
    ).flatten()

    similar_indices = similarities.argsort()[-(num_recommendations + 1):-1][::-1]

    recommendations = cluster_movies.iloc[similar_indices][
        ['Series_Title', 'Overview', 'IMDB_Rating', 'Poster_Link']
    ]

    return recommendations.reset_index(drop=True)

# ---------------- HEADER ----------------
st.markdown(
    "<h1 style='text-align:center;'>🎬 Movie Recommendation System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;color:gray;font-size:17px;'>"
    "AI-based Movie Recommendation using TF-IDF, Cosine Similarity & Clustering"
    "</p>",
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🎥 Select Movie")
movie_list = sorted(df['Series_Title'].dropna().unique())
selected_movie = st.sidebar.selectbox("Choose a Movie", movie_list)
num_recommendations = st.sidebar.slider(
    "Number of Recommendations", 3, 10, 6
)

# ---------------- MAIN ACTION ----------------
st.markdown("### 🔎 Get Similar Movie Recommendations")

if st.button("🍿 Recommend Movies"):
    output = recommend_movies(
        selected_movie,
        df,
        num_recommendations=num_recommendations
    )

    if output is None:
        st.error("❌ Movie not found in dataset.")
    else:
        st.success(f"✅ Showing recommendations similar to **{selected_movie}**")

        for i in range(0, len(output), 3):
            cols = st.columns(3)

            for j in range(3):
                if i + j < len(output):
                    row = output.iloc[i + j]

                    with cols[j]:
                        st.image(row['Poster_Link'], use_container_width=True)
                        st.markdown(f"### 🎞 {row['Series_Title']}")
                        st.write(f"⭐ **IMDB Rating:** {row['IMDB_Rating']}")
                        st.write(f"📝 {row['Overview'][:120]}...")
                        st.markdown("---")

# ---------------- FOOTER ----------------
st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
<style>
.footer {
    background-color: #0f172a;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
.footer p {
    color: #e5e7eb;
    font-size: 15px;
    margin: 6px 0;
}
.footer a {
    color: #38bdf8;
    text-decoration: none;
    font-weight: 600;
    margin: 0 10px;
}
.footer a:hover {
    color: #22c55e;
}
</style>

<div class="footer">
    <p>Developed by <b>Sanskar Jadhav</b></p>
    <p>
        <a href="https://mern-portfolio-pink.vercel.app/" target="_blank">🌐 Portfolio</a> |
        <a href="https://github.com/sanskarjadhav015" target="_blank">💻 GitHub</a> |
        <a href="https://www.linkedin.com/in/jadhav-sanskar-kishor" target="_blank">🔗 LinkedIn</a> |
        <a href="mailto:sanskarjadhav015@gmail.com">✉ Email</a>
    </p>
</div>
""", unsafe_allow_html=True)
