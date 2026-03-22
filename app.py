import requests
import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# ================= Load Artifacts =================
with open("anime_df.pkl", "rb") as f:
    df = pickle.load(f)

with open("anime_tfidf_matrix.pkl", "rb") as f:
    X = pickle.load(f)

# ================= Helper Function =================
def get_anime_data(mal_id):
    url = f"https://api.jikan.moe/v4/anime/{mal_id}"
    response = requests.get(url)

    if response.status_code != 200:
        return None

    return response.json()["data"]

def recommend_anime(anime_name, df, X, top_n=5):
    idx = df[df["Name"] == anime_name].index[0]
    sim_scores = cosine_similarity(X[idx], X)[0]
    sim_scores = sorted(
        list(enumerate(sim_scores)),
        key=lambda x: x[1],
        reverse=True
    )[1:top_n + 1]

    anime_indices = [i[0] for i in sim_scores]
    return df.iloc[anime_indices]

# ================= Streamlit UI =================
st.title("🎌 Anime Recommendation System")

selected_anime = st.selectbox(
    "Select an Anime:",
    df["Name"].values
)

if st.button("Recommend"):
    recommendations = recommend_anime(selected_anime, df, X)

    # ===== Selected Anime =====
    st.subheader("Selected Anime")
    mal_id = df[df["Name"] == selected_anime]["MAL_ID"].iloc[0]
    data = get_anime_data(mal_id)

    if data:
        poster = data["images"]["jpg"]["large_image_url"]

        col1, col2 = st.columns([1, 3])
        with col1:
            if poster:
                st.image(poster, width=120)
        with col2:
            st.markdown(f"### {selected_anime}")
            st.write(f"Episodes: {data['episodes']}")
            st.write(f"Rating: {data['rating']}")
            st.write(data["synopsis"])

    # ===== Recommended Anime =====
    st.subheader("Recommended Anime")

    for _, row in recommendations.iterrows():
        rec_data = get_anime_data(row["MAL_ID"])
        if not rec_data:
            continue

        poster = rec_data["images"]["jpg"]["large_image_url"]

        col1, col2 = st.columns([1, 3])
        with col1:
            if poster:
                st.image(poster, width=120)
        with col2:
            st.markdown(f"### {row['Name']}")
            st.write(f"Episodes: {rec_data['episodes']}")
            st.write(f"Rating: {rec_data['rating']}")
            st.write(rec_data["synopsis"])
