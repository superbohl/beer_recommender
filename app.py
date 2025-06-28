import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import pandas as pd
import numpy as np
import gdown


# Load enriched dataset
@st.cache_data
def load_data():
    file_id = "1qWIC8AamOlGmdLAXZqJenYB1WGBgUV2V"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "beer_reviews_enriched.csv"
    gdown.download(url, output, quiet=False)
    df = pd.read_csv(output)
    df = df.dropna(subset=["beer_style", "beer_abv", "review_overall"])
    return df

df = load_data()

# Define trait list
traits = ["hoppy", "malty", "dark", "light", "bitter", "sweet", "sour", "fruity"]

# --- UI ---
st.title("🍺 Beer Recommender (Flavor Match Edition)")
st.markdown("Answer a couple of quick questions and we'll recommend beers that match your taste!")

# Question 1
experience_q = st.radio(
    "What kind of beer experience are you looking for?",
    [
        "Crisp and refreshing",
        "Sweet and smooth",
        "Bold and bitter",
        "Dark and roasty",
        "Fruity or sour"
    ]
)

experience_map = {
    "Crisp and refreshing": {"light": 1, "hoppy": 0.5},
    "Sweet and smooth": {"malty": 1, "sweet": 1},
    "Bold and bitter": {"hoppy": 1, "bitter": 1},
    "Dark and roasty": {"dark": 1, "malty": 1, "bitter": 0.5},
    "Fruity or sour": {"fruity": 1, "sour": 1}
}

# Question 2
second_q = st.radio(
    "Pick a secondary characteristic you like:",
    ["Not too strong", "High ABV", "Fruity", "Dark", "I’m open to anything"]
)

secondary_map = {
    "Not too strong": {"light": 1},
    "High ABV": {"bitter": 0.5},
    "Fruity": {"fruity": 1},
    "Dark": {"dark": 1},
    "I’m open to anything": {}
}

# Build user trait vector
user_pref_vector = {trait: 0.0 for trait in traits}

for trait, weight in experience_map[experience_q].items():
    user_pref_vector[trait] += weight

for trait, weight in secondary_map[second_q].items():
    user_pref_vector[trait] += weight

# Normalize
max_val = max(user_pref_vector.values()) or 1
user_pref_vector = {k: round(v / max_val, 2) for k, v in user_pref_vector.items()}

# ABV filter
abv_range = st.slider("Select your preferred ABV range:", 0.0, 15.0, (4.0, 8.0), 0.1)
filtered = df[df["beer_abv"].between(*abv_range)]

# Score beers
def score_beer(row, user_prefs):
    beer_vector = np.array([row[t] for t in traits])
    user_vector = np.array([user_prefs[t] for t in traits])
    return np.dot(beer_vector, user_vector)

for trait in traits:
    if trait not in filtered.columns:
        filtered[trait] = 0.0
    else:
        filtered.loc[:, trait] = filtered[trait].fillna(0.0)

filtered["match_score"] = filtered.apply(score_beer, axis=1, user_prefs=user_pref_vector)

# Top matches
top_matches = (
    filtered[["beer_name", "brewery_name", "beer_abv", "review_overall", "match_score"]]
    .drop_duplicates()
    .sort_values(by="match_score", ascending=False)
    .head(20)
)

# Display results
st.subheader("🎯 Recommended Beers for You")
if top_matches.empty:
    st.warning("No beers match your preferences. Try adjusting your answers or ABV range.")
else:
    st.dataframe(top_matches)
