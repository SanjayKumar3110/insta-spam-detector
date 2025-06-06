import streamlit as st
import instaloader      
import re
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import requests
from PIL import Image
from io import BytesIO

# Load model and dataset
MODEL_PATH = os.path.join(os.path.dirname(__file__), "instagram_fake_detection.pkl")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "instagram_comments.csv")

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATASET_PATH)

# Define features
feature_columns = [
    'profile pic', 'nums/length username', 'fullname words', 'nums/length fullname',
    'name==username', 'description length', 'external URL', 'private',
    '#posts', '#followers', '#follows'
]

# Convert binary columns
binary_map = {'Yes': 1, 'No': 0, 'TRUE': 1, 'FALSE': 0, 'true': 1, 'false': 0}
for col in ['profile pic', 'name==username', 'external URL', 'private']:
    if df[col].dtype == object:
        df[col] = df[col].map(binary_map)

# Fit the scaler
X = df[feature_columns]
scaler = StandardScaler()
scaler.fit(X)

# Streamlit UI setup
st.set_page_config(page_title="Instagram Spam Detector", layout="centered")
st.title("📸 Instagram Spam Detector")
st.markdown("Enter an Instagram profile URL to detect spam account probability.")

# Cached loader with optional session
@st.cache_resource
def get_loader():
    L = instaloader.Instaloader()
    try:
        L.load_session_from_file() # replace this with your instagram account(optional)
    except Exception:
        pass
    return L

L = get_loader()

# Input field
url = st.text_input("🔗 Instagram Profile URL")
if url and "instagram.com/" in url:
    username_match = re.findall(r"instagram\.com/([^/?#]+)", url)
    if username_match:
        username = username_match[0]

        if st.button("Analyze Profile"):
            try:
                with st.spinner("Fetching data..."):
                    profile = instaloader.Profile.from_username(L.context, username)

                    # Feature extraction
                    profile_pic = 1
                    nums_len_username = sum(char.isdigit() for char in profile.username) / len(profile.username)
                    fullname_words = len(profile.full_name.split())
                    nums_len_fullname = (
                        sum(char.isdigit() for char in profile.full_name) / len(profile.full_name)
                        if len(profile.full_name) > 0 else 0
                    )
                    name_eq_username = 1 if profile.username.lower() == profile.full_name.lower().replace(" ", "") else 0
                    desc_length = len(profile.biography)
                    external_url = 1 if profile.external_url else 0
                    is_private = 1 if profile.is_private else 0
                    posts = profile.mediacount
                    followers = profile.followers
                    follows = profile.followees

                    features = [[
                        profile_pic, nums_len_username, fullname_words, nums_len_fullname,
                        name_eq_username, desc_length, external_url, is_private,
                        posts, followers, follows
                    ]]

                    features_scaled = scaler.transform(features)
                    probability = model.predict_proba(features_scaled)[0][1]
                    prediction = model.predict(features_scaled)[0]

                # Show profile data
                col1, col2 = st.columns([1, 2])
                with col1:
                    try:
                        response = requests.get(profile.profile_pic_url, timeout=10)
                        img = Image.open(BytesIO(response.content))
                        st.image(img, width=120)
                    except Exception:
                        st.warning("⚠️ Could not load profile picture.")

                with col2:
                    st.markdown(f"**Username:** @{profile.username}")
                    st.markdown(f"**Full Name:** {profile.full_name}")
                    st.markdown(f"**Followers:** {followers}")
                    st.markdown(f"**Following:** {follows}")
                    st.markdown(f"**Posts:** {posts}")
                    st.markdown(f"**Private Account:** {'Yes' if is_private else 'No'}")

                # Show probability
                st.markdown("### 🔎 Spam Probability")
                spam_percent = int(probability * 100)
                st.progress(spam_percent)
                st.markdown(f"**Spam Likelihood: {spam_percent}%**")
                st.markdown("This result is just a prediction AI can sometime make mistakes")
                st.subheader("🧠 Prediction Result")
                if prediction == 1:
                    st.error("🚨 This account is likely a FAKE/SPAM profile.")
                else:
                    st.success("✅ This account seems to be REAL.")
            except Exception as e:
                st.error(f"❌ Error fetching profile: {e}")
    else:
        st.warning("Invalid Instagram URL format.")

else:
    st.info("Enter a full Instagram profile URL to start.")


# run this file in terminal using the command -- streamlit run app.py
