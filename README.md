# Instagram Spam Detector

A **Streamlit web app** that predicts whether an Instagram account is **fake or real** using machine learning. Just paste an Instagram profile URL, and the model analyzes key profile features to detect spammy or bot-like behavior.

## 🚀 Live Demo

👉 [Try it on Streamlit](https://your-streamlit-link.streamlit.app)  
---

## 🔍 How It Works

1. **User inputs an Instagram profile URL**
2. **Instaloader** fetches public profile data
3. Extracts features like:
   - Username & full name structure
   - Profile picture presence
   - Bio length, external links
   - Follower/following/post count
4. Uses a **Random Forest Classifier** to predict spam probability

---

## 🧠 Machine Learning Details

- Trained on custom-labeled Instagram dataset (`instagram_comments.csv`)
- Preprocessed using `StandardScaler`
- Model: `RandomForestClassifier` from `scikit-learn`
- Trained and saved using `core.py` as `instagram_fake_detection.pkl`

---

## 📁 File Structure

📦instagram-spam-detector

┣ 📄 app.py # Streamlit app for user interaction
┣ 📄 core.py # Training script for the model
┣ 📄 instagram_comments.csv # Dataset used for training
┣ 📄 instagram_fake_detection.pkl # Saved model
┣ 📄 requirements.txt # Python dependencies
┗ 📄 README.md


---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/SanjayKumar3110/insta-spam-detector.git
cd insta-spam

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
