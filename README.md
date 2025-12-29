# Twitter sentimental analysis App

A Machine Learning application that fetches tweets and analyzes their sentiment (Positive/Negative). It uses a **Flask** backend for the logic and a **Streamlit** frontend for the interactive dashboard.

## Features
* **Real-time Analysis:** Enter any hashtag to see sentiment.
* **Dual Mode:** Works with Simulation data (`dummy_tweets.txt`) or Live Twitter API.
* **Visualizations:**
    * Sentiment Distribution (Pie Chart)
    * Word Cloud of most frequent words
    * Tweet Length analysis

## Tech Stack
* **Backend:** Python, Flask
* **Frontend:** Streamlit
* **ML:** Scikit-learn, NLTK
* **Data:** Pandas

## ⚙️ How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/BasilZafar11/twitter-sentiment-analysis.git](https://github.com/BasilZafar11/twitter-sentiment-analysis.git)
    cd twitter-sentiment-analysis
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Start the Backend (Flask):**
    ```bash
    python app.py
    ```

4.  **Start the Frontend (Streamlit):**
    Open a new terminal and run:
    ```bash
    streamlit run dashboard.py
    ```

## 📂 Project Structure
* `app.py`: The main backend server.
* `dashboard.py`: The user interface.
* `trained_model.sav`: The pre-trained Sentiment Analysis model.
* `dummy_tweets.txt`: Sample data for testing without API keys.