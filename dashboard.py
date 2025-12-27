import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Page Configuration 
st.set_page_config(page_title="Tweet Sentiment Dashboard", layout="wide")
st.title("Twitter Sentiment Analysis Dashboard")
st.write("Enter a hashtag to fetch 10 tweets and analyze their sentiment using a custom-trained model.")

# User Input
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    hashtag = st.text_input("Enter a Hashtag (e.g., 'python', 'AI'):", "python")
    analyze_button = st.button("Analyze Sentiment", use_container_width=True)

st.divider()

#  Submit Button and API Call 
if analyze_button:
    if not hashtag:
        st.error("Please enter a hashtag.")
    else:
        # Define the backend API endpoint
        api_url = "http://127.0.0.1:5000/predict-hashtag"
        payload = {"hashtag": hashtag}
        
        try:
            with st.spinner(f"Fetching and analyzing tweets for '#{hashtag}'..."):
                response = requests.post(api_url, json=payload)
            
            if response.status_code == 200:
                results = response.json()
                
                st.success("Analysis Complete!")
                
                df = pd.DataFrame(results)

                # Add emoji to sentiment for display
                df['sentiment_display'] = df['sentiment'].apply(
                    lambda s: f"Positive ✅" if s == "Positive" else f"Negative ❌"
                )
                
                df['word_count'] = df['cleaned_text'].apply(lambda x: len(x.split()))
                # Character count 
                df['char_count'] = df['tweet'].apply(lambda x: len(x))

                tab1, tab2, tab3 = st.tabs([" Metrics", " Exploratory Data Analysis (EDA)", " Raw Data"])

                #Main Metrics 
                with tab1:
                    st.subheader(f"Sentiment Breakdown for '#{hashtag}'")
                    sentiment_counts = df['sentiment_display'].value_counts()
                    
                    met_col1, met_col2 = st.columns(2)
                    with met_col1:
                        st.metric("Total Tweets", len(df))
                        st.metric("Positive Tweets", sentiment_counts.get("Positive ✅", 0))
                        st.metric("Negative Tweets", sentiment_counts.get("Negative ❌", 0))
                    
                    with met_col2:
                        # Graph 1: Sentiment Bar Chart
                        st.write("Sentiment Chart")
                        chart_data = sentiment_counts.rename_axis('Sentiment').reset_index(name='Tweets')
                        st.bar_chart(chart_data, x='Sentiment', y='Tweets')
                
                # Exploratory Data Analysis (EDA) 
                with tab2:
                    st.subheader("Analysis of Tweet Text")
                    
                    #Tables
                    col_t1, col_t2 = st.columns(2)
                    with col_t1:
                        # Table 1: Summary Statistics
                        st.write("Table 1: Text Statistics (Original Tweet)")
                        st.dataframe(df[['word_count', 'char_count']].describe(), use_container_width=True)
                    
                    with col_t2:
                        # Table 2: Statistics by Sentiment
                        st.write("Table 2: Average Statistics by Sentiment")
                        st.dataframe(df.groupby('sentiment')[['word_count', 'char_count']].mean(), use_container_width=True)
                    
                    st.divider()
                    
                    # Graphs 
                    col_g1, col_g2 = st.columns(2)
                    
                    with col_g1:
                        # Graph 2: Word Count Distribution
                        st.write("Graph 2: Word Count Distribution (Cleaned Text)")
                        st.bar_chart(df['word_count'].value_counts().sort_index())

                        # Graph 3: Word Count by Sentiment 
                        st.write("Graph 3: Word Count by Sentiment")
                        fig, ax = plt.subplots(figsize=(5, 2.5))
                        df.boxplot(column='word_count', by='sentiment', ax=ax, grid=False)
                        plt.suptitle('')
                        plt.title('Word Count Distribution', fontsize=10)
                        st.pyplot(fig)
                        
                        # Graph 4: Word Count vs. Character Count 
                        st.write("Graph 4: Word Count vs. Character Count")
                        st.scatter_chart(df, x='word_count', y='char_count', color='sentiment')
                    
                    with col_g2:
                        # Graph 5: Character Count Distribution 
                        st.write("Graph 5: Character Count Distribution (Original Tweet)")
                        st.bar_chart(df['char_count'].value_counts().sort_index())
                        
                        # Graph 6: Word Cloud
                        st.write("Graph 6: Most Common Words")
                        all_text = " ".join(df['cleaned_text'])
                        if all_text:
                            wordcloud = WordCloud(width=400, height=200, background_color='white').generate(all_text)
                            fig_wc, ax_wc = plt.subplots(figsize=(6, 3))
                            ax_wc.imshow(wordcloud, interpolation='bilinear')
                            ax_wc.axis('off')
                            st.pyplot(fig_wc)
                        else:
                            st.write("Not enough text to generate a word cloud.")

                
                with tab3:
                    st.subheader("Analyzed Tweets Data")
                    st.dataframe(df[['tweet', 'sentiment_display', 'word_count', 'char_count']], use_container_width=True, hide_index=True)

            else:
                st.error(f"Error from backend: {response.json().get('error', 'Unknown error')}")
        
        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to the backend.")
            st.info("Please makeD sure the Flask server is running in a separate terminal.")
        except Exception as e:
            st.error(f"An unknown error occurred: {e}")