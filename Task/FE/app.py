def funPack():
    pip install streamlit==1.24.0 fastapi==0.95.1 uvicorn[standard]==0.22.0 transformers==4.28.1 torch==2.1.0 sentencepiece==0.1.97 wordcloud==1.9.2 matplotlib==3.7.3 pydantic==1.10.7 requests==2.28.2 numpy==1.24.3 pandas==2.2.3 python-multipart==0.0.6
funPack()
import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

# FastAPI backend URL
backend_url = "http://127.0.0.1:8000"  # Update this URL if necessary

st.set_page_config(
    page_title="Job Market Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def fetch_dataset_preview():
    """Fetches a preview of the dataset from the backend."""
    try:
        response = requests.get(f"{backend_url}/data/preview", timeout=300)
        if response.status_code == 200:
            return pd.DataFrame(response.json())
        else:
            st.error(f"Failed to fetch dataset preview. Status code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching dataset preview: {e}")
        return None

def filter_data(experience_level):
    """Fetches filtered data from the backend based on experience level."""
    try:
        payload = {"experience_level": experience_level}
        response = requests.post(f"{backend_url}/data/filter", json=payload, timeout=10)
        if response.status_code == 200:
            return pd.DataFrame(response.json())
        else:
            st.error(f"Failed to fetch filtered data. Status code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching filtered data: {e}")
        return None

def get_ai_insights(query):
    """Fetches AI insights from the backend using job descriptions."""
    try:
        payload = {"query": query}
        response = requests.post(f"{backend_url}/ai/query", json=payload, timeout=1000)
        if response.status_code == 200:
            return response.json().get("insights")
        else:
            st.error(f"Failed to fetch AI insights. Status code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching AI insights: {e}")
        return None

# Load the dataset preview
jobs = fetch_dataset_preview()

if jobs is not None:
    st.title("📊 Job Market Analysis")
    st.write("Explore insights from the job market based on the dataset of job descriptions, required skills, and salaries.")

    st.write("### Dataset Preview")
    st.dataframe(jobs.head(10))

    st.write("### Dataset Information")
    col1, col2 = st.columns(2)
    col1.metric("Total Rows", jobs.shape[0])
    col2.metric("Total Columns", jobs.shape[1])

    st.write("### Experience Level Filter")
    experience_levels = ["all", "entry-level", "mid-level", "senior-level"]
    selected_experience = st.selectbox("Select Experience Level", experience_levels)

    filtered_jobs = filter_data(selected_experience)

    if filtered_jobs is not None and not filtered_jobs.empty:
        st.write("### Filtered Data")
        st.dataframe(filtered_jobs.head(10))

        # Extract necessary columns for visualizations
        if "Tokenized_Job_Description" in filtered_jobs.columns and "Skills_Required" in filtered_jobs.columns:
            all_tokens = [word for tokens in filtered_jobs["Tokenized_Job_Description"] for word in eval(tokens)]
            token_freq = Counter(all_tokens)
            most_common_words = token_freq.most_common(20)
            words, counts = zip(*most_common_words)

            skills_list = [skill.strip().lower() for skills in filtered_jobs["Skills_Required"] for skill in skills.split(",")]
            skill_freq = Counter(skills_list)
            top_skills = skill_freq.most_common(10)
            skills, skill_counts = zip(*top_skills)

            if "Sentiment_Label" in filtered_jobs.columns:
                sentiment_counts = filtered_jobs["Sentiment_Label"].value_counts()

            # Plot visualizations
            fig, axs = plt.subplots(2, 2, figsize=(18, 12))

            # Top 20 Most Common Words Plot
            axs[0, 0].bar(words, counts, color="mediumslateblue")
            axs[0, 0].set_title("Top 20 Most Common Words", fontsize=16)
            axs[0, 0].set_xlabel("Words", fontsize=14)
            axs[0, 0].set_ylabel("Frequency", fontsize=14)
            axs[0, 0].tick_params(axis="x", rotation=45)

            # Salary Distribution by Experience Level Plot
            if "Experience_Level" in filtered_jobs.columns and "Salary_Range" in filtered_jobs.columns:
                sns.countplot(data=filtered_jobs, x="Experience_Level", hue="Salary_Range", ax=axs[0, 1], palette="coolwarm")
                axs[0, 1].set_title("Salary Distribution by Experience Level", fontsize=16)
                axs[0, 1].set_xlabel("Experience Level", fontsize=14)
                axs[0, 1].set_ylabel("Count", fontsize=14)

            # Most In-Demand Skills Plot
            axs[1, 0].bar(skills, skill_counts, color="darkorange")
            axs[1, 0].set_title("Most In-Demand Skills", fontsize=16)
            axs[1, 0].set_xlabel("Skills", fontsize=14)
            axs[1, 0].set_ylabel("Frequency", fontsize=14)
            axs[1, 0].tick_params(axis="x", rotation=45)

            # Sentiment Distribution Plot
            if "Sentiment_Label" in filtered_jobs.columns:
                sentiment_counts.plot(kind="bar", color=["lightgreen", "lightcoral", "lightgray"], ax=axs[1, 1])
                axs[1, 1].set_title("Sentiment Distribution", fontsize=16)
                axs[1, 1].set_xlabel("Sentiment", fontsize=14)
                axs[1, 1].set_ylabel("Number of Job Descriptions", fontsize=14)

            st.pyplot(fig)

            # Word Cloud
            st.subheader("Word Cloud for Job Descriptions")
            wordcloud = WordCloud(width=1000, height=500, background_color="white").generate(" ".join(all_tokens))
            fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
            ax_wc.imshow(wordcloud, interpolation="bilinear")
            ax_wc.axis("off")
            st.pyplot(fig_wc)

    else:
        st.warning(f"No data available for {selected_experience} experience level.")

    # AI Insights Section
    st.write("### Future Skill Trends Prediction")
    user_query = st.text_area("Enter a job description or query:")
    if st.button("Get AI Insights"):
        if user_query.strip():
            insights = get_ai_insights(user_query)
            if insights:
                st.subheader("AI Insights on Future Skills")
                st.write(insights)
        else:
            st.warning("Please enter a query.")
