import streamlit as st
from notes_generator import fetch_wikipedia_content, summarize_text

# Streamlit UI Setup
st.set_page_config(page_title="📚 Quikk Notes AI", layout="wide")

st.title("📚 Quikk Notes AI")
st.write("Enter a topic, and get concise, summarized notes from Wikipedia!")

# Initialize session state for summary
if "summary" not in st.session_state:
    st.session_state.summary = ""

# User Input
topic = st.text_input("🔍 Enter Topic")

if st.button("Generate Notes"):
    st.write("🔄 Fetching & Summarizing... Please wait.")

    # Fetch & Summarize Content
    extracted_content = fetch_wikipedia_content(topic)
    
    if extracted_content.startswith("Wikipedia page not found"):
        st.error(extracted_content)
        st.session_state.summary = ""  # Reset summary if error occurs
    else:
        st.session_state.summary = summarize_text(extracted_content)  # Store summary in session

# Ensure summary is displayed only if it exists
if st.session_state.summary:
    st.subheader("📌 Summarized Notes")
    st.write(st.session_state.summary)  # Display summary only once

st.markdown("---")
st.write("🚀 Built with Hugging Face Transformers and Streamlit")
st.write("Wikipedia Summarizer Project By [Mohit Gaur](https://www.linkedin.com/in/SWEMohitGaur)")
st.write("Special Thanks to [Harshit Soni BOSS](https://www.linkedin.com/in/harshit-soni78/) and AI PlaneTech")
