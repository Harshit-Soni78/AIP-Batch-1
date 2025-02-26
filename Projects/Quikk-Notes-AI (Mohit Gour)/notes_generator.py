import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# Load the NLP Summarization Model
summarizer = pipeline("summarization")

def fetch_wikipedia_content(topic):
    """
    Fetches relevant text content from Wikipedia for the given topic.
    
    :param topic: The topic to search on Wikipedia
    :return: Extracted relevant text content
    """
    url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
    response = requests.get(url)

    if response.status_code != 200:
        return "Wikipedia page not found."

    # Parse HTML using BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all paragraphs
    paragraphs = soup.find_all("p")

    # Extract relevant text content (ignoring empty paragraphs)
    content = ""
    for para in paragraphs:
        text = para.get_text().strip()
        if text and len(text) > 50:  # Avoid very short or empty lines
            content += text + "\n\n"

    return content if content else "No relevant content found."

def summarize_text(text):
    """
    Summarizes the extracted Wikipedia content using an NLP model.
    
    :param text: The extracted Wikipedia content
    :return: Summarized text
    """
    if not text.strip():
        return "No content available for summarization."

    # Limiting input text length for better summarization
    text = text[:3000]  # Summarizing only the first 3000 characters

    summary = summarizer(text, max_length=400, min_length=100)
    return summary[0]['summary_text']