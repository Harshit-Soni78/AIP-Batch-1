# Subscriptify - Website Subscription Checker Bot

## Overview
Subscriptify is an AI-powered chatbot that determines whether a website is free, freemium, or requires a paid subscription. It uses **spaCy** for NLP-based keyword detection and **BeautifulSoup** for web scraping. The project now features a **Streamlit-based GUI** for easy user interaction.

## Features
✅ Fetches and extracts visible text from a website.
✅ Uses **spaCy** NLP to detect subscription-related terms.
✅ Identifies websites as **free, freemium, or paid**.
✅ **Streamlit Web UI** for an interactive experience.
✅ Error handling for better reliability.

## Tech Stack
- **Python** 🐍
- **spaCy** (Natural Language Processing)
- **BeautifulSoup** (Web Scraping)
- **Requests** (HTTP requests handling)
- **Streamlit** (Graphical User Interface)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/subscriptify.git
   cd subscriptify
   ```
2. Create a virtual environment (Optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download spaCy's English model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage
### **Using the Graphical Interface (Streamlit Web UI)**
To run the GUI version:
```bash
streamlit run subscriptify_gui.py
```
A web interface will open where you can paste the website URL and check its subscription status.

## Future Enhancements
- Improve UI design with **better interactivity** in Streamlit.
- Implement **Selenium** for dynamic page scraping.
- Create a **dashboard** to track website data over time.
- Integrate **AI-powered text summarization** to provide concise subscription details.
- Develop a **browser extension** for real-time subscription status checking.
- Add **multi-language support** to analyze international websites.
- Implement a **machine learning model** to improve subscription classification accuracy.

## Contributing
Feel free to fork the repository and submit pull requests. 🚀


