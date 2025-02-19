# 📌 Natural Language Processing (NLP) Basics

## **1. Tokenization**

### **🔹 What is Tokenization?**

Tokenization is the process of breaking down text into smaller components, known as tokens. These tokens can be words, sentences, or subwords.

### **🔹 Types of Tokenization**

- **Word Tokenization:** Splitting text into individual words.
- **Sentence Tokenization:** Splitting text into sentences.

### **📌 Example: Word Tokenization**

#### **Input:**

```text
"AI is revolutionizing the world of technology! NLP is a key part of AI."
```

#### **Output:**

```text
['AI', 'is', 'revolutionizing', 'the', 'world', 'of', 'technology', '!', 'NLP', 'is', 'a', 'key', 'part', 'of', 'AI', '.']
```

### **💻 Code Example (NLTK - Word and Sentence Tokenization)**

```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt')  # Download tokenizer models

text = "AI is revolutionizing the world of technology! NLP is a key part of AI."

# Word Tokenization
word_tokens = word_tokenize(text)
print("Word Tokens:", word_tokens)

# Sentence Tokenization
sentence_tokens = sent_tokenize(text)
print("Sentence Tokens:", sentence_tokens)
```

---

## **2. Stopwords Removal**

### **🔹 What are Stopwords?**

Stopwords are common words that do not add significant meaning to a sentence. Examples include _is, the, and, of, etc._

### **📌 Example: Stopword Removal**

#### **Input:**

```text
"This is an amazing NLP tutorial for beginners!"
```

#### **Output (After Removing Stopwords):**

```text
['amazing', 'NLP', 'tutorial', 'beginners', '!']
```

### **💻 Code Example (NLTK - Stopwords Removal)**

```python
from nltk.corpus import stopwords

nltk.download('stopwords')  # Download stopwords dataset

stop_words = set(stopwords.words('english'))

# Example text
tokens = word_tokenize("This is an amazing NLP tutorial for beginners!")

# Removing stopwords
filtered_tokens = [word for word in tokens if word.lower() not in stop_words] # used list comprehension of python

print("Original Tokens:", tokens)
print("Filtered Tokens (Stopwords removed):", filtered_tokens)
```

---

## **3. Stemming**

### **🔹 What is Stemming?**

Stemming reduces words to their base or root form by chopping off suffixes. The output may not always be a real word.

### **📌 Example: Stemming**

| Word      | Stemmed Form |
| --------- | ------------ |
| Running   | run          |
| Studies   | studi        |
| Beautiful | beauti       |

### **💻 Code Example (NLTK - Porter Stemmer)**

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

words = ["running", "studies", "beautiful", "flies", "crying"]
stemmed_words = [stemmer.stem(word) for word in words]  # used list comprehension of python

print("Stemmed Words:", stemmed_words)
```

---

## **4. Lemmatization**

### **🔹 What is Lemmatization?**

Lemmatization is a more accurate way to get the base form of a word, considering its meaning and grammar.

### **📌 Example: Lemmatization**

| Word    | Lemmatized Form |
| ------- | --------------- |
| Running | run             |
| Studies | study           |
| Mice    | mouse           |
| Went    | go              |

### **💻 Code Example (NLTK - WordNet Lemmatizer)**

```python
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')  # Download dataset for lemmatization

lemmatizer = WordNetLemmatizer()

words = ["running", "studies", "mice", "went", "crying"]
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

print("Lemmatized Words:", lemmatized_words)
```

---

## **5. Stemming vs. Lemmatization**

| Feature | Stemming                         | Lemmatization              |
| ------- | -------------------------------- | -------------------------- |
| Output  | Can produce non-dictionary words | Always produces real words |
| Speed   | Faster                           | Slower                     |
| Example | "Studies" → "Studi"              | "Studies" → "Study"        |

### **🔹 When to Use What?**

✔ **Stemming** is useful when speed is more important than accuracy.  
✔ **Lemmatization** is preferred when meaning and correct grammar are required.

---

## **🎯 Conclusion**

1️⃣ **Tokenization** → Splitting text into words/sentences.  
2️⃣ **Stopwords Removal** → Removing common words that add no meaning.  
3️⃣ **Stemming** → Cutting words to their root form (not always a real word).  
4️⃣ **Lemmatization** → Converting words to meaningful root forms.

---

🚀 **Next Steps:** Try applying these techniques to real-world text data and analyze how they impact NLP applications like sentiment analysis, chatbot responses, and text summarization!

## Author

**Harshit Soni**  
GitHub: [Harshit-Soni78](https://github.com/Harshit-Soni78)

---
Made with ❤️ by Harshit Soni
