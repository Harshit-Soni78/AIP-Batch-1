import os
import tkinter as tk
from tkinter import scrolledtext
from openai import OpenAI

# Loading API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("Error: OPENAI_API_KEY is not set. Please set it as an environment variable.")

client = OpenAI(api_key=api_key)

conversation_history = [{"role": "system", "content": "You are a helpful learning assistant."},
                        {"role": "user", "content": "Q: Sum of 2, 1 and 4 is?"},
                        {"role":"assistant", "content": "A: 7"}]

# Function to get AI response
def ask_ai():
    query = user_input.get().strip()
    if not query:
        return

    conversation_history.append({"role": "user", "content": query})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversation_history
        )
        reply = response.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": reply})
        
    except Exception as e:
        print(f"Error: {e}")
        reply = "Sorry, I encountered an issue while processing your request."

    # Display chat history
    chat_history.config(state=tk.NORMAL)
    chat_history.insert(tk.END, "You:  " + query + "\n", "user")
    chat_history.insert(tk.END, "AI:  " + reply + "\n\n", "ai")
    chat_history.yview(tk.END)  # Auto-scroll to latest message
    chat_history.config(state=tk.DISABLED)

    # Clear input field
    user_input.delete(0, tk.END)
    user_input.focus_set()

# Create main window
root = tk.Tk()
root.title("AI Study Assistant")
root.geometry("700x600")  # Increased window size
root.minsize(600, 500)  # Prevents making the window too small
root.configure(bg="#f0f0f0")  # Light gray background

# Styling
font_style = ("Arial", 14)
button_style = {"font": font_style, "bg": "#4CAF50", "fg": "white", "padx": 10, "pady": 5}

# Chat history (Scrollable text box)
chat_history = scrolledtext.ScrolledText(
    root, wrap=tk.WORD, state=tk.DISABLED, height=25, width=80, font=font_style, bg="white", fg="#333333"
)
chat_history.grid(row=0, column=0, columnspan=2, padx=20, pady=20, sticky="nsew")

# Input field
user_input = tk.Entry(root, font=font_style)
user_input.grid(row=1, column=0, padx=20, pady=10, ipady=5, sticky="ew")

# Send button (stretched properly)
send_button = tk.Button(root, text="Ask AI", command=ask_ai, **button_style)
send_button.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

# Define chat history styles
chat_history.tag_configure("user", foreground="magenta")
chat_history.tag_configure("ai", foreground="blue")

# Configure row and column weights to ensure resizing works
root.grid_rowconfigure(0, weight=1)  # Chat history expands vertically
root.grid_columnconfigure(0, weight=1)  # Input field expands horizontally
root.grid_columnconfigure(1, weight=0)  # Button remains fixed width

# Bind Enter key to send message
root.bind("<Return>", lambda event: ask_ai())

# Run Tkinter event loop
root.mainloop()
