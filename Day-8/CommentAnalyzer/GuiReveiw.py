import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import tkinter as tk
import os
from tkinter import scrolledtext
from tkinter import messagebox
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

analyzer=SentimentIntensityAnalyzer()


def get_sentiment_score(score):
    if(score>=0.05):
        return 1
    elif(score<=-0.05):
        return -1
    else:
        return 0


def statistical_analyzer(reviews,details=False):
    posCount,negCount,neuCount=0,0,0
    responseList=[]
    for review in reviews:
        result=analyzer.polarity_scores(review)
        response=get_sentiment_score(result['compound'])
        responseList.append(response)
        if(details):
            print(result)
            print(f"Text: {review}\nResult:{response}\n\n")

    for response in responseList:
        if response==1:
            posCount+=1
        elif response==-1:
            negCount+=1
        else:
            neuCount+=1
    # print(responseList,"\n\n\n",len(responseList))
    return posCount,negCount,neuCount

def graphical(pos,neg,neu):
    context=["Positive","Negative","Neutral"]
    percentage=[pos,neg,neu]
    colors = ['#4CAF50', '#F44336', '#9E9E9E']
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(percentage, labels=context, autopct="%.1f%%", colors=colors, startangle=90)
    ax.set_title("Sentiment Distribution")

    for widget in chart_frame.winfo_children():
        widget.destroy()  # Clear previous chart

    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()
    



#------------------tkinter part--------------------------------------------

def analyze_review():
    input_text=text_area.get("1.0",tk.END).strip()

    if(not input_text):
        messagebox.showwarning("Warning!","Please enter reviews for analyzed Report")
        return
    reviews=input_text.split("\n")
    pos,neg,neu=statistical_analyzer(reviews)


    #updating
    result_text.config(state=tk.NORMAL) if 'tk' in globals() else print("Tk not defined!")

    result_text.delete("1.0",tk.END)
    result_text.insert(tk.END, f"Positive Reviews: {pos}\n")
    result_text.insert(tk.END, f"Negative Reviews: {neg}\n")
    result_text.insert(tk.END, f"Neutral Reviews: {neu}\n")
    result_text.config(state=tk.DISABLED)

    graphical(pos,neg,neu)


#------------------------------setting up Tkinter---------------------------------------
    
root=tk.Tk()
root.title("Review Analysis")
root.geometry("900x600")
try:
    

    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script location
    image_path = os.path.join(script_dir, "back.jpeg")  # Assuming image is in the same folder

    if os.path.exists(image_path):
        image = Image.open(image_path)
        image = image.resize((900, 600), Image.LANCZOS)  # Resize to match window size
   
        bg_image = ImageTk.PhotoImage(image)

        #Set Background Label
        bg_label = tk.Label(root, image=bg_image)
        bg_label.place(relwidth=1, relheight=1)  # Stretch to full window size
    else:
        print("Warning: Background image not found. Running without it.")

 
except:
    print("Error loading the file")

# Place widgets on top of the background
frame = tk.Frame(root, bg="white", bd=5)  # A frame to hold widgets (improves visibility)
frame.place(relx=0.5, rely=0.05, anchor="n")  # Centered at top

tk.Label(root,text="Enter Set of Reviews",font=("Times New Roman",25,"bold"),fg="white",bg="black").pack(pady=10)
text_area=scrolledtext.ScrolledText(root,width=70,height=8,font=("Helvetica", 18, "italic"))
text_area.pack(pady=20,padx=20)

#Analyze and show result button
analyze_button=tk.Button(root,text="Submit",font=("Helvetica", 18, "italic"),command=analyze_review)
analyze_button.pack(pady=10)



# Frame for Result & Chart Side by Side
result_chart_frame = tk.Frame(root, bg="black")
result_chart_frame.pack(fill="both", expand=True)


# Results Section (Left Side) without Scroll
result_text = tk.Text(result_chart_frame, width=5, height=1, font=("Arial", 12), state=tk.DISABLED)
result_text.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")  # Expand with window


# Make it expand with window
#chart
chart_frame = tk.Frame(result_chart_frame, bg="pink")
chart_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

# Adjust column weights so they resize proportionally
result_chart_frame.columnconfigure(0, weight=1)
result_chart_frame.columnconfigure(1, weight=1)
result_chart_frame.rowconfigure(0, weight=1)

root.mainloop()