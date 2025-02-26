import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

analyzer=SentimentIntensityAnalyzer()

def get_sentiment_score(score, sentiment):
    if sentiment['neu'] > 0.7:  # If more than 70% neutral, force neutrality
        return 0
    elif score >= 0.05:
        return 1
    elif score <= -0.05:
        return -1
    else:
        return 0

def statistical_analyzer(reviews,details=False):
    posCount,negCount,neuCount=0,0,0
    responseList=[]
    for review in reviews:
        result=analyzer.polarity_scores(review)
        response=get_sentiment_score(result['compound'],result)
        responseList.append(response)
        if(details):
            print(result)
            print(f"Text: {review}\nVADER: {response}\nResult:{response}\n\n")

    for response in responseList:
        if response==1:
            posCount+=1
        elif response==-1:
            negCount+=1
        else:
            neuCount+=1

    return posCount,negCount,neuCount

def graphical(pos,neg,neu):
    context=["Positive","Negative","Neutral"]
    percentage=[pos,neg,neu]
    colors = ['#4CAF50', '#F44336', '#9E9E9E']
    
    plt.title("Graphical Representation!")
    plt.pie(percentage,labels=context,autopct="%.1f%%",colors=colors)
    plt.show()


commentsSample1 = [
    "This app is absolutely amazing! So smooth and easy to use.",  # Positive 😊
    "I hate the latest update. It ruined everything!",  # Negative 😡
    "The UI is decent, but nothing special.",  # Neutral 😐
    "Great features, but some bugs make it frustrating to use.",  # Mixed (Positive + Negative)
    "It’s okay, but I wouldn’t recommend it over other apps.",  # Neutral 😐
    "I love how intuitive the design is!",  # Positive 😊
    "Too many ads! It’s annoying and ruins the experience.",  # Negative 😡
    "The app works fine, but it lacks innovation.",  # Neutral 😐
    "This has potential, but it needs serious improvements.",  # Mixed (Neutral + Negative)
    "Best app I’ve used in a long time! 10/10 would recommend!",  # Positive 😊
]
commentsSample2=[
    # Positive Reviews 😊
    "This app is absolutely amazing! So smooth and easy to use.",  
    "I love how intuitive the design is!",  
    "Best app I’ve used in a long time! 10/10 would recommend!",  
    "Great features, and the customer support is excellent!",  
    "The latest update fixed all the bugs. Works perfectly now!",  
    "Super fast and reliable. Exactly what I needed.",  
    "Fantastic user experience. Highly recommended!",  

    # Negative Reviews 😡
    "I hate the latest update. It ruined everything!",  
    "Too many ads! It’s annoying and ruins the experience.",  
    "The app crashes every time I try to open it. Useless!",  
    "Terrible UI. Hard to navigate and confusing.",  
    "Customer support is completely unresponsive!",  
    "I regret downloading this. Waste of time.",  
    "So many bugs! It’s frustrating to use.",  

    # Neutral Reviews 😐
    "The UI is decent, but nothing special.",  
    "It’s okay, but I wouldn’t recommend it over other apps.",  
    "The app works fine, but it lacks innovation.",  
    "Nothing extraordinary, just an average app.",  
    "This has potential, but it needs serious improvements.",  
    "Some features are good, but others are pointless."  
]
commentsSample3=["Some good moments, but overall just another average movie.",
"Not bad, not great. Just… fine. I’d watch it on a lazy Sunday.",
"Good performances, but the story felt kinda weak.",
"I liked the visuals, but the pacing was all over the place."]

p,neg,neu=statistical_analyzer(commentsSample3,True)
print(p,neg,neu)
graphical(p,neg,neu)
    
