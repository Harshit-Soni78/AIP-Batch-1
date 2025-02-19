from nltk.chat.util import Chat, reflections

pairs = [
    ["hi|hello", ["Hello!", "Hi there!"]],
    ["how are you?", ["I'm a bot, I'm always good!"]],
    ["what's your name?", ["I'm a simple chatbot!"]],
]

chatbot = Chat(pairs, reflections)
chatbot.converse()
