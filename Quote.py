# Feature 3: Inspirational quote 

import requests, json
import pyttsx3 as speak

# Call the api and store the response
response = requests.get("https://api.quotable.io/random?,famous-quotes").json()

# Generate statement to be spoken
statement3 = response["content"] + ". This quote was by " + response["author"]

# Speak the statement
engine = speak.init()
engine.say(statement3)
engine.runAndWait()
