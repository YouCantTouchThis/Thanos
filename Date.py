# Feature 2: Date

from datetime import date
import pyttsx3 as speak


# Get the date
today = date.today()
strdate = today.strftime("%B %d, %Y")

# Create the statement to be spoken
statement2 = "Today is " + strdate + ", have a great day"

# Speak the statement
engine = speak.init()
engine.say(statement2)
engine.runAndWait()
