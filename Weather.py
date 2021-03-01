# All in 1 now

import geocoder
import geopy
from geopy.geocoders import Nominatim
import requests, json 
import pytemperature as ptp
import pyttsx3 as speak

# Get IP
g = geocoder.ip('me')

# Get location from IP
geolocator = Nominatim(user_agent="geoapiExercises")

latitude = str(geocoder.ip('me').latlng[0])
longitude = str(geocoder.ip('me').latlng[1])

location = geolocator.reverse(latitude+","+longitude) 

# Use location for city and weather

key = "ENTERYOURKEYHERE"

language = 'en'

city_name = location[0].split(",")[3]

base_url = "http://api.openweathermap.org/data/2.5/weather?"

complete_url = base_url + "appid=" + key + "&q=" + city_name 

response = requests.get(complete_url) 

x = response.json() 

# Get temperature and create a statement to be spoken

f = ptp.k2f(x["main"]["temp"])
print(f)

statement = "The temperature in " + str(x["name"]) + " is " + str(f) + " degrees farenheit"

# Speak the statement

engine = speak.init()
engine.say(statement)
engine.runAndWait()
