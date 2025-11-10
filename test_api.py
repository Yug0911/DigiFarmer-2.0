import requests
import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv('OPENWEATHER_API_KEY')
print('API Key loaded:', bool(key))
print('Key value:', key)

if key:
    response = requests.get(f'https://api.openweathermap.org/data/2.5/weather?lat=28.6139&lon=77.2090&appid={key}&units=metric')
    print('Status Code:', response.status_code)
    if response.status_code == 200:
        data = response.json()
        print('SUCCESS! Weather data:')
        print(f"Temperature: {data['main']['temp']}°C")
        print(f"Humidity: {data['main']['humidity']}%")
        print(f"Wind Speed: {data['wind']['speed']} m/s")
    else:
        print('Error response:', response.text)
else:
    print('No API key found!')