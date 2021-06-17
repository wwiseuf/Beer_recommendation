  
import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'beer': "Smuttynose Octoberfest"})

print(r.json())