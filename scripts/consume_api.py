import requests

base_url = 'http://127.0.0.1:8000'

start_response = requests.get(f'{base_url}/start_server')
print('Status Code: ', start_response.status_code)
print('JSON Response: ', start_response.json())

payload = {
    "review_text": "M-Pesa iko sawa lakini inakataa login 😤",
    "rating": 2,
    "thumbs_up": 3,
    "is_code_mixed": True,
    "is_sheng_like": True
}

predict_response = requests.post(f"{base_url}/predict", json=payload)

print("Status Code:", predict_response.status_code)
print("RAW Response:", predict_response.text)  # <-- add this
# Only try json() if it's actually JSON
try:
    print("JSON Response:", predict_response.json())
except Exception as e:
    print("Could not parse JSON:", e)
    
print("Status Code:", predict_response.status_code)
print("JSON Response:", predict_response.json())
