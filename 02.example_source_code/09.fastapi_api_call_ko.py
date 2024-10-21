import requests
import json

# The URL where your FastAPI server is running
API_URL = "http://localhost:8000/generate"

# The request payload
payload = {
    # "prompt": "Once upon a time in a digital world,",
    "messages":[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "대한민국의 수도는 어디인가요? 그리고 그 수도에 대해 설명해주세요."}, 
    ],
    "max_tokens": 1000,
    "temperature": 0.8,
    "top_p": 0.95
}

# Send a POST request to the API
response = requests.post(API_URL, json=payload)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    result = response.json()
    
    # Print the generated text
    print("Generated Text:")
    print(result['generated_text'])
else:
    print(f"Error: {response.status_code}")
    print(response.text)