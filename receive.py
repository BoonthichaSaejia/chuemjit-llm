import requests

api_url = 'https://chuemjit-2024.et.r.appspot.com/api/query'
query_params = {
    'query': 'เขาจะทิ้งเธอไปแล้ว',
    'threshold': 0.8
}

response = requests.get(api_url, params=query_params)

if response.status_code == 200:
    json_response = response.json()
    print("Received Response:")
    print(json_response)
else:
    print(f"Error: Request failed with status code {response.status_code}")