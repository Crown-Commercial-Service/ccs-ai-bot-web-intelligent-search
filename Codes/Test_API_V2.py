import requests

response = requests.get("https://azd-uks-ai-webpilot-intelligentsearch-api-a3ewg3deabbmasab.uksouth-01.azurewebsites.net/health")

print("Status Code:", response.status_code)
print("Response Body:", response.text)