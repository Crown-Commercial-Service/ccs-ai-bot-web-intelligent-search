import requests
import json
import os

# For local testing
API_URL = "http://localhost:8000/search"
API_KEY = "web-intelligentSearchNS-20250403"  # Same key as in your API code

# Test query
test_query = {
    "query": "Which framework provides AI support?"
}

def test_search_api():
    """Test the search API endpoint"""
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=test_query  # Using json parameter instead of data
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("Success! API Response:")
            response_data = response.json()  # Already parsed JSON
            # with open("api_response.json", "w") as f:
            #     json.dump(response_data, f, indent=2)
            return response_data  # Return parsed dict instead of string
        else:
            print(f"Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"Exception occurred: {e}")
        return None

def test_deployed_api(deployed_url):
    """Test the deployed API"""
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            f"{deployed_url}/search",
            headers=headers,
            json=test_query  # Using json parameter instead of data
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("Success! Deployed API Response:")
            response_data = response.json()  # Already parsed JSON
            response_data = json.dumps(response_data, indent=2)
            return response_data  # Return parsed dict instead of string
        else:
            print(f"Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"Exception occurred: {e}")
        return None

if __name__ == "__main__":
    # For testing deployed API
    
    print("\nTesting local API...")
    output = test_search_api()
    print('yes done')

    if output:
        print(output)
        print(type(output))

    azure_url = "https://azd-uks-ai-webpilot-intelligentsearch-api-a3ewg3deabbmasab.uksouth-01.azurewebsites.net"
    print("\nTesting deployed API...")
    output = test_deployed_api(azure_url)

    if output:
        print(output)
        print(type(output))

    