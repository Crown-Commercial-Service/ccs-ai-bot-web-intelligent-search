import requests
import json 
import os 

# for local testing 
API_URL = "http://localhost:8000/search"
API_KEY = os.environ.get("API_KEY", "default_dev_key")

# Test query

test_query = {"query": "In RM6098, what guidelines must suppliers follow during the standstill period before finalizing a contract award under the framework?"}

def test_search_api():
    headers = {"X-API-Key": API_KEY,
               "Content-Type": "application/json"}
    
    try:
        response = requests.post(API_URL, 
                                 headers=headers,
                                 data = json.dumps(test_query))
        
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            print("Sucess! API Response:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Error: {response.text}")

    except Exception as e:
        print(f"Error: {e}")


def test_deployed_api(deployed_url):

    headers = {"X-API-Key": API_KEY,
              "Content-Type": "application/json"}
    
    try:
        response = requests.post(f"{deployed_url}/search",
                                 headers=headers,
                                 data = json.dumps(test_query))
        print(f"Status Code:  {response.status_code}")

        if response.status_code == 200:
            print("Sucess! Deployed API Response:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Error: {response.text}")
    
    except Exception as e:
        print(f"Exception occurred: {e}")


if __name__ == "__main__":
    print("Testing local API.....")
    test_search_api()


    # URL_link = ""
    # test_deployed_api(URL_link)