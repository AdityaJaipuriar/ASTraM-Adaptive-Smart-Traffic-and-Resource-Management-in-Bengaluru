import requests

# 1. Paste your key here
API_KEY = "prcxkqskbpysewcmqmdljgprqviuivdubumh"

# 2. Coordinates
SILK_BOARD = "77.6229,12.9172"
INDIRANAGAR = "77.6389,12.9719"
HSR_BLOCKED = "77.6451,12.9218"

print("--- TESTING STANDARD ROUTE ---")
base_url = f"https://apis.mappls.com/advancedmaps/v1/{API_KEY}/route_adv/driving/{SILK_BOARD};{INDIRANAGAR}"
response = requests.get(base_url)
print(f"Status Code: {response.status_code}")
print(f"Response: {response.text[:200]}...\n")

print("--- TESTING DIVERSION (AVOID) ROUTE ---")
avoid_url = base_url + f"?avoid={HSR_BLOCKED}"
response_avoid = requests.get(avoid_url)
print(f"Status Code: {response_avoid.status_code}")
print(f"Response: {response_avoid.text[:200]}...")