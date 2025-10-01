import requests
import time
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

SERVER_URL = "http://paffenroth-23.dyn.wpi.edu:8014/"
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
MONITOR_INTERVAL = 120

def monitor_server():
    try:
        response = requests.get(SERVER_URL, timeout=10)
        if response.status_code == 200:
            return True, "Status: OK"
        else:
            return False, f"Status: HTTP Error {response.status_code}"
    except Exception as e:
        return False, "Status: DOWN"
    
def send_discord_alert(server_up, status):
    if not server_up:
        data = {
            'content': "The application's server is currently down.",
            'username': "Server Monitor",
            'embeds': [
                {
                    'description': status,
                    'title': "Server Status"
                }
            ]
        }

        response = requests.post(DISCORD_WEBHOOK_URL, json=data)

        try:
            response.raise_for_status()
        except Exception as e:
            print(f"Failed to send discord notification: {str(e)}")
        else:
            print("Successfully sent discord notificaion")

    return

def main():
    print("Beginning monitor for Multimodal Chatbot")
    print(f"Monitor interval: {MONITOR_INTERVAL} seconds")

    while True:
        server_up, status = monitor_server()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Timestamp: {timestamp}")
        print(f"{status}")
        send_discord_alert(server_up, status)
        time.sleep(MONITOR_INTERVAL)

if __name__ == "__main__":
    main()