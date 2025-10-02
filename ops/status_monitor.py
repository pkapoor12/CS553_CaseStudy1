import requests
import time
from datetime import datetime
from dotenv import load_dotenv
import os
import pexpect
import shutil

load_dotenv()

SERVER_URL = "http://paffenroth-23.dyn.wpi.edu:8014/"
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
SSH_KEY_PASSPHRASE = os.getenv("SSH_KEY_PASSPHRASE")
SUDO_PASSWORD = os.getenv("SUDO_PASSWORD")
MONITOR_INTERVAL = 120
REDEPLOY_DELAY = 1200

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

def redeploy():
    # Copy deploy.sh from current dir to home dir
    src = os.path.join(os.getcwd(), "deploy.sh")
    dst = os.path.expanduser("~/deploy.sh")
    try:
        shutil.copy2(src, dst)
        os.chmod(dst, 0o755)  # make sure it’s executable
        print(f"Copied deploy.sh to {dst}")
    except Exception as e:
        print(f"Failed to copy deploy.sh: {e}")
        return

    # Run the one in home dir
    child = pexpect.spawn(dst, encoding="utf-8")

    while True:
        i = child.expect([
            r"Enter passphrase for key.*:",                     # generic ssh prompt
            r"Enter passphrase for .*secure_key.*:",            # ssh-agent prompt with path
            r"\[sudo\] password for.*:",                        # sudo password
            r"Are you sure you want to continue connecting.*",  # trust host prompt
            pexpect.EOF,
            pexpect.TIMEOUT
        ], timeout=120)

        if i == 0:
            child.sendline(SSH_KEY_PASSPHRASE)
        elif i == 1:
            child.sendline(SSH_KEY_PASSPHRASE)
        elif i == 2:
            child.sendline(SUDO_PASSWORD)
        elif i == 3:
            child.sendline("yes")
        elif i == 4:  # finished
            break
        elif i == 5:  # timeout
            print("Timeout while waiting for deploy.sh")
            break

    child.close()
    print("Redeploy finished with exit status:", child.exitstatus)

def main():
    print("Beginning monitor for Multimodal Chatbot")
    print(f"Monitor interval: {MONITOR_INTERVAL} seconds")

    while True:
        server_up, status = monitor_server()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Timestamp: {timestamp}")
        print(f"{status}")
        
        if server_up:
            last_downtime = None
        else:
            if last_downtime is None:
                last_downtime = time.time()
                send_discord_alert(server_up, status)
            else:
                elapsed_time = time.time() - last_downtime
                if elapsed_time > REDEPLOY_DELAY:
                    print(f"App down for more than {REDEPLOY_DELAY} seconds, starting redeployment")
                    send_discord_alert(server_up, status=f"App down for more than {REDEPLOY_DELAY/60} minute(s), starting redeployment")
                    redeploy()
                    last_downtime = None

        time.sleep(MONITOR_INTERVAL)

if __name__ == "__main__":
    main()