from flask import Flask, request, jsonify

import platform
if platform.system() == "Darwin":
    from keyboard.macos import PressKey, ReleaseKey
elif platform.system() == "Windows":
    from keyboard.windows import PressKey, ReleaseKey
from keyboard.generic import W, A, S, D

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from Learning.agent import Agent

app = Flask(__name__)

driver = Agent()

new_press_keys = set()

@app.route("/api/data", methods=["POST"])
def handle_data():
    data = request.get_json()

    if not data :
        return jsonify({"error": "No JSON payload provided"}), 400
    
    ## Clear all pressed keys
    old_pressed_keys = new_press_keys.copy()
    new_press_keys.clear()

    ## Feed agent with data and get action
    keys = driver.feed(data)

    # Press and Release keys accordingly
    for key in keys:
        print(key)
        new_press_keys.add(eval(key))

    for k in new_press_keys - old_pressed_keys:
        PressKey(k)
    for k in old_pressed_keys - new_press_keys:
        ReleaseKey(k)

    return jsonify({"status": "success", "received": data}), 200

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
