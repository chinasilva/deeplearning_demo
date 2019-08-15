import webbrowser
import json
import torch
import base64

from flask import Flask, request, jsonify, json, render_template
from core import core
from src.main import device_fun


app = Flask(__name__)

device=device_fun() 

net = torch.jit.load("models/net.pth")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/start', methods=['post'])
def start():
    data = json.loads(request.get_data())
    img = base64.b64decode(data["img"])
    img_path = "static/images/example.jpg"

    with open(img_path, 'wb') as file:
        file.write(img)

    result_data = core(net, img_path)
    return json.dumps(result_data)

if __name__ == '__main__':


    # webbrowser.open("http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5001)

