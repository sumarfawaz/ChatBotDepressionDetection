from flask import Flask, request, jsonify
import random
import json
import openai
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from flask_cors import CORS  # Import the CORS module

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes in your Flask app

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "EmoAssist"
chat_log = []

@app.route('/api/chat', methods=['POST'])
def chat():
    global response_text
    data = request.get_json()
    user_message = data.get('message')

    sentence = tokenize(user_message)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response_text = random.choice(intent['responses'])
    else:
        openai.api_key = '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'
        chat_log.append({"role": "user", "content": f'"{user_message}"'})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=chat_log
        )
        response_text = response['choices'][0]['message']['content']

    return jsonify({'response': response_text})

if __name__ == '__main__':
    app.run(debug=True)
