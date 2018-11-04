import sys
from flask import Flask, render_template, request, redirect, url_for, jsonify
import settings

sys.path.append(settings.BASE_DIR)

from dialogManager import DialogueManager

app = Flask(__name__)


# home page
@app.route('/')
def index():
    return render_template('chatBotTemplate.html')


def get_response(text):
    chatBot = DialogueManager()
    chatBot.createBot()

    if text:
        response = chatBot.getAnswer(text)

        return response


@app.route('/send_message', methods=['POST'])
def send_message():
    message = request.form['message']
    response_text = get_response(message)
    response_text = {"message":  response_text}

    return jsonify(response_text)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
