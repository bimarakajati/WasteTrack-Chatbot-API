import numpy as np
import argparse, datetime, io, pickle, os, random, string, torch
from PIL import Image
from keras.models import load_model
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"

# Define your model and tokenizer here
chatbot_model = load_model('model/model.h5')
tokenizer = pickle.load(open('model/tokenizer.pkl','rb'))
max_sequence_length = pickle.load(open('model/max_sequence_length.pkl','rb'))
le = pickle.load(open('model/le.pkl','rb'))
responses = pickle.load(open('model/responses.pkl','rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/chat', methods=['POST'])
def chat():
    texts_p = []
    prediction_input = request.form['message']

    # Remove punctuation and convert to lowercase
    prediction_input = [letter.lower() for letter in prediction_input if letter not in string.punctuation]
    prediction_input = ''.join(prediction_input)
    texts_p.append(prediction_input)

    # Tokenize and pad the input
    prediction_input = tokenizer.texts_to_sequences(texts_p)
    prediction_input = np.array(prediction_input).reshape(-1)
    prediction_input = pad_sequences([prediction_input], max_sequence_length)

    # Get the model's output prediction
    output = chatbot_model.predict(prediction_input)
    output = output.argmax()

    # Find the corresponding response based on the predicted tag
    response_tag = le.inverse_transform([output])[0]
    response = random.choice(responses[response_tag])

    return {'response': response}

@app.route('/detection', methods=["POST"])
def predict():
    if not request.method == "POST":
        return

    if request.files.get("image"):
        image_file = request.files["image"]
        image_bytes = image_file.read()
        img = Image.open(io.BytesIO(image_bytes))
        results = model(img, size=640) # reduce size=320 for faster inference

        results.render()  # updates results.imgs with boxes and labels
        now_time = datetime.datetime.now().strftime(DATETIME_FORMAT)
        img_savename = f"static/{now_time}.png"
        Image.fromarray(results.ims[0]).save(img_savename)
        gambar = {'image': img_savename}

        hasil = results.pandas().xyxy[0].to_dict(orient='records')
        # hasil = results.pandas().xyxy[0].to_json(orient="records")
        hasil.insert(0, gambar)
        return hasil

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Flask api exposing yolov5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    parser.add_argument('--model', default='yolov5s', help='model to run, i.e. --model yolov5s')
    args = parser.parse_args()

    model = torch.hub.load('yolov5', 'custom', path='model/last.pt', source='local')
    app.run(debug=True, port=os.getenv("PORT", default=5000))