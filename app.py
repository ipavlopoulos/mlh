import os
from flask import Flask, render_template
import torch
import numpy as np
import io
import os
import requests
import boto3
import json
from timm import create_model
from PIL import Image
from torch.nn.functional import softmax
from torchvision import transforms
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS # Required for cross-origin requests from the front-end

DEVICE = torch.device("cpu")
MODEL_PATH = "best_swin_nocurr_seed42.pth"
LABELS = ["Healthy", "Patient"]
IMG_SIZE = (224, 224)

# Model loading and setup
swin_model = None

try:
    print(f"Attempting to load model from: {MODEL_PATH}")
    swin_model = create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=2)
    # Check if the model file exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    swin_model.load_state_dict(state_dict)
    swin_model.to(DEVICE)
    swin_model.eval()
    print("Model loaded successfully.")

except FileNotFoundError as e:
    # IMPORTANT: The model file (best_swin_nocurr_seed42.pth) must be in the same directory as app.py
    print(f"ERROR: {e}")
    print("Please download your model weights and place 'best_swin_nocurr_seed42.pth' next to this script.")
    swin_model = None # Keep the model None if loading failed

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# It is highly recommended to set this as an environment variable:
# e.g., export GEMINI_API_KEY="YOUR_API_KEY_HERE"
# The app will not call Gemini successfully unless this is set in the runtime environment.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("AWS_REGION", "us-east-1")
)

MODEL_ID = "meta.llama3-70b-instruct-v1:0"


app = Flask(__name__)
CORS(app) # Enable CORS for the front-end
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.route('/llm', methods=['GET', 'POST'])
def llm():
    # 1. If the user submits the form (POST)
    if request.method == 'POST':
        # Grab the inputs from the HTML form
        user_query = request.form.get('query')
        form_username = request.form.get('username')
        form_password = request.form.get('password')

        try:
            r = requests.post(
                "http://195.251.252.25:5000/ask",
                auth=(form_username, form_password),
                json={"query": user_query},
                timeout=120
            )
            r.raise_for_status()
            data = r.json()
            llm_output = data.get('answer', 'The "answer" key was not found in the response.')

        except requests.exceptions.RequestException as e:
            llm_output = f"An error occurred: {e}"

        # Pass the output back to the template
        return render_template('llm.html', response_text=llm_output)

    # 2. If the user just loads the page (GET), show the form with no output yet
    return render_template('llm.html', response_text=None)

@app.route('/alzheimer')
def alzheimer():
    return render_template('alzheimer.html')

@app.route('/waytoschool')
def waytoschool():
    """Renders the main quiz page."""
    return render_template('waytoschool.html', api_key=GEMINI_API_KEY)

@app.route('/walkfree')
def walkfree():
    """Renders the pavement annotation page."""
    return render_template('walkfree.html', api_key=GEMINI_API_KEY)

@app.route('/waytoschool/en')
def walkfree_en():
    """Renders the main quiz page for English."""
    return render_template('waytoschool_en.html', api_key=GEMINI_API_KEY)


@app.route('/')
def index():
    """Renders the main project index page."""
    return render_template('index.html')

@app.route('/walkfree/educator')
def walkfree_admin():
    """The admin educator's of walkfree"""
    return render_template('walkfree_edu.html', api_key=GEMINI_API_KEY)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles image upload and runs the Swin Transformer prediction.
    """
    if swin_model is None:
        return jsonify({"error": "Model failed to load. Check model path and file.", "healthy_prob": 0.5, "patient_prob": 0.5, "prediction": "Error"}), 503

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    image_file = request.files['image']

    try:
        # Read the image data from the file stream
        image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
        # Preprocess the image
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)
        # Model inference
        with torch.no_grad():
            logits = swin_model(img_tensor)
            probs = softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)

        # Prepare the response data
        response = {
            "filename": image_file.filename,
            "prediction": LABELS[pred_idx],
            "healthy_prob": round(float(probs[0]), 4),
            "patient_prob": round(float(probs[1]), 4)
        }
        print(f"Prediction result for {image_file.filename}: {response['prediction']}")
        return jsonify(response)

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return jsonify({"error": f"Prediction failed due to an internal server error: {str(e)}"}), 500

@app.route("/agentstalk")
def agentstalk():
    """The llm gossip app"""
    return render_template('llm_dis.html')

@app.route("/api/bedrock", methods=["POST"])
def call_bedrock():
    data = request.get_json()
    prompt = data.get("prompt")

    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400

    try:
        response = bedrock.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "prompt": prompt,
                "max_gen_len": 300,
                "temperature": 0.7,
                "top_p": 0.9
            })
        )

        body = json.loads(response["body"].read())
        return jsonify({"text": body["generation"]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
