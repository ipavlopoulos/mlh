import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from config import Config
from services.bedrock_service import BedrockService
from services.gemini_service import GeminiService
from services.model_service import SwinPredictor


class PrefixMiddleware:
    def __init__(self, app, prefix):
        self.app = app
        self.prefix = prefix.rstrip("/")

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "")
        if self.prefix and (path == self.prefix or path.startswith(f"{self.prefix}/")):
            environ["SCRIPT_NAME"] = environ.get("SCRIPT_NAME", "") + self.prefix
            environ["PATH_INFO"] = path[len(self.prefix):] or "/"
        return self.app(environ, start_response)


app = Flask(__name__)
app.config.from_object(Config)
CORS(app)
app.wsgi_app = PrefixMiddleware(app.wsgi_app, Config.URL_PREFIX)

predictor = SwinPredictor(Config.MODEL_PATH, Config.LABELS, Config.IMG_SIZE)
gemini = GeminiService(Config.GEMINI_API_KEY, Config.GEMINI_MODEL)
bedrock = BedrockService(Config.AWS_REGION, Config.BEDROCK_MODEL_ID)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": predictor.loaded,
        "gemini_configured": gemini.configured,
    })


@app.route("/health/model")
def model_health():
    status = 200 if predictor.loaded else 503
    return jsonify({
        "loaded": predictor.loaded,
        "model_path": Config.MODEL_PATH,
    }), status


@app.route('/llm', methods=['GET', 'POST'])
def llm():
    if request.method == 'POST':
        user_query = request.form.get('query')
        form_username = request.form.get('username')
        form_password = request.form.get('password')

        try:
            r = requests.post(
                Config.REMOTE_LLM_URL,
                auth=(form_username, form_password),
                json={"query": user_query},
                timeout=Config.REMOTE_LLM_TIMEOUT,
            )
            r.raise_for_status()
            data = r.json()
            llm_output = data.get('answer', 'The "answer" key was not found in the response.')

        except requests.exceptions.RequestException as e:
            llm_output = f"An error occurred: {e}"

        return render_template('llm.html', response_text=llm_output)

    return render_template('llm.html', response_text=None)

@app.route('/alzheimer')
def alzheimer():
    return render_template('alzheimer.html')

@app.route('/waytoschool')
def waytoschool():
    """Renders the main quiz page."""
    return render_template(
        'waytoschool.html',
        firebase_config=Config.WAY_TO_SCHOOL_FIREBASE_CONFIG,
    )

@app.route('/walkfree')
def walkfree():
    """Renders the pavement annotation page."""
    return render_template(
        'walkfree.html',
        firebase_config=Config.WALKFREE_FIREBASE_CONFIG,
    )

@app.route('/waytoschool/en')
def walkfree_en():
    """Renders the main quiz page for English."""
    return render_template(
        'waytoschool_en.html',
        firebase_config=Config.WAY_TO_SCHOOL_FIREBASE_CONFIG,
    )


@app.route('/')
def index():
    """Renders the main project index page."""
    return render_template('index.html')

@app.route('/walkfree/educator')
def walkfree_admin():
    """The admin educator's of walkfree"""
    return render_template(
        'walkfree_edu.html',
        firebase_config=Config.WAY_TO_SCHOOL_FIREBASE_CONFIG,
    )

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles image upload and runs the Swin Transformer prediction.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    try:
        response = predictor.predict_file(request.files['image'])
        print(f"Prediction result for {response['filename']}: {response['prediction']}")
        return jsonify(response)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        return jsonify({"error": str(e), "healthy_prob": 0.5, "patient_prob": 0.5, "prediction": "Error"}), 503
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return jsonify({"error": "Prediction failed due to an internal server error."}), 500


@app.route("/agentstalk")
def agentstalk():
    """The llm gossip app"""
    return render_template('llm_dis.html')


@app.route("/api/gemini", methods=["POST"])
def call_gemini():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Missing JSON payload"}), 400

    try:
        return jsonify(gemini.generate_content(payload))
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/bedrock", methods=["POST"])
def call_bedrock():
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt")

    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400

    try:
        return jsonify({"text": bedrock.generate(prompt)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    import os

    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
