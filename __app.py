import os
from flask import Flask, render_template

# It is highly recommended to set this as an environment variable:
# e.g., export GEMINI_API_KEY="YOUR_API_KEY_HERE"
# A placeholder is used if not found, but the app will not work without a real key.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "PLACEHOLDER_API_KEY_NEEDS_TO_BE_REPLACED")

app = Flask(__name__)

@app.route('/')
def index():
    """Renders the main quiz page."""
    return render_template('index.html', api_key=GEMINI_API_KEY)

@app.route('/walkfree')
def walkfree():
    """Renders the pavement annotation page."""
    return render_template('walkfree.html', api_key=GEMINI_API_KEY)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
