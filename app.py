# app.py
from flask import Flask, render_template
from jee import jee_bp  # Make sure this is the correct import
from mht_cet import mht_cet_bp  # Assuming similar setup for MHT CET

app = Flask(__name__)

# Register the blueprints
app.register_blueprint(jee_bp)
app.register_blueprint(mht_cet_bp)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
