import json
import os
from pydantic import ValidationError
import predict
import classes
from flask import Flask, request, jsonify, url_for, send_from_directory
import pathlib
import traceback


# Initialize Flask app
app = Flask(__name__)


@app.route("/favicon.ico")
def favicon():
    return send_from_directory("static", "favicon.ico")


@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.debug(f"Exception encountered: {e}")
    if app.debug:
        tb = traceback.format_exc()
        app.logger.debug(tb)
        return jsonify({"error": str(e), "traceback": tb}), 500
    else:
        return jsonify({"error": str(e)}), 500


@app.route('/', methods=['GET'])
def index():
    """
    This endpoint returns a list of available endpoints.
    Returns: JSON result
    """
    output = []
    for rule in app.url_map.iter_rules():
        if rule.endpoint != 'static':
            methods = ','.join(sorted(rule.methods - {'HEAD', 'OPTIONS'}))
            output.append({
                "endpoint": rule.endpoint,
                "methods": methods,
                "url": str(rule),
                "link": url_for(rule.endpoint, _external=True)
            })
    return jsonify(output)


@app.route('/models', methods=['GET'])
def predict_get():
    """
    This endpoint returns a list of available models and their features.
    Returns: JSON result
    """
    index_file = pathlib.Path("models" , "model_index.json")
    with open(index_file, 'r') as f:
        response = json.load(f)
    return jsonify(response)


@app.route('/predict', methods=['POST'])
def predict_post():
    """
    This endpoint expects a JSON object with a formatted request based on the enhanced model features for the given model
    Returns: JSON result with prediction and metadata

    """
    try:
        # Validate and parse input
        input_model = classes.EnhancedPredictionRequest(**request.get_json())

        # Get the prediction
        response = predict.get_prediction(input_model)

        # Send the response
        return response.model_dump_json(indent=4)

    except ValidationError as ve:
        return jsonify({"validation_error": ve.errors()}), 422


@app.route('/predict-basic', methods=['POST'])
def predict_basic_post():
    """
    This endpoint expects a JSON object with a formatted request based on the basic model features for the given model
    Returns: JSON result with prediction and metadata

    """
    try:
        # Validate and parse input
        input_model = classes.BasicPredictionRequest(**request.get_json())

        # Get the prediction
        response = predict.get_prediction(input_model)

        # Send the response
        return response.model_dump_json(indent=4)

    except ValidationError as ve:
        return jsonify({"validation_error": ve.errors()}), 422


if __name__ == '__main__':
    host = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    port = os.getenv("FLASK_RUN_PORT", 5000)
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.logger.info(
        f"Starting the Flask app on {host}:{port} (DEBUG={debug})"
    )
    app.run(host = host, port = port, debug = debug)
