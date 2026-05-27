import os
import tempfile
from flask import Flask, request, jsonify, send_from_directory
from src.extractor import extract_features_from_svg
from src.forecaster import MarketForecaster
from src.predictor import CostPredictor

# Initialize Flask app
# Serves static files from 'static/' folder
app = Flask(__name__, static_folder="static", static_url_path="")

# Initialize models
try:
    forecaster = MarketForecaster()
    predictor = CostPredictor()
    models_ready = True
except Exception as e:
    print(f"Error loading models: {e}")
    models_ready = False


@app.route("/")
def index():
    """Serves the front-end dashboard homepage."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    """Exposes prediction pipeline as a JSON API endpoint."""
    if not models_ready:
        return jsonify({
            "success": False,
            "error": "Model files are not ready on the server. Please run offline training scripts."
        }), 500

    try:
        # 1. Parse parameters
        city = request.form.get("city", "Helsinki").strip()
        year = int(request.form.get("year", 2026))
        confidence = float(request.form.get("confidence", 90.0))
        use_demo_raw = request.form.get("use_demo")
        use_demo = use_demo_raw == "on" or str(use_demo_raw).lower() in ("true", "on")
        
        print(f"\n--- API REQUEST - City: {city}, Year: {year}, Confidence: {confidence} ---")
        print(f"use_demo raw value: {use_demo_raw!r} -> resolved to use_demo: {use_demo}")

        # 2. Get SVG path
        svg_path = None
        temp_file_path = None
        base_dir = os.path.dirname(os.path.abspath(__file__))

        if use_demo:
            # Locate sample SVG in the workspace root
            svg_path = os.path.join(base_dir, "sample_apartment.svg")
            if not os.path.exists(svg_path):
                return jsonify({
                    "success": False,
                    "error": "Sample apartment SVG not found on the server."
                }), 404
        else:
            if "file" not in request.files or request.files["file"].filename == "":
                return jsonify({
                    "success": False,
                    "error": "No floor plan SVG file uploaded."
                }), 400
                
            uploaded_file = request.files["file"]
            # Save to temporary file to run extraction
            fd, temp_file_path = tempfile.mkstemp(suffix=".svg")
            os.close(fd) # Close file descriptor immediately
            uploaded_file.save(temp_file_path)
            svg_path = temp_file_path
            print(f"Uploaded file saved to temporary path: {temp_file_path}")

        # 3. Step 0: Extract geometric features
        print(f"Extracting features from path: {svg_path}")
        features = extract_features_from_svg(svg_path)
        print(f"extract_features_from_svg returned: {features}")
        gfa = features["gfa"]
        rooms = features["rooms"]

        # Clean up temp file if one was created
        if temp_file_path is not None and not use_demo:
            try:
                os.remove(temp_file_path)
            except Exception:
                pass

        # 4. Step 1: Forecast market price
        base_price = forecaster.forecast_price(city, year)

        # 5. Step 2: Predict construction cost and Confidence Interval
        results = predictor.predict_cost(
            gfa=gfa,
            rooms=rooms,
            base_market_price=base_price,
            city=city,
            year=year,
            confidence_level=confidence
        )

        # 6. Return payload
        return jsonify({
            "success": True,
            "gfa": gfa,
            "rooms": rooms,
            "base_market_price": base_price,
            "predicted_cost": results["predicted_cost"],
            "ci_lower": results["ci_lower"],
            "ci_upper": results["ci_upper"],
            "tree_predictions": results["tree_predictions"],
            "mae": predictor.mae
        })

    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Internal pipeline error: {str(e)}"}), 500


def main():
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting PCM cost predictor server on http://localhost:{port}...")
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
