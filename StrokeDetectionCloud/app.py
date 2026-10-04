
from flask import Flask, render_template, request, jsonify
from resource import predict_stroke

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None

    if request.method == "POST":
        try:
            values = [
                float(request.form["age"]),
                float(request.form["hypertension"]),
                float(request.form["heart_disease"]),
                float(request.form["avg_glucose_level"]),
                float(request.form["bmi"])
            ]

            result = predict_stroke(values)

        except (ValueError, TypeError, KeyError) as e:
            error = str(e)
        except Exception as e:
            print("Prediction error:", repr(e))
            error = "Prediction failed. Check the terminal."

    return render_template(
        "index.html",
        result=result,
        error=error
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()

        if not data or "sequence" not in data:
            return jsonify({
                "error": "Missing sequence in JSON request."
            }), 400

        result = predict_stroke(data["sequence"])

        return jsonify(result), 200

    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        return jsonify({
            "error": "Prediction failed."
        }), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "application": "Stroke Detection LSTM"
    }), 200


if __name__ == "__main__":
    app.run(debug=False)