from flask import Flask, jsonify
import numpy as np
import config

app = Flask(__name__)

@app.route("/add_matrices", methods=["GET"])
def add_matrices():
    matrix1 = np.array(config.MATRIX_1)
    matrix2 = np.array(config.MATRIX_2)
    result = np.add(matrix1, matrix2)
    return jsonify(result.tolist())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)