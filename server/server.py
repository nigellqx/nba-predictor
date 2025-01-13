from flask import Flask, request, jsonify
from flask_cors import CORS

import util

app = Flask(__name__)
CORS(app)

@app.route('/nba_predict', methods=['POST'])
def nba_predict():
    data = request.get_json()
    nba_team = data['homeTeam']
    nba_opponent = data['awayTeam']
    playing = data['playing']
    response = jsonify(util.nba_predict(nba_team, nba_opponent, playing))
    return response

if __name__ == "__main__":
    util.load_saved_artifacts()
    app.run(port=5000)
