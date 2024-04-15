from flask import jsonify, request
from app import app
from config import INSTANCIA_TEMPLATE
from services.predict_service import classificar_instancia

from utils import (
    get_model_with_extension,
    get_models_dataframe,
)

modelos_chagas = get_models_dataframe("chagas", extension=".pkl")
base_url_api = "/api"


@app.route(f"{base_url_api}/models/chagas", methods=["GET"])
def get_models_chagas():
    return jsonify(modelos_chagas)


@app.route(f"{base_url_api}/chagas", methods=["POST"])
def post_classificar_chagas():
    data = request.get_json()
    validation = validation_data(data)
    if not validation[0]:
        return validation[1], 400

    model: str = data["model"]
    model_pkl = get_model_with_extension(
        dataframe_path="chagas", file=model, extension=".pkl"
    )
    if not model_pkl:
        return jsonify({"Erro": f"Modelo {model} nao encontrado."}), 404

    atributos_em_ordem = list(INSTANCIA_TEMPLATE.keys())
    instance_dict = data["instancia"]
    instance = {attr: instance_dict[attr] for attr in atributos_em_ordem}
    predict, predict_proba = classificar_instancia(model_pkl, instance)
    return jsonify(
        {"predict": predict.tolist(), "predict_proba": predict_proba.tolist()}
    )


def validation_data(data):
    if "model" not in data:
        return False, jsonify({"error": "Modelo nao encontrado"})
    if "instancia" not in data:
        return False, jsonify({"error": "Instancia nao encontrada"})
    if type(data["instancia"]) != dict:
        return False, jsonify(
            {"error": "Tipo invalido para a instancia, deve ser um dicionario"}
        )
    return True, None
