from flask import jsonify, request
import numpy as np
from app import app
from config import INSTANCIA_TEMPLATE
from utils import (
    get_model_with_extension,
    get_models_dataframe,
)

modelos_chagas = get_models_dataframe("chagas", extension=".pkl")
base_url_api = "/api"


@app.route(f"{base_url_api}/models/chagas", methods=["GET"])
def get_models_chagas():
    return jsonify(modelos_chagas)


def classificar_instancia(model_pkl, instance):
    values_list = np.array(list(instance.values()))
    values_list = values_list[:-1]
    values_list = values_list.reshape(1, -1)
    predict = model_pkl.predict(values_list)
    predict_proba = model_pkl.predict_proba(values_list)
    return predict, predict_proba


@app.route(f"{base_url_api}/chagas", methods=["POST"])
def post_classificar_chagas():
    data = request.get_json()
    if data and "model" in data and "instancia" in data:
        model: str = data["model"]
        instancia_template = INSTANCIA_TEMPLATE
        atributos_em_ordem = list(instancia_template.keys())
        instance_dict: list = data["instancia"]
        instance = {attr: instance_dict[attr] for attr in atributos_em_ordem}
        model_pkl = get_model_with_extension(
            dataframe_path="chagas", file=model, extension=".pkl"
        )
        if not model_pkl:
            return jsonify({"Erro": f"Modelo {model} não encontrado."}), 404

        predict, predict_proba = classificar_instancia(model_pkl, instance)
        
        return jsonify(
            {"predict": predict.tolist(), "predict_proba": predict_proba.tolist()}
        )
    else:
        return jsonify({"error": "Parâmetros inválidos"}), 400
