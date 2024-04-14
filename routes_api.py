from dataclasses import dataclass
from flask import jsonify, request
import numpy as np
from app import app
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
    # todo: fazer uma funcao para conferir se a instancia está na ordem correta de atributos/features
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
        # Defina a ordem dos atributos da classe Instancia
        
        instancia_template = {
        "Sexo": 1.0,
        "BMI": 28.0,
        "Cancer": 0.0,
        "HAS": 0.0,
        "DM2": 1.0,
        "Cardiopatia Outra": 0.0,
        "Marcapasso": 0.0,
        "Sincope": 0.0,
        "Fibrilação/Flutter Atrial": 0.0,
        "I R Crônica": 0.0,
        "DLP": 1.0,
        "Coronariopatia": 0.0,
        "Embolia Pulmonar": 0.0,
        "Ins Cardiaca": 0.0,
        "AVC": 0.0,
        "DVP": 0.0,
        "TSH": 2.0,
        "Tabagismo": 0.0,
        "Alcoolismo": 0.0,
        "Sedentarismo": 0.0,
        "FC": 72.0,
        "Alt Prim": 0.0,
        "Pausa > 3s": 0.0,
        "ESV": 0.0,
        "EV": 0.0,
        "TVMNS": 0.0,
        "Area Elet inativa": 0.0,
        "Dist Cond AtrioVent": 0.0,
        "Disf Nodulo Sinusal": 0.0,
        "Fibri/Flutter Atrial": 0.0,
        "FC media": 68.0,
        "TVS": 0.0,
        "TVMNS.1": 0.0,
        "EV.1": 0.0,
        "EVTotal": 456.0,
        "AE diam": 3.1,
        "VED": 4.2,
        "VES": 3.0,
        "FE Teicholz": 0.77,
        "Deficit Seg": 0.0,
        "Rassi pontos": 2.0,
        "CDI": 0.0,
        "Ablações": 0.0,
        "Amiodarona": 0.0,
        "Idade Holter": 62.0,
        "Rassi escore_baixo": 1.0,
        "Rassi escore_intermediario": 0.0,
        "Diretriz 2005_A": 1.0,
        "Diretriz 2005_B1": 0.0,
        "Diretriz 2005_B2": 0.0,
        "Classificação_Normal": 0.0,
        "Classificação_Disf Leve": 0.0,
        "Classificação_Disf Moderada": 1.0,
        "Dist Cond AtrioVent _0": 1.0,
        "Dist Cond AtrioVent _3": 0.0,
        "Dist Cond AtrioVent _1": 0.0,
        "Dist Cond AtrioVent _2": 0.0,
        "Dist Cond InterVent _0": 1.0,
        "Dist Cond InterVent _3": 0.0,
        "Dist Cond InterVent _1": 0.0,
        "Dist Cond InterVent _2": 0.0,
        "Disf Diastolica_1": 0.0,
        "Disf Diastolica_0": 1.0,
        "Disf Diastolica_2": 0.0,
        "NYHA_1": 0.0,
        "NYHA_2": 1.0,
        "NYHA_0": 0.0,
        "NYHA_3": 0.0,
        "Obito_MS": 0.0,
    }
        atributos_em_ordem = list(instancia_template.keys())
        instance_dict:list = data["instancia"]
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
