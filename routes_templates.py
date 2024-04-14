from flask import jsonify, render_template, request
import requests
from app import app
from utils import get_models_dataframe

base_url: str = "http://localhost:5000"
base_url_api: str = f"{base_url}/api"

modelos_chagas = get_models_dataframe("chagas", ".pkl")


@app.route("/")
def index():
    return render_template("home.html", modelos=modelos_chagas)


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/classify/chagas", methods=["GET"])
def get_classify_chagas():
    modelos = modelos_chagas
    # TODO: receber do banco de dados o exemplo de instancia para o modelo especifico
    instancia = {
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
    return render_template(
        "/classify/chagas.html", modelos=modelos, instancia=instancia
    )


# @app.route("/classify/chagas", methods=["POST"])
# def post_classify_chagas():
#     selected_model = request.form["model"]
#     # Criar um dicionário para armazenar a instância editada
#     instancia = {}
#     # Iterar sobre os itens enviados no formulário e armazenar na instância
#     for key, value in request.form.items():
#         # Ignorar o modelo, pois já foi obtido acima
#         if key != "model":
#             instancia[key] = value

#     # Enviar os dados para a outra API
#     try:
#         response = requests.post(
#             url=f"{base_url_api}/chagas",
#             json={"model": selected_model, "instancia": instancia},
#         )
#         sts = response.status_code
#         json_data = response.json()
#         return jsonify(json_data), sts
#     except requests.exceptions.RequestException as e:
#         return (
#             jsonify({"error": f"Erro ao enviar solicitação para a outra API: {e}"}),
#             500,
#         )


@app.route("/classify/chagas", methods=["POST"])
def post_classify_chagas():

    selected_model = request.form["model"]
    # Criar um dicionário para armazenar a instância editada
    instancia = {}
    # Iterar sobre os itens enviados no formulário e armazenar na instância
    for key, value in request.form.items():
        # Ignorar o modelo, pois já foi obtido acima
        if key != "model":
            instancia[key] = float(value) #converter todos os valores para float
 
    with app.test_client() as c: 
        # instance_values = instancia.values()
        # return jsonify(list(instance_values))
        response = c.post(
            f"{base_url_api}/chagas", json={"model": selected_model, "instancia": instancia}
        )

        # Verificando se a solicitação foi bem-sucedida e retornando a resposta
        if response.status_code == 200:
            response_data = response.json
            return jsonify(response_data), 200
        else:
            return jsonify({"error": "Erro ao chamar rota2"}), 500
    
    # return 500
