from flask import jsonify, render_template, request
import requests
from app import app
from utils import get_models_dataframe
from config import INSTANCIA_TEMPLATE

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
    # TODO: receber do banco de dados os modelos disponiveis
    modelos = modelos_chagas
    # TODO: receber do banco de dados o exemplo de instancia para o modelo especifico
    return render_template(
        "/classify/chagas.html", modelos=modelos, instancia=INSTANCIA_TEMPLATE
    )


@app.route("/classify/chagas", methods=["POST"])
def post_classify_chagas():
    selected_model = request.form["model"]
    instancia = {}
    for key, value in request.form.items():
        if key != "model":
            instancia[key] = float(value)  # converter todos os valores para float

    
    response = requests.post(
        f"{base_url_api}/chagas",
        json={"model": selected_model, "instancia": instancia},
    )
    if response.status_code == 200:
        return render_template(
            "/result/chagas.html",
            model=selected_model,
            instancia=instancia,
            response_data=response.json(),
        )
    else:
        return render_template(
            "error.html",
            response_json=response.json(),
            status_code=response.status_code,
        )
