from flask import Flask


app = Flask(__name__)


from routes_api import *
from routes_templates import *


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
