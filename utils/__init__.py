from app import app
import os
import pickle


def get_models_dataframe(dataframe_path, extension=".pkl"):
    folder_path = os.path.join(app.root_path, "utils", "modelos", dataframe_path)
    model_files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(extension):
            model_files.append(file_name)
    return model_files


def get_model_with_extension(
    dataframe_path: str = "", file: str = "", extension: str = ".pkl"
):
    folder_path = os.path.join(app.root_path, "utils", "modelos", dataframe_path)

    if not file.endswith(extension):
        return None

    model_file = os.path.join(folder_path, file)
    if os.path.exists(model_file):
        with open(model_file, "rb") as f:
            model = pickle.load(f)
        return model

    return None
