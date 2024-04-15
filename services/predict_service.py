import numpy as np


def classificar_instancia(model_pkl, instance):
    values_list = np.array(list(instance.values()))
    values_list = values_list[:-1]
    values_list = values_list.reshape(1, -1)
    predict = model_pkl.predict(values_list)
    predict_proba = model_pkl.predict_proba(values_list)
    return predict, predict_proba
