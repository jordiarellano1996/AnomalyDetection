import json
import os
from keras.models import model_from_json


def save(model, path):
    json_config = model.to_json()
    try:
        os.makedirs(path)
    except FileExistsError:
        print("Directory ", path, " already exists")

    with open(os.path.join(path, 'model_architecture.json'), 'w') as fp:
        json.dump(json_config, fp)


def load(path):
    with open(path) as json_data:
        json_config = json.load(json_data)
    return model_from_json(json_config)
