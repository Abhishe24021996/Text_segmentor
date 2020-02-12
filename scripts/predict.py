import json
from .BILSTM_CRF_model import BILSTM_CRF
from .config import config
import re

def load_model(config_path):
	print("Loading model.....")
    config_params = json.load(open(config_path))
    config_p = Config(**config_params, load=True)
    model = BILSTM_CRF(config_p)
    model.build()
    model.restore_session(config.model_path)
    print("Model Loaded")
    return model

def predict(model, in_sequence = []):
    pred_sequence = model.predict(in_sequence)
    return pred_sequence

if __name__ == '__main__':
	print("Enter config_params path:")
	config_path = input()
	model = load_model(config_path)
	while True:
		try:
			in_seq = input()
			in_seq = re.split("\s+",in_seq)
			pred_sequence = predict(model,in_sequence=in_seq)
			slots = "{:8}"*len(in_seq)
			print(slots.format(w for w in in_seq))
			print(slots.format(w for w in pred_sequence))
		except KeyboardInterrupt:
			break
    
