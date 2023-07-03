import pickle


from transformers import ElectraForTokenClassification
import torch


model=torch.load("./model.pkl",map_location=torch.device("cpu"))

model.module.save_pretrained("./save_models_1")








