import pickle
import json 
import gradio as gr
import numpy as np
import pandas as pd
import sklearn

# File Paths
model_path = 'knn_model.sav'
component_config_path = "component_configs.json"
examples_path = "examples.pkl"

# predefined
feature_order =  ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 
                 'Q10', 'Q11', 'Q12', 'Q13', 'Q14', 'Q15', 'Q16', 'Q17',
                 'Q18', 'Q19', 'Q20', 'Q21', 'Q22', 'Q23', 'Q24', 'Q25',
                 'Q26', 'Q27', 'Q28']

# Loading the files
model = pickle.load(open(model_path, 'rb'))
with open(examples_path,"rb") as f: examples = pickle.load(f) 
feature_limitations = json.load(open(component_config_path, "r"))


# Util function
def predict(*args):
    # preparing the input into convenient form
    features = pd.Series([*args], index=feature_order)
    features = np.array(features).reshape(-1,len(feature_order))
  
    # compute distances to each cluster center
    distances = model.transform(features)
    
    # convert distances to probabilities using softmax
    exp_distances = np.exp(-distances)
    class_probabilities = exp_distances / np.sum(exp_distances, axis=1, keepdims=True)
    
    # create dictionary of class probabilities
    class_probs_dict = {}
    for i in range(model.n_clusters):
        class_probs_dict[f'cluster_{i}'] = round(class_probabilities[0, i], 3)
    
    return class_probs_dict

# Creating the gui component according to component.json file
inputs = list()
for col in feature_order:
  if col in feature_limitations["cat"].keys():
    
    # extracting the params
    vals = feature_limitations["cat"][col]["values"]
    def_val = feature_limitations["cat"][col]["def"]
    
    # creating the component
    inputs.append(gr.inputs.Dropdown(vals, default=def_val, label=col))
  else:
    
    # extracting the params
    min = feature_limitations["num"][col]["min"]
    max = feature_limitations["num"][col]["max"]
    def_val = feature_limitations["num"][col]["def"]
    
    # creating the component
    inputs.append(gr.inputs.Slider(minimum=min, maximum=max, default=def_val, label=col) )




# creating the app
demo_app = gr.Interface(predict, inputs, "label",examples=examples)

# Launching the demo
if __name__ == "__main__":
    demo_app.launch()