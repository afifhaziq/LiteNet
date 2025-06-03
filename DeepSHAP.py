import numpy as np
import torch
from model import NtCNN

sequence = 37
features = 20
learning_rate = 0.001
batch_size = 64
num_class = 10
num_epoch = 30

device = "cuda" if torch.cuda.is_available() else "cpu"

model = NtCNN(sequence=sequence, features=features, num_class=num_class).to(device)
model.load_state_dict(torch.load("saved_dict//NtCNN_MALAYAGT_740Features_best_model.pth"))
model.eval()

train = np.load("MALAYAGT//train.npy", allow_pickle=True)

train = torch.from_numpy(train.astype(np.float32))

train[:, [12,13,14,15,16,17,18,19]] = 0

x_train = train[:, :-1]
y_train = train[:, -1]

import shap

shap.initjs()

# Select the first n samples as background
background = x_train[:10000].to (device)

# Initialize GradientExplainer
explainer = shap.GradientExplainer(model, background)
print('start calculating shap')

# Compute SHAP values
shap_value = explainer.shap_values(background)
print('done')
# For multi-class, select SHAP values for a specific class, e.g., class 0
#shap_value = shap_value[3]

#global SHAP values
shap_value_global = shap_value.sum(axis=2)  # Sum over class dimension
 
shap.summary_plot(shap_value_global, background.cpu().numpy(), max_display=20)