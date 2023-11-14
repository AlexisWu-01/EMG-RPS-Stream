import numpy as np
from runPythonModel import RunPythonModel

model = RunPythonModel('models/rf_best.joblib')
print(model)
data = np.random.rand(4,1400)
print(model.get_rps(data))