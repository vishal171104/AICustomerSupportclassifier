import pickle
import numpy as np

with open("model/priority_pipeline.pkl", "rb") as f:
    pipe = pickle.load(f)

text = "The payment gateway is returning 500 errors for all users. Production is down and customers cannot checkout."
print("Classes:", pipe.classes_)
print("Proba:", pipe.predict_proba([text]))
print("Pred:", pipe.predict([text]))

with open("model/category_pipeline.pkl", "rb") as f:
    pipe_c = pickle.load(f)

print("Cat Classes:", pipe_c.classes_)
print("Cat Proba:", pipe_c.predict_proba([text]))
print("Cat Pred:", pipe_c.predict([text]))

