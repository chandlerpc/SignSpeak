import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras

m = keras.models.load_model('./checkpoints/custom_asl_best.h5')
print("Trained model layers:")
for i, l in enumerate(m.layers):
    weights_info = ""
    if len(l.get_weights()) > 0:
        weights_info = f" - weights: {[list(w.shape) for w in l.get_weights()]}"
    print(f"{i}: {l.name} ({l.__class__.__name__}){weights_info}")
