import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model("../models/malaria_model.h5")

img_path = "sample_cell.webp"  # change this to any test image

img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img)

img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("Parasitized (Malaria Detected)")
else:
    print("Uninfected (Healthy Cell)")