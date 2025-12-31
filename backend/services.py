import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class DigitRecognitionService:

    def __init__(self):
        # Load model once when service is initialized
        self.model = load_model("model/3BModel.h5")

    def predict_digit(self, img_path):
        """
        Takes image path as input
        Returns predicted digit (int)
        """

        img = image.load_img(
            img_path,
            color_mode='grayscale',
            target_size=(28, 28)
        )

        img_array = image.img_to_array(img)
        img_array = 255 - img_array   # invert colors
        img_array = img_array.reshape(1, 28, 28, 1).astype("float32") / 255.0

        prediction = self.model.predict(img_array)
        predicted_digit = int(np.argmax(prediction))

        return predicted_digit
