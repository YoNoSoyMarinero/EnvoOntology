import numpy as np
import make_prediction
import onto
from onto import SPARQL, IndividualGenerator
from img_spliter import image_spliter
import os                       

def main(path, long, lat):

    image_spliter(path)
    test_image_paths = make_prediction.get_dataset_slice_paths('croped_images')
    model = make_prediction.segmentation_model()
    model.load_weights('model_third_parameteres.h5')

    for img in test_image_paths:

        test_dataset = make_prediction.get_test_data(img)
        y_true_image = make_prediction.get_images_and_segments_test_arrays(test_dataset)
        print("____________________________ Make prediction ____________________________")
        prediction = model.predict(test_dataset)
        prediction = np.argmax(prediction, axis=3)
        onto.create_image_individual(path, long, lat)
        x_coordinate = img.split("-")[1]
        y_coordinate = img.split("-")[2]
        predicted_image = prediction[0]
        onto.create_terrene_individuals(predicted_image, img, x_coordinate, y_coordinate.split(".")[0])
        
    os.rmdir("croped_images")
