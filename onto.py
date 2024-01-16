import cv2
import numpy as np
from onto_model import IndividualGenerator, SPARQL


def image_statistics(image_mask) -> zip:
    flat_image: np.array = image_mask.flatten()
    unique_values, counts = np.unique(flat_image, return_counts=True)
    counts = [round(el, 2) for el in 100*counts/flat_image.shape[0]]
    return zip(unique_values, counts)

def create_terrene_individuals(image, image_name: str, x: int, y: int) -> None:
    image_statistics_zip: zip = image_statistics(image)
    for key, value in image_statistics_zip:
        IndividualGenerator.create_terrene_individual(int(x), int(y), value, key, image_name)


def create_image_individual(image_uri, long, lat):
    IndividualGenerator.create_image_individual(image_uri, long, lat)