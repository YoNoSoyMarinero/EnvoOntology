from PIL import Image
import os

Image.MAX_IMAGE_PIXELS = None


def image_spliter(image_path):
    large_img = Image.open(image_path)
    large_img = large_img.convert('RGB')
    croped_images = []
    width, height = 512, 512
    img_name = image_path
    os.makedirs("croped_images", exist_ok=True)

    for i in range(0, large_img.width, width):
        for j in range(0, large_img.height, height):
            box = (i, j, i+width, j+height)
            region = large_img.crop(box)
            filename = f'{img_name[:-4]}-{i}-{j}.png'
            output_path = os.path.join("croped_images", filename)
            region.save(output_path)