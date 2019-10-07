from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def images_to_gif(imgs, output_path):
    pil_imgs = [Image.fromarray(img) for img in imgs]
    pil_imgs[0].save(output_path, save_all=True, append_images=pil_imgs[1:], duration=500, disposal=2)


def add_text_to_image(img, text):
    h, w = img.shape[:2]
    new_h = h + 15
    new_img = np.full((new_h, w, 3), 255, dtype=np.uint8)
    new_img[:h, :] = img
    cv2.putText(new_img, text, (5, new_h - 5), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    return new_img


def main():
    input_dir = Path("input")
    file_name_1 = "degu_00.jpg"
    file_name_2 = "degu_01.jpg"
    file_name_3 = "degu_02.jpg"
    file_name_4 = "degu_03.jpg"
    file_name_5 = "degu_04.jpg"
    file_names = [file_name_1, file_name_2, file_name_3, file_name_4, file_name_5]
    imgs = [cv2.cvtColor(cv2.imread(str(input_dir/name)), cv2.COLOR_BGR2RGB) for name in file_names]

    output_path = "output/result.gif"
    param_name = "num"
    params = [0, 1, 2, 3, 4]
    new_imgs = []
    for img, param in zip(imgs, params):
        new_imgs.append(add_text_to_image(img, f"{param_name}: {param}"))
    images_to_gif(new_imgs, output_path)


if __name__ == "__main__":
    main()
