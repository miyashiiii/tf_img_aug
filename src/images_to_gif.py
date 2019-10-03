import cv2
import numpy as np
from PIL import Image


def images_to_gif(imgs, output_path):
    pil_imgs = [Image.fromarray(img) for img in imgs]
    pil_imgs[0].save(output_path, save_all=True, append_images=pil_imgs[1:], duration=500, disposal=2)


def images_to_gif_with_param(imgs, output_path, param_name, params, before_str="", after_str=""):
    new_imgs = []
    if before_str:
        before_str += " "
    if after_str:
        after_str = " " + after_str
    for img, param in zip(imgs, params):
        print(img[0,0,0])
        h, w = img.shape[:2]
        new_h = h + 15
        new_img = np.full((new_h, w, 3), 255, dtype=np.uint8)
        new_img[:h, :] = img
        cv2.putText(new_img, f"{before_str}{param_name}: {param}{after_str}", (5, new_h - 5), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        new_imgs.append(new_img)
    images_to_gif(new_imgs, output_path)


def main():
    file_path_1 = "result_00.jpg"
    file_path_2 = "result_01.jpg"
    file_path_3 = "result_02.jpg"
    file_path_4 = "result_03.jpg"
    file_path_5 = "result_04.jpg"
    file_paths = [file_path_1, file_path_2, file_path_3, file_path_4, file_path_5]
    imgs = [cv2.imread(path) for path in file_paths]
    output_path = "result.gif"
    param_name = "num"
    params = [0, 1, 2, 3, 4]
    images_to_gif_with_param(imgs, output_path, param_name, params)


if __name__ == "__main__":
    main()
