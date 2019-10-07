import dataclasses
from pathlib import Path
from typing import Union, List

import tensorflow as tf

from images_to_gif import add_text_to_image, images_to_gif


@dataclasses.dataclass
class Param:
    name: str
    values: List[Union[int, float]]
    default: Union[int, float]




def img_aug_session(img_path, fn, param_dict):
    image_r = tf.io.read_file(img_path)
    image = tf.image.decode_image(image_r, channels=3)

    image = fn(image, **param_dict)
    return image


def param_dict_to_str(param_dict):
    result = ""
    for k, v in param_dict.items():
        result += f"{k}: {v}, "
    return result[:-2]


def img_aug(input_path, fn_name, fn, params):
    output_dir = Path("output")

    param_dict_default = {param.name: param.default for param in params}
    for param in params:
        param_dict = param_dict_default.copy()
        imgs = []
        for v in param.values:
            param_dict[param.name] = v
            with tf.compat.v1.Session() as sess:
                img = sess.run(img_aug_session(input_path, fn, param_dict))
            img = add_text_to_image(img, param_dict_to_str(param_dict))
            imgs.append(img)
        images_to_gif(imgs, str(output_dir / f"{fn_name}_{param.name}.gif"))


def main():
    input_dir = Path("input")
    input_path = str(input_dir / "degu.jpg")

    values = [round(i * 0.4, 1) for i in range(20)]

    # adjust_gamma
    img_aug(
        input_path=input_path,
        fn_name="adjust_gamma",
        fn=tf.image.adjust_gamma,
        params=[
            Param("gamma", values, 1),
            Param("gain", values, 1)
        ]
    )

    # adjust_contrast
    img_aug(
        input_path=input_path,
        fn_name="adjust_contrast",
        fn=tf.image.adjust_contrast,
        params=[
            Param("contrast_factor", values, 1),
        ]
    )



if __name__ == "__main__":
    main()
