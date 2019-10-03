from pathlib import Path

import tensorflow as tf

from src.tf_img_aug import img_aug


def adjust_gamma_fixed_gamma(image, gain):
    image =tf.image.adjust_gamma(
        image,
        gamma=1,
        gain=gain
    )
    return image


def adjust_gamma_fixed_gain(image, gamma):
    return tf.image.adjust_gamma(
        image,
        gamma=gamma,
        gain=1
    )


def main():
    input_dir:Path = Path()/"input"
    input_path = str(input_dir / "degu.jpg")

    # adjust_gamma
    params = [round(i * 0.4, 1) for i in range(10)]
    img_aug(input_path, "gamma_gamma.gif", adjust_gamma_fixed_gain, params, "gamma", after_str="gain: 1")
    img_aug(input_path, "gamma_gain.gif", adjust_gamma_fixed_gamma, params, "gain", before_str="gamma: 1")


if __name__ == "__main__":
    main()
