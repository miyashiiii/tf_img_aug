from pathlib import Path

import tensorflow as tf

from images_to_gif import images_to_gif_with_param


def adjust_gamma_fixed_gamma(image, gain):
    return tf.image.adjust_gamma(
        image,
        gamma=1,
        gain=gain
    )


def adjust_gamma_fixed_gain(image, gamma):
    return tf.image.adjust_gamma(
        image,
        gamma=gamma,
        gain=1
    )


def img_aug_session(img_path, fn, param):
    image_r = tf.io.read_file(img_path)
    image = tf.image.decode_image(image_r, channels=3)

    image = fn(image, param)
    return image


def img_aug(input_path, output_path, fn, params, before_str="", after_str=""):
    imgs = []
    for param in params:
        with tf.compat.v1.Session() as sess:
            img = sess.run(img_aug_session(input_path, fn, param))

        imgs.append(img)

    images_to_gif_with_param(imgs, output_path, "gamma", params, before_str, after_str)


def main():
    input_dir = Path("input")
    output_dir = Path("output")
    input_path = str(input_dir / "degu.jpg")
    params = [round(i * 0.4, 1) for i in range(10)]
    img_aug(input_path, str(output_dir / "gamma_gamma.gif"), adjust_gamma_fixed_gain, params, after_str="gain: 1")
    img_aug(input_path, str(output_dir / "gamma_gain.gif"), adjust_gamma_fixed_gamma, params, before_str="gamma: 1")


if __name__ == "__main__":
    main()
