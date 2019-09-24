from pathlib import Path

import tensorflow as tf

from images_to_gif import images_to_gif_with_param


def img_aug_session(img_path, fn, param):
    image_r = tf.io.read_file(img_path)
    image = tf.image.decode_image(image_r, channels=3)

    image = fn(image, param)
    return image


def img_aug(input_path, output_name, fn, params, param_name, before_str="", after_str=""):
    output_dir = Path("output")
    imgs = []
    for param in params:
        with tf.compat.v1.Session() as sess:
            img = sess.run(img_aug_session(input_path, fn, param))

        imgs.append(img)

    images_to_gif_with_param(imgs, str(output_dir / output_name), param_name, params, before_str, after_str)


def main():
    input_dir = Path("input")
    input_path = str(input_dir / "degu.jpg")

    # adjust_gamma
    img_aug(
        input_path=input_path,
        output_name="gamma_gain.gif",
        fn=lambda image, gain: tf.image.adjust_gamma(image, 1, gain),
        params=[round(i * 0.4, 1) for i in range(10)],
        param_name="gain",
        before_str="gamma: 1")


if __name__ == "__main__":
    main()
