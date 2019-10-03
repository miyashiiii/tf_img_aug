from pathlib import Path

import tensorflow as tf

from src.images_to_gif import images_to_gif_with_param


def img_aug_session(img_path, fn, param):
    image_r = tf.io.read_file(img_path)
    image = tf.image.decode_image(image_r, channels=3)
    image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8))
    # image.set_shape([None, None, 3])
    image = tf.div(image,255)
    image,_,flt_image ,_= fn(image, param)
    image = tf.multiply(image,255)
    # image = tf.image.convert_image_dtype(image, dtype=tf.uint8,saturate=True)
    return image,flt_image


def img_aug(input_path, output_name, fn, params, param_name, before_str="", after_str=""):
    output_dir = Path() / "output"
    imgs = []
    for param in params:
        with tf.compat.v1.Session() as sess:
            img,flt_image = sess.run(img_aug_session(input_path, fn, param))
            print("flt_image", flt_image[0,0,0])
        imgs.append(img)

    images_to_gif_with_param(imgs, str(output_dir / output_name), param_name, params, before_str, after_str)


def main():
    input_dir = Path() / "input"
    input_path = str(input_dir / "degu.jpg")

    # adjust_gamma
    img_aug(
        input_path=input_path,
        output_name="gamma_gain.gif",
        fn=lambda image, gain: tf.image.adjust_gamma(image, 1, gain),
        # fn=lambda image, gain: image,
        params=[round(i * 0.4, 1) for i in range(10)],
        param_name="gain",
        before_str="gamma: 1")


if __name__ == "__main__":
    main()
