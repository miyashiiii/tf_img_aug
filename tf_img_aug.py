from pathlib import Path

import cv2
import tensorflow as tf


def tf_img_aug(img_path, delta):
    image_r = tf.io.read_file(img_path)
    image = tf.image.decode_image(image_r, channels=3)
    image = tf.image.adjust_gamma(
        image,
        gain=delta
    )
    # image = tf.image.encode_png(image)

    return image


def main():
    input_dir = Path("input")
    output_dir = Path("output")

    img_name = "000001.jpg"
    input_path = str(input_dir / img_name)

    for i in range(10):
        output_path = str(output_dir / f"{i:02}_{img_name}")

        delta = i * 0.2
        img_aug = tf_img_aug(input_path, delta)

        with tf.compat.v1.Session() as sess:
            img_rgb = sess.run(img_aug)

        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, img)


if __name__ == "__main__":
    main()
