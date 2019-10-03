import argparse
from pathlib import Path

import tensorflow as tf

from src.tf_img_aug import img_aug


def main():
    parser = argparse.ArgumentParser(description='Do some process to image')
    parser.add_argument("--gamma_gamma", "-gg", help="adjust gamma with this gamma value")
    parser.add_argument("--gamma_gain", "-ga", help="adjust gamma with this gamma value")

    input_dir = Path("input")
    input_path = str(input_dir / "degu.jpg")

    # adjust_gamma
    img_aug(
        input_path=input_path,
        output_name="adjust_brightness.gif",
        fn=lambda image, delta: tf.image.adjust_brightness(image, delta),
        params=[round(i * 0.1, 1) for i in range(10)],
        param_name="delta",
        )


if __name__ == "__main__":
    main()
