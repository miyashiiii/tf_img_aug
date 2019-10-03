from pathlib import Path

import cv2
import numpy as np

from src.adjust_gamma import adjust_gamma_fixed_gain
from src.tf_img_aug import img_aug


def main():
    input_dir = Path()
    input_path = str(input_dir / "gray.jpg")
    # input_path = str(input_dir / "input"/"degu.jpg")

    # adjust_gamma
    img_aug(
        input_path=input_path,
        output_name="d.gif",
        fn=adjust_gamma_fixed_gain,
        # fn=lambda image, gain: image,
        params=[round(i * 0.1, 1) for i in range(20)],
        param_name="gamma",
    )


def make_img():
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img += 240
    cv2.imwrite("gray.jpg", img)


if __name__ == "__main__":
    main()
    # make_img()
"""
1.0
1.7299005
2.9925559
5.176824
8.95539
15.491933
26.799507
46.36047
80.19901
138.73628
240.0
415.17618
718.21356
1242.4374
2149.2932
3718.064
6431.882
11126.517
19247.756
33296.707
"""