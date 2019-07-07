import cv2
import numpy as np

from Define import *

def Save(fake_images, save_path):
    save_image = np.zeros((IMAGE_HEIGHT * SAVE_HEIGHT, IMAGE_WIDTH * SAVE_WIDTH, OUTPUT_IMAGE_CHANNEL), dtype = np.uint8)

    # 0 ~ 255
    # -1 ~ 1 -> 0 ~ 2 * 127,5 -> 0 ~ 255
    for y in range(SAVE_HEIGHT):
        for x in range(SAVE_WIDTH):
            fake_image = (fake_images[y * SAVE_WIDTH + x] + 1) * 127.5
            save_image[y * IMAGE_HEIGHT : (y + 1) * IMAGE_HEIGHT, x * IMAGE_WIDTH : (x + 1) * IMAGE_WIDTH] = fake_image.astype(np.uint8)

    cv2.imwrite(save_path, save_image)
