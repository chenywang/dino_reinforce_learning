import pickle
from io import BytesIO
import cv2
import numpy as np
from PIL import Image
import base64

getbase64Script = "canvasRunner = document.getElementById('runner-canvas'); \
return canvasRunner.toDataURL().substring(22)"


def save_obj(obj, name):
    with open('objects/' + name + '.pkl', 'wb') as f:  # dump files into objects folder
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('objects/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def grab_screen(_driver):
    image_b64 = _driver.execute_script(getbase64Script)
    screen = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
    image = process_img(screen)  # processing image as required
    return image


def process_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # RGB to Grey Scale
    image = image[:300, :500]  # Crop Region of Interest(ROI)
    image = cv2.resize(image, (80, 80))
    return image


def show_img(graphs=False):
    """
    Show images in new window
    """
    while True:
        screen = (yield)
        window_title = "logs" if graphs else "game_play"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        imS = cv2.resize(screen, (800, 400))
        cv2.imshow(window_title, screen)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break
