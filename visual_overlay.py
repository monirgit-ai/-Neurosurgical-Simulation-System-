import cv2
import numpy as np

def draw_roi_boundary(image, mask):
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
    return output

def draw_crack_line(image, mask):
    output = draw_roi_boundary(image, mask)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return output

    x_min, x_max = np.min(xs), np.max(xs)
    y_center = int(np.mean(ys))

    for i in range(x_min, x_max, 10):
        y_offset = np.random.randint(-5, 5)
        pt1 = (i, y_center + y_offset)
        pt2 = (i + 10, y_center + np.random.randint(-5, 5))
        cv2.line(output, pt1, pt2, (0, 0, 255), 2)

    return output
