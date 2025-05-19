import cv2
import numpy as np

def preprocess_image(image):
    # Step 1: Denoising
    denoised = cv2.GaussianBlur(image, (5, 5), 0)

    # Step 2: Contrast enhancement
    equalized = cv2.equalizeHist(denoised)

    return equalized

def auto_segment(image):
    # Auto segmentation using thresholding (Otsu)
    _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

def manual_segment(image):
    roi = []
    drawing = False

    def draw_roi(event, x, y, flags, param):
        nonlocal drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            roi.append((x, y))
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            roi.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    temp_img = image.copy()
    cv2.namedWindow("Manual ROI Selector")
    cv2.setMouseCallback("Manual ROI Selector", draw_roi)

    while True:
        temp = cv2.cvtColor(temp_img.copy(), cv2.COLOR_GRAY2BGR)
        if len(roi) > 1:
            for i in range(len(roi)-1):
                cv2.line(temp, roi[i], roi[i+1], (0, 255, 0), 2)
            cv2.line(temp, roi[-1], roi[0], (0, 255, 0), 1)

        for pt in roi:
            cv2.circle(temp, pt, 3, (0, 0, 255), -1)

        cv2.imshow("Manual ROI Selector", temp)
        key = cv2.waitKey(1)

        if key == 13:  # Enter key
            break
        elif key == 27:  # Escape key
            roi.clear()
            break

    mask = np.zeros_like(image, dtype=np.uint8)
    if len(roi) >= 3:
        pts = np.array(roi, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)

    cv2.destroyAllWindows()
    return mask