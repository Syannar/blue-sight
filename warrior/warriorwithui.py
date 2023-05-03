import cv2
import numpy as np

"""
You have to pick an ".png" image which has the size of 640x480 pixels.
Other pixel configurations such as 1920x1080 can cause errors.

https://github.com/Syannar

If you want to see the annoying grids you can delete line 89-97 :D
"""

def main():
    cap = cv2.VideoCapture(0)

    # PNG dosyasını yükle
    img = cv2.imread('kenan.png', cv2.IMREAD_UNCHANGED)
    if img is None:
        print("no such file or directory")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame, img)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_frame(frame, img):
    height, width, _ = frame.shape
    center = (width // 2, height // 2)

    # For PNG
    img_height, img_width, _ = img.shape
    scale_factor = max(height / img_height, width / img_width)
    new_height = int(img_height * scale_factor)
    new_width = int(img_width * scale_factor)
    img_resized = cv2.resize(img, (new_width, new_height))
    y_offset = (height - new_height) // 2
    x_offset = (width - new_width) // 2
    alpha_s = img_resized[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(0, 3):
        frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width, c] = (alpha_s * img_resized[:, :, c] + alpha_l * frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width, c])

    # HSV of the blue object
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find the contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # To find the object which has the biggest contour area
    max_area = 1000
    max_cnt = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            max_cnt = cnt

    if len(max_cnt) > 0:
        M = cv2.moments(max_cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # This part points the middle of the contour area
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.line(frame, center, (cx, cy), (0, 0, 255), 2)

        # This part draws two lines to measure distance from the middle point to x-axis and y-axis
        cv2.line(frame, center, (cx, center[1]), (255, 0, 0), 4)
        cv2.line(frame, center, (center[0], cy), (255, 0, 0), 4)

        #This part is to calculate rectangle location
        rect_size = min(height, width) // 6
        x_idx = (cx - 1) // rect_size
        y_idx = (cy - 1) // rect_size

        #This part draws lines to make screen grid by grid
        """
        for i in range(10):
            for j in range(10):
                x1 = i * rect_size
                y1 = j * rect_size
                x2 = (i + 1) * rect_size
                y2 = (j + 1) * rect_size
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
        """
        # Piksel lokasyonlarını ve kare numaralarını yazdır
        text = f"X: {cx}, Y: {cy}, Rectangle_Location: ({y_idx+1}, {x_idx+1})"
        cv2.putText(frame, text, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame

if __name__ == '__main__':
    main()
