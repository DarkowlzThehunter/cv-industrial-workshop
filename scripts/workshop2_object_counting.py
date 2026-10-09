"""
Workshop#2 : Object Counting ด้วย HSV  (วันที่ 1 ช่วง 13.00-16.00)
ขั้นตอนเหมือนสไลด์:
  1.Load image -> 2.Convert to HSV -> 3.Set Lower/Upper -> 4.Create mask
  -> 5.Bitwise And -> 6.Find contour -> 7.Draw contour (+ กรองด้วยพื้นที่)

รัน:
  python workshop2_object_counting.py                    # ภาพจากไฟล์
  python workshop2_object_counting.py --source emu       # กล้อง Basler จำลอง
  python workshop2_object_counting.py --source basler    # กล้อง Basler จริง
ปุ่ม: q = ออก, s = บันทึกภาพ    แถบเลื่อน: H low / H high / S min / V min / Min Area
"""
import argparse
import time

import cv2
import numpy as np

from vision_camera import add_source_args, open_source, fit_to_screen, put_label


def process(image_bgr, h_low, h_high, min_area, s_min=50, v_min=50):
    t0 = time.perf_counter()

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)                 # 2
    lower = np.array([h_low, s_min, v_min])                          # 3
    upper = np.array([h_high, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)                            # 4
    result = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)        # 5

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)            # 6
    draw = image_bgr.copy()
    cv2.drawContours(draw, contours, -1, (255, 0, 0), 2)             # 7
    count_color = len(contours)

    # นับเฉพาะวัตถุที่ใหญ่กว่า Min Area (ตัดจุดรบกวน/ชิ้นเล็ก)
    count_area = 0
    for c in contours:
        if cv2.contourArea(c) > min_area:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(result, (x, y), (x + w, y + h), (200, 0, 100), 3)
            count_area += 1

    ms = (time.perf_counter() - t0) * 1000
    return draw, result, count_color, count_area, ms


def main():
    ap = argparse.ArgumentParser()
    add_source_args(ap, default_image="shapes.png")
    ap.add_argument("--low", type=int, default=0)
    ap.add_argument("--high", type=int, default=10)
    ap.add_argument("--area", type=int, default=1000)
    args = ap.parse_args()

    src = open_source(args)

    win = "Workshop2 - Object Counting"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("H low", win, args.low, 179, lambda v: None)
    cv2.createTrackbar("H high", win, args.high, 179, lambda v: None)
    cv2.createTrackbar("S min", win, 50, 255, lambda v: None)
    cv2.createTrackbar("V min", win, 50, 255, lambda v: None)
    cv2.createTrackbar("Min Area", win, args.area, 20000, lambda v: None)

    try:
        while True:
            frame = src.read()
            if frame is None:
                continue
            draw, result, n_color, n_area, ms = process(
                frame,
                cv2.getTrackbarPos("H low", win),
                cv2.getTrackbarPos("H high", win),
                cv2.getTrackbarPos("Min Area", win),
                cv2.getTrackbarPos("S min", win),
                cv2.getTrackbarPos("V min", win))

            view = np.hstack([draw, result])
            put_label(view, f"Color objects: {n_color}   Area objects: {n_area}   {ms:.1f} ms")
            cv2.imshow(win, fit_to_screen(view))

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                cv2.imwrite("workshop2_result.png", view)
                print("บันทึก workshop2_result.png")
    finally:
        src.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
