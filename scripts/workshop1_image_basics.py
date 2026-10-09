"""
Workshop#1 : Image Loading & Display  (วันที่ 1 ช่วง 10.40-12.00)
เรียนรู้: ภาพคือ array ของตัวเลข / ขนาด / channel / Gray vs Color / RGB vs HSV

รัน:
  python workshop1_image_basics.py                       # ภาพจากไฟล์ shapes.png
  python workshop1_image_basics.py --source emu          # ภาพจากกล้อง Basler จำลอง
ปุ่ม:  1 = ภาพสี   2 = ภาพเทา   3 = แยก B,G,R   4 = แยก H,S,V   s = บันทึก   q = ออก
คลิกเมาส์บนภาพ = ดูค่าสีของพิกเซลตรงนั้น
"""
import argparse

import cv2
import numpy as np

from vision_camera import add_source_args, open_source, fit_to_screen, put_label


def print_properties(img):
    h, w = img.shape[:2]
    ch = 1 if img.ndim == 2 else img.shape[2]
    print("===== คุณสมบัติของภาพ =====")
    print(f"ขนาด (Width x Height) : {w} x {h} พิกเซล")
    print(f"จำนวน channel         : {ch}")
    print(f"ชนิดข้อมูล            : {img.dtype}  (ค่า 0-255 ต่อ channel)")
    print(f"Bit depth             : {8 * ch} bit")
    print(f"จำนวนพิกเซลทั้งหมด    : {w * h:,}")
    print(f"ขนาดในหน่วยความจำ     : {img.nbytes / 1024:.1f} KB")


def split_view(img, mode):
    """แสดง 3 channel เรียงกัน"""
    if mode == "BGR":
        chans, names = cv2.split(img), ["B (Blue)", "G (Green)", "R (Red)"]
    else:
        chans = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        names = ["H (Hue 0-179)", "S (Saturation)", "V (Value)"]
    tiles = []
    for c, n in zip(chans, names):
        t = cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)
        put_label(t, n)
        tiles.append(t)
    return np.hstack(tiles)


def main():
    ap = argparse.ArgumentParser()
    add_source_args(ap, default_image="shapes.png")
    args = ap.parse_args()

    src = open_source(args)
    first = src.read()
    print_properties(first)

    mode = "color"
    clicked = {"pt": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["pt"] = (x, y)

    win = "Workshop1"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    try:
        while True:
            img = src.read()
            if img is None:
                continue

            if mode == "color":
                view = img.copy()
            elif mode == "gray":
                view = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                    cv2.COLOR_GRAY2BGR)
            else:
                view = split_view(img, mode)

            # คลิกดูค่าพิกเซล (ใช้ได้ในโหมดภาพสี/เทา)
            if clicked["pt"] and mode in ("color", "gray"):
                x, y = clicked["pt"]
                if x < img.shape[1] and y < img.shape[0]:
                    b, g, r = img[y, x]
                    h, s, v = cv2.cvtColor(img[y:y+1, x:x+1], cv2.COLOR_BGR2HSV)[0, 0]
                    cv2.circle(view, (x, y), 6, (0, 0, 255), 2)
                    put_label(view, f"({x},{y}) BGR=({b},{g},{r})  HSV=({h},{s},{v})",
                              org=(10, view.shape[0] - 15), scale=0.6)

            put_label(view, f"mode: {mode}   [1]color [2]gray [3]BGR [4]HSV [q]quit")
            cv2.imshow(win, fit_to_screen(view))

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("1"):
                mode = "color"
            elif key == ord("2"):
                mode = "gray"
            elif key == ord("3"):
                mode = "BGR"
            elif key == ord("4"):
                mode = "HSV"
            elif key == ord("s"):
                cv2.imwrite(f"workshop1_{mode}.png", view)
                print(f"บันทึก workshop1_{mode}.png")
    finally:
        src.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
