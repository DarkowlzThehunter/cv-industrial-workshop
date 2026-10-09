"""
Workshop#3 : OCR อ่านตัวอักษรจากภาพ  (วันที่ 2 ช่วง 10.40-12.00)
ตัวอย่างงานจริง: อ่านวันหมดอายุ, เลข LOT, รหัสสินค้าบนฉลาก

ต้องติดตั้งโปรแกรม Tesseract ก่อน (ดู README):
  Windows : ตัวติดตั้ง Tesseract-OCR (UB Mannheim)
  Mac     : brew install tesseract
และ        pip install pytesseract

รัน:
  python workshop3_ocr.py                          # อ่านจาก ocr_sample.png
  python workshop3_ocr.py --digits                 # อ่านเฉพาะตัวเลข
  python workshop3_ocr.py --source emu             # ภาพจากกล้อง Basler จำลอง
ปุ่ม:  r = อ่าน (Read) 1 ครั้ง    b = เปิด/ปิด การทำภาพขาวดำก่อนอ่าน    q = ออก
"""
import argparse
import os
import shutil
import time

import cv2
import pytesseract

from vision_camera import add_source_args, open_source, fit_to_screen, put_label

# Windows: ถ้าหา tesseract ไม่เจอ ให้ชี้ไปที่ตำแหน่งติดตั้งมาตรฐาน
WIN_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if shutil.which("tesseract") is None and os.path.exists(WIN_PATH):
    pytesseract.pytesseract.tesseract_cmd = WIN_PATH


def preprocess(img, binarize=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if not binarize:
        return gray
    # Otsu: หาค่า threshold อัตโนมัติ -> ตัวอักษรดำ พื้นขาว
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def read_text(img, binarize=True, digits_only=False, min_conf=40):
    t0 = time.perf_counter()
    prep = preprocess(img, binarize)

    config = "--psm 6"      # psm 6 = มองเป็นบล็อกข้อความหลายบรรทัด
    if digits_only:
        config += " -c tessedit_char_whitelist=0123456789-/."
        min_conf = 0        # โหมด whitelist ของ Tesseract มักรายงานความมั่นใจเป็น 0

    data = pytesseract.image_to_data(prep, config=config,
                                     output_type=pytesseract.Output.DICT)

    out = img.copy()
    lines = {}
    for i, word in enumerate(data["text"]):
        word = word.strip()
        conf = float(data["conf"][i])
        if not word or conf < min_conf:
            continue
        x, y, w, h = (data[k][i] for k in ("left", "top", "width", "height"))
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 0), 2)
        key = (data["block_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)

    text_lines = [" ".join(ws) for _, ws in sorted(lines.items())]
    ms = (time.perf_counter() - t0) * 1000
    return out, prep, text_lines, ms


def main():
    ap = argparse.ArgumentParser()
    add_source_args(ap, default_image="ocr_sample.png")
    ap.add_argument("--digits", action="store_true", help="อ่านเฉพาะตัวเลข")
    args = ap.parse_args()

    src = open_source(args)
    binarize = True
    out, lines, ms = None, [], 0.0

    win = "Workshop3 - OCR"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = src.read()
            if frame is None:
                continue
            # อ่านอัตโนมัติครั้งแรก จากนั้นอ่านเมื่อกด r (OCR ช้ากว่างานนับวัตถุมาก)
            if out is None:
                out, _, lines, ms = read_text(frame, binarize, args.digits)
                print("ผลการอ่าน:", lines, f"({ms:.0f} ms)")

            view = out.copy()
            put_label(view, f"{ms:.0f} ms  binarize={'on' if binarize else 'off'}  [r]read [b]toggle [q]quit",
                      scale=0.6)
            for i, t in enumerate(lines):
                put_label(view, t, org=(10, 60 + 30 * i), scale=0.7, color=(0, 255, 0))
            cv2.imshow(win, fit_to_screen(view))

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                out, _, lines, ms = read_text(frame, binarize, args.digits)
                print("ผลการอ่าน:", lines, f"({ms:.0f} ms)")
            if key == ord("b"):
                binarize = not binarize
                out = None
    finally:
        src.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
