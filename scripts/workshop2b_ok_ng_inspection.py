"""
Workshop#2B : ตรวจชิ้นงาน OK / NG  (วันที่ 2 ช่วง 09.00-10.30)
ต่อยอดจาก Workshop#2 -> เปลี่ยน "นับได้กี่ชิ้น" เป็น "ผ่านหรือไม่ผ่าน" แบบในไลน์ผลิต

โจทย์ตัวอย่าง: ในถาดต้องมีชิ้นสีแดงขนาดใหญ่ "ครบ N ชิ้น" ถ้าขาดหรือเกิน = NG

รัน:
  python workshop2b_ok_ng_inspection.py --expect 5
  python workshop2b_ok_ng_inspection.py --source emu --expect 5
ปุ่ม:
  t = Trigger ตรวจ 1 ครั้ง (เหมือนเซนเซอร์สั่งถ่ายเมื่อชิ้นงานมาถึง) -> บันทึกลง inspection_log.csv
  q = ออก
"""
import argparse
import csv
import os
from datetime import datetime

import cv2

from vision_camera import add_source_args, open_source, fit_to_screen, put_label
from workshop2_object_counting import process

LOG_FILE = "inspection_log.csv"


def judge(count, expect):
    return "OK" if count == expect else "NG"


def draw_banner(img, verdict, count, expect, ok_total, ng_total, ms):
    color = (0, 170, 0) if verdict == "OK" else (0, 0, 220)
    h, w = img.shape[:2]
    bar = max(60, h // 8)
    cv2.rectangle(img, (0, 0), (w, bar), color, -1)
    cv2.putText(img, verdict, (15, int(bar * 0.8)), cv2.FONT_HERSHEY_SIMPLEX,
                bar / 40, (255, 255, 255), 3)
    put_label(img, f"found {count} / expect {expect}   {ms:.1f} ms",
              org=(int(bar * 2.2), int(bar * 0.6)), scale=0.7, color=(255, 255, 255))
    put_label(img, f"Total  OK: {ok_total}   NG: {ng_total}   [t]=trigger [q]=quit",
              org=(10, h - 15), scale=0.6)


def write_log(row):
    new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "verdict", "count", "expect", "process_ms"])
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    add_source_args(ap, default_image="shapes.png")
    ap.add_argument("--expect", type=int, default=5, help="จำนวนชิ้นที่ต้องมี")
    ap.add_argument("--low", type=int, default=0)
    ap.add_argument("--high", type=int, default=10)
    ap.add_argument("--area", type=int, default=1000)
    args = ap.parse_args()

    src = open_source(args)
    ok_total = ng_total = 0

    win = "Workshop2B - OK/NG Inspection"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Expect", win, args.expect, 50, lambda v: None)
    cv2.createTrackbar("Min Area", win, args.area, 20000, lambda v: None)

    try:
        while True:
            frame = src.read()
            if frame is None:
                continue
            expect = cv2.getTrackbarPos("Expect", win)
            _, result, _, count, ms = process(
                frame, args.low, args.high, cv2.getTrackbarPos("Min Area", win))
            verdict = judge(count, expect)

            # วาดกรอบบนภาพจริง (ไม่ใช่ภาพ mask) ให้ operator ดูง่าย
            view = frame.copy()
            mask_boxes = result.any(axis=2)
            view[mask_boxes] = (0.5 * view[mask_boxes] + 0.5 * result[mask_boxes]).astype("uint8")
            draw_banner(view, verdict, count, expect, ok_total, ng_total, ms)
            cv2.imshow(win, fit_to_screen(view))

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key == ord("t"):
                if verdict == "OK":
                    ok_total += 1
                else:
                    ng_total += 1
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                write_log([now, verdict, count, expect, f"{ms:.1f}"])
                if verdict == "NG":          # เก็บภาพชิ้นงานเสียไว้ตรวจสอบย้อนหลัง
                    os.makedirs("ng_images", exist_ok=True)
                    cv2.imwrite(f"ng_images/NG_{now.replace(':', '-')}.png", view)
                print(f"{now}  {verdict}  ({count}/{expect})")
    finally:
        src.close()
        cv2.destroyAllWindows()
        print(f"สรุป: OK {ok_total}  NG {ng_total}  -> ดูบันทึกใน {LOG_FILE}")


if __name__ == "__main__":
    main()
