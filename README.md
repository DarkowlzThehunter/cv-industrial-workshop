# Computer Vision สำหรับงานอุตสาหกรรม

ชุด Workshop อบรม 2 วัน (10–11 ตุลาคม 2569): กล้องอุตสาหกรรม Basler + Python + OpenCV
จากภาพหนึ่งภาพ สู่การตรวจชิ้นงาน OK / NG

ผู้สอน: ผศ.ดร.สัญชัย เอียดปราบ  วิศวกรรมระบบสมองกลฝังตัว มหาวิทยาลัยบูรพา

---

## ในชุดนี้มีอะไร

```
cv-industrial-workshop/
├── notebooks/                    ← ผู้เรียนใช้ไฟล์ในโฟลเดอร์นี้
│   ├── 01_image_basics.ipynb       Workshop 1: ภาพคืออะไร (พิกเซล RGB HSV Threshold Canny)
│   ├── 02_object_counting.ipynb    Workshop 2: นับวัตถุตามสี + กล้อง Basler จำลอง
│   ├── 03_ok_ng_inspection.ipynb   Workshop 2B: ตรวจชิ้นงาน OK / NG
│   └── 04_ocr.ipynb                Workshop 3: อ่านตัวอักษร (OCR)
├── scripts/                      ← โปรแกรมภาพสด (live view) สำหรับกล้องจริง
│   ├── vision_camera.py            ต่อกล้อง / แสดงภาพ (ทุก Workshop ใช้ร่วมกัน)
│   ├── workshop1_image_basics.py
│   ├── workshop2_object_counting.py
│   ├── workshop2b_ok_ng_inspection.py
│   └── workshop3_ocr.py
├── images/                       ← ภาพตัวอย่าง
│   ├── shapes.png
│   └── ocr_sample.png
└── requirements.txt
```

---

## 1. ดาวน์โหลด

กดปุ่มสีเขียว **Code → Download ZIP** ด้านบนของหน้านี้ แล้วแตกไฟล์ไว้ที่ Desktop

หรือถ้ามี git:
```
git clone https://github.com/SANCHAIE/cv-industrial-workshop.git
```

---

## 2. ติดตั้ง (ทำครั้งเดียว)

### Windows
1. **Python 3.10 ขึ้นไป** จาก [python.org](https://www.python.org/downloads/)  ติ๊ก **Add Python to PATH** ตอนติดตั้ง
2. **VS Code** จาก [code.visualstudio.com](https://code.visualstudio.com/)  แล้วลงส่วนเสริม (Extension) **Python** และ **Jupyter**
3. **pylon** จาก [baslerweb.com](https://www.baslerweb.com/en/downloads/software/) เลือกติดตั้งแบบ Custom ติ๊ก **Camera Emulation Support** และ **pylon Viewer**
4. **Tesseract OCR** ตัวติดตั้งของ UB Mannheim ไว้ที่ `C:\Program Files\Tesseract-OCR`
5. เปิดโฟลเดอร์ `cv-industrial-workshop` ใน VS Code (File → Open Folder) เปิด Terminal (`Ctrl + `` ` ``) แล้วพิมพ์
   ```
   pip install -r requirements.txt
   ```

### Mac
1. **Python 3.10 ขึ้นไป** จาก python.org
2. **VS Code** + ส่วนเสริม **Python** และ **Jupyter**
3. **pylon สำหรับ macOS** จาก baslerweb.com
4. Tesseract: `brew install tesseract`
5. เปิดโฟลเดอร์ใน VS Code เปิด Terminal แล้วพิมพ์
   ```
   pip3 install -r requirements.txt
   ```

### ทดสอบว่าพร้อม
เปิด `notebooks/01_image_basics.ipynb` ใน VS Code กด **Select Kernel** มุมขวาบน เลือก Python ที่ติดตั้งไว้ แล้วกด **Run All**
ถ้าเห็นภาพรูปทรงหลายสี แปลว่าพร้อมแล้ว

---

## 3. วิธีใช้ Notebook

- คลิกที่ cell แล้วกด **Shift + Enter** รันทีละ cell จากบนลงล่าง
- cell ที่มี **แถบเลื่อน** ให้เลื่อนแล้วดูภาพเปลี่ยนทันที
- Workshop 2B: **รัน cell Trigger หนึ่งครั้ง = ตรวจชิ้นงานหนึ่งครั้ง**

### เลือกแหล่งภาพ
ทุก Notebook ใช้ `grab_one()` ถ่ายภาพ 1 ภาพแล้วปิดกล้องทันที (กล้องไม่ค้าง)

| ค่า | แหล่งภาพ |
|---|---|
| `"file"` | ภาพจากไฟล์ใน `images/` |
| `"emu"` | กล้อง Basler จำลอง (ไม่ต้องมีกล้องจริง) |
| `"basler"` | กล้อง Basler จริง |
| `"webcam"` | webcam |

> **สำคัญ:** ถ้าจะสลับระหว่าง `"emu"` กับ `"basler"` ให้กด **Restart Kernel** ก่อน

---

## 4. ภาพสดจากกล้อง (สำหรับสาธิต)

Notebook ไม่เหมาะกับภาพสดต่อเนื่อง ให้ใช้สคริปต์ใน `scripts/` แทน เปิด Terminal ที่โฟลเดอร์ `scripts` แล้วรัน

```
python workshop2_object_counting.py --source emu
python workshop2_object_counting.py --source basler
python workshop2b_ok_ng_inspection.py --source basler --expect 5
python workshop3_ocr.py --source basler
```
หน้าต่างมีแถบเลื่อนให้ปรับค่า  กด `q` เพื่อออก

---

## 5. สลับไปใช้กล้องจริง

1. **ปิด pylon Viewer ก่อน** กล้องเปิดได้ทีละโปรแกรม
2. เช็กสาย: กล้อง → PoE switch → คอมพิวเตอร์ (โน้ตบุ๊กไม่มีช่อง LAN ใช้อะแดปเตอร์ USB-C เป็น LAN แบบ Gigabit)
3. ตั้งการ์ดแลนเป็น Auto IP / DHCP (กล้องใช้ IP 169.254.x.x)
4. เปลี่ยนแหล่งภาพเป็น `"basler"` (Restart Kernel ก่อน)
5. **จูนค่าใหม่:** Exposure, ช่วง HSV, S min, V min, Min Area
   ค่าที่ใช้กับภาพไฟล์ใช้กับกล้องจริงไม่ได้ทันที เพราะแสงและสีต่างกัน ไม่ได้แปลว่าโค้ดผิด

---

## 6. ปัญหาที่พบบ่อย

| อาการ | วิธีแก้ |
|---|---|
| `ModuleNotFoundError` | ยังไม่ได้ `pip install -r requirements.txt` หรือ Kernel ที่เลือกไม่ตรงกับ Python ที่ติดตั้ง |
| `ไม่พบกล้อง` | ปิด pylon Viewer / เช็กสาย PoE / ตั้งการ์ดแลนเป็น Auto IP |
| กล้องค้าง เปิดไม่ได้ | Restart Kernel ใน Notebook หรือปิดโปรแกรมที่ใช้กล้องอยู่ |
| สลับ emu กับ basler แล้วไม่เปลี่ยน | Restart Kernel ก่อนสลับ |
| `TesseractNotFoundError` | ยังไม่ติดตั้ง Tesseract หรือติดตั้งไว้คนละที่ |
| ตัวอักษรไทยในกราฟเป็นช่องสี่เหลี่ยม | เครื่องไม่มีฟอนต์ไทย ไม่กระทบผลลัพธ์ |
| ปรับ Exposure / Gain แล้วภาพไม่เปลี่ยน | ปกติของกล้องจำลอง ใช้กล้องจริงจึงเห็นผล |

---

## 7. License ของซอฟต์แวร์ที่ใช้

OpenCV และ Tesseract ใช้สัญญาอนุญาต Apache 2.0  Python และ VS Code ใช้ฟรี ทั้งหมดใช้ในเชิงพาณิชย์ได้
pypylon และ pylon ใช้ฟรีกับกล้อง Basler
ควรตรวจสอบเงื่อนไขล่าสุดของแต่ละซอฟต์แวร์ก่อนนำไปใช้ในการผลิตจริง
