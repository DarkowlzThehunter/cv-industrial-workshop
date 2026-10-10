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
│   ├── 02b_emulator_play.ipynb     เล่นกับกล้องจำลอง: AOI, Pixel Format, สายการผลิต, จำลองกล้องจริง
│   ├── 03_ok_ng_inspection.ipynb   Workshop 2B: ตรวจชิ้นงาน OK / NG
│   ├── 04_ocr.ipynb                Workshop 3: อ่านตัวอักษร (OCR)
│   ├── 05_factory_simulation.ipynb จำลองกะการผลิต 40 ถาด: แสงเปลี่ยน ของเสีย 4 แบบ ต้นทุน
│   ├── 06_industrial_tasks.ipynb   งานโรงงาน 4 แบบ: สิ่งแปลกปลอม ฉลาก นับโลหะ วัดขนาด
│   ├── 07_ai_classifier.ipynb      ต่อยอดสู่ AI: สอน Image Classifier เทียบกับการเขียนกฎ
│   ├── 08_product_components.ipynb แยกส่วนประกอบในภาพเดียว (Rack server): ROI map, หาอัตโนมัติ, AI detection
│   └── 09_challenges.ipynb         Challenge 8 ข้อ: แก้โค้ดเองให้ผ่านตัวตรวจอัตโนมัติ (มีเฉลย)
├── scripts/                      ← โปรแกรมภาพสด (live view) สำหรับกล้องจริง
│   ├── vision_camera.py            ต่อกล้อง / แสดงภาพ (ทุก Workshop ใช้ร่วมกัน)
│   ├── workshop1_image_basics.py
│   ├── workshop2_object_counting.py
│   ├── workshop2b_ok_ng_inspection.py
│   └── workshop3_ocr.py
├── images/                       ← ภาพตัวอย่าง
│   ├── shapes.png
│   ├── ocr_sample.png
│   ├── line/                       ภาพถาด 6 ถาด สำหรับสายการผลิตจำลอง
│   ├── factory/                    ภาพถาด 40 ถาด สำหรับจำลองกะการผลิต
│   ├── industrial/                 ภาพถาด ฉลาก ชิ้นส่วนโลหะ สำหรับ notebook 06
│   ├── rack/                       ภาพหน้าตู้ Rack server สำหรับ notebook 08
│   └── factory_labels.csv          เฉลยของแต่ละถาด
└── requirements.txt
```

---

## 1. ติดตั้งโปรแกรม (ทำครั้งเดียว ก่อนวันอบรม)

ใช้เวลาประมาณ 20–30 นาที ต้องต่ออินเทอร์เน็ต ทำตามทีละขั้น ทุกขั้นมี **"ตรวจสอบ"** ให้เช็กว่าทำสำเร็จก่อนไปขั้นถัดไป

- [Windows](#windows)
- [Mac](#mac)
- [ติดตั้งไม่ผ่าน ดูตรงนี้](#5-ปัญหาที่พบบ่อย)

---

### Windows

#### ขั้นที่ 1: ติดตั้ง Python
1. เข้า https://www.python.org/downloads/ กดปุ่มสีเหลือง **Download Python 3.x.x**
2. เปิดไฟล์ที่ดาวน์โหลดมา (อยู่ในโฟลเดอร์ Downloads)
3. **สำคัญที่สุด:** ที่หน้าแรกของตัวติดตั้ง ติ๊กช่อง **Add python.exe to PATH** ด้านล่างก่อน
4. กด **Install Now** รอจนเสร็จ แล้วกด **Close**

**ตรวจสอบ:** กดปุ่ม Windows พิมพ์ `cmd` แล้วกด Enter จะเปิดหน้าต่างสีดำ พิมพ์คำสั่งนี้แล้วกด Enter
```
python --version
```
ต้องขึ้นว่า `Python 3.x.x`
ถ้าขึ้นข้อความว่าไม่รู้จักคำสั่ง หรือเปิด Microsoft Store ขึ้นมา แปลว่าลืมติ๊ก Add to PATH ให้ถอนการติดตั้งแล้วทำขั้นที่ 1 ใหม่

#### ขั้นที่ 2: ติดตั้ง VS Code และส่วนเสริม
1. เข้า https://code.visualstudio.com/ กด **Download for Windows**
2. เปิดไฟล์ที่ดาวน์โหลดมา กด **I accept** แล้วกด **Next** ไปเรื่อยๆ จนกด **Install** และ **Finish**
3. เปิด VS Code กดปุ่ม **Ctrl + Shift + X** จะเปิดหน้าส่วนเสริม (Extensions) ทางซ้าย
4. พิมพ์ค้นหา `Python` เลือกตัวที่ผู้สร้างคือ **Microsoft** แล้วกด **Install**
5. พิมพ์ค้นหา `Jupyter` เลือกตัวที่ผู้สร้างคือ **Microsoft** แล้วกด **Install**

**ตรวจสอบ:** พิมพ์ `@installed` ในช่องค้นหาส่วนเสริม ต้องเห็นทั้ง Python และ Jupyter

#### ขั้นที่ 3: ดาวน์โหลดชุด Workshop
1. เข้า https://github.com/SANCHAIE/cv-industrial-workshop
2. กดปุ่มสีเขียว **Code** แล้วกด **Download ZIP**
3. เปิดโฟลเดอร์ Downloads คลิกขวาที่ไฟล์ `cv-industrial-workshop-main.zip` เลือก **Extract All...** แล้วกด **Extract**
4. ย้ายโฟลเดอร์ `cv-industrial-workshop-main` ที่แตกออกมาไปไว้ที่ **Desktop**

**ตรวจสอบ:** เปิดโฟลเดอร์ `cv-industrial-workshop-main` บน Desktop ต้องเห็นโฟลเดอร์ `notebooks`, `scripts`, `images` และไฟล์ `requirements.txt`
(ถ้าเห็นโฟลเดอร์ชื่อเดียวกันซ้อนอีกชั้น ให้ใช้โฟลเดอร์ข้างในแทน)

#### ขั้นที่ 4: ติดตั้งไลบรารี Python
1. ใน VS Code กดเมนู **File → Open Folder...** เลือกโฟลเดอร์ `cv-industrial-workshop-main` บน Desktop แล้วกด **Select Folder**
2. ถ้ามีหน้าต่างถามว่าเชื่อถือโฟลเดอร์นี้ไหม กด **Yes, I trust the authors**
3. กดเมนู **Terminal → New Terminal** จะมีช่องพิมพ์คำสั่งเปิดขึ้นด้านล่าง
4. พิมพ์คำสั่งนี้แล้วกด Enter รอประมาณ 3–5 นาที
```
python -m pip install -r requirements.txt
```

**ตรวจสอบ:** บรรทัดสุดท้ายต้องขึ้นว่า `Successfully installed ...` (ถ้ามีข้อความสีเหลือง WARNING ไม่เป็นไร)

#### ขั้นที่ 5: ทดสอบเปิด Notebook
1. ในแถบซ้ายของ VS Code เปิดโฟลเดอร์ `notebooks` แล้วคลิกไฟล์ `01_image_basics.ipynb`
2. กด **Select Kernel** มุมขวาบน → **Python Environments** → เลือก **Python 3.x.x** ที่ติดตั้งในขั้นที่ 1
3. ถ้ามีหน้าต่างขอติดตั้ง `ipykernel` กด **Install**
4. กด **Run All** ด้านบน

**ตรวจสอบ:** ใต้ cell ต้องขึ้นข้อความ `OpenCV 4.x.x พร้อมใช้งาน` และเห็นภาพรูปทรงหลายสี **ถ้าเห็นแปลว่าพร้อมสำหรับวันที่ 1 แล้ว**

#### ขั้นที่ 6: ติดตั้ง Tesseract (ใช้วันที่ 2 อ่านตัวอักษร)
1. เข้า https://github.com/UB-Mannheim/tesseract/wiki
2. กดลิงก์ไฟล์ติดตั้งล่าสุด ชื่อประมาณ `tesseract-ocr-w64-setup-5.x.x.exe`
3. เปิดไฟล์ กด **Next** ไปเรื่อยๆ **ไม่ต้องเปลี่ยนที่ติดตั้ง** (ต้องเป็น `C:\Program Files\Tesseract-OCR`) แล้วกด **Install**

**ตรวจสอบ:** เปิด `notebooks/04_ocr.ipynb` กด Run All ต้องเห็นคำว่า `LOT 775758`

#### ขั้นที่ 7 (ไม่บังคับ): ติดตั้ง pylon Viewer
ใช้ดูภาพจากกล้อง Basler และปรับค่ากล้องแบบกดปุ่ม **Notebook ใช้กล้องจำลองได้แม้ไม่ติดตั้งข้อนี้**
1. เข้า https://www.baslerweb.com/en/downloads/software/ เลือก **pylon** รุ่นล่าสุด → **Windows** → **pylon Software Suite for Windows** (ไฟล์ใหญ่ประมาณ 1.6 GB)
2. เปิดไฟล์ เลือก Profile เป็น **Custom**
3. ติ๊ก **Camera Emulation Support** และ **pylon Viewer** แล้วกด Next จนติดตั้งเสร็จ

---

### Mac

#### ขั้นที่ 1: ติดตั้ง Python
1. เข้า https://www.python.org/downloads/ กดปุ่มสีเหลือง **Download Python 3.x.x**
2. เปิดไฟล์ `.pkg` ที่ดาวน์โหลดมา กด **Continue / Agree / Install** ใส่รหัสผ่านเครื่องเมื่อถูกถาม
3. เมื่อติดตั้งเสร็จ จะมีหน้าต่างโฟลเดอร์ Python 3.x เปิดขึ้นมา **ดับเบิลคลิกไฟล์ `Install Certificates.command`** (ถ้าไม่ทำ ขั้นที่ 4 อาจดาวน์โหลดไม่ได้)

**ตรวจสอบ:** กด **Cmd + Space** พิมพ์ `Terminal` แล้วกด Enter พิมพ์คำสั่งนี้แล้วกด Enter
```
python3 --version
```
ต้องขึ้นว่า `Python 3.x.x`

#### ขั้นที่ 2: ติดตั้ง VS Code และส่วนเสริม
1. เข้า https://code.visualstudio.com/ กด **Download for Mac**
2. เปิดไฟล์ zip ที่ดาวน์โหลดมา แล้วลาก **Visual Studio Code** ไปไว้ในโฟลเดอร์ **Applications**
3. เปิด VS Code กด **Cmd + Shift + X** เปิดหน้าส่วนเสริม
4. ค้นหาและติดตั้ง **Python** และ **Jupyter** (ผู้สร้างคือ Microsoft ทั้งคู่)

**ตรวจสอบ:** พิมพ์ `@installed` ในช่องค้นหาส่วนเสริม ต้องเห็นทั้ง Python และ Jupyter

#### ขั้นที่ 3: ดาวน์โหลดชุด Workshop
1. เข้า https://github.com/SANCHAIE/cv-industrial-workshop
2. กดปุ่มสีเขียว **Code** แล้วกด **Download ZIP** (Safari จะแตกไฟล์ให้เอง ถ้าไม่แตก ให้ดับเบิลคลิกไฟล์ zip)
3. ย้ายโฟลเดอร์ `cv-industrial-workshop-main` ไปไว้ที่ **Desktop**

**ตรวจสอบ:** ในโฟลเดอร์ต้องเห็น `notebooks`, `scripts`, `images` และ `requirements.txt`

#### ขั้นที่ 4: ติดตั้งไลบรารี Python
1. ใน VS Code กด **File → Open Folder...** เลือกโฟลเดอร์ `cv-industrial-workshop-main` บน Desktop แล้วกด **Open**
2. ถ้าถามว่าเชื่อถือโฟลเดอร์ไหม กด **Yes, I trust the authors**
3. กด **Terminal → New Terminal**
4. พิมพ์คำสั่งนี้แล้วกด Enter รอประมาณ 3–5 นาที
```
python3 -m pip install -r requirements.txt
```

**ตรวจสอบ:** บรรทัดสุดท้ายต้องขึ้นว่า `Successfully installed ...`

#### ขั้นที่ 5: ทดสอบเปิด Notebook
1. เปิดไฟล์ `notebooks/01_image_basics.ipynb`
2. กด **Select Kernel** มุมขวาบน → **Python Environments** → เลือก **Python 3.x.x** (ที่อยู่ `/Library/Frameworks/...` หรือ `/usr/local/bin/python3`)
3. ถ้าขอติดตั้ง `ipykernel` กด **Install**
4. กด **Run All**

**ตรวจสอบ:** เห็นข้อความ `OpenCV 4.x.x พร้อมใช้งาน` และภาพรูปทรงหลายสี

#### ขั้นที่ 6: ติดตั้ง Tesseract (ใช้วันที่ 2 อ่านตัวอักษร)
Mac ต้องติดตั้งผ่าน Homebrew ทำใน Terminal ทีละคำสั่ง

1. ติดตั้ง Homebrew (ถ้าเคยติดตั้งแล้วข้ามได้) คัดลอกคำสั่งนี้ทั้งบรรทัดไปวางใน Terminal แล้วกด Enter ใส่รหัสผ่านเครื่องเมื่อถูกถาม (พิมพ์แล้วจะไม่เห็นตัวอักษร เป็นเรื่องปกติ) รอ 5–10 นาที
```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
2. ถ้าเครื่องเป็นชิป Apple (M1/M2/M3/M4) ให้รัน 2 คำสั่งนี้ต่อ (ตัวติดตั้งจะแสดงคำสั่งนี้ตอนท้ายด้วย)
```
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```
3. ติดตั้ง Tesseract
```
brew install tesseract
```

**ตรวจสอบ:** พิมพ์ `tesseract --version` ต้องขึ้นเลขเวอร์ชัน แล้วลองเปิด `notebooks/04_ocr.ipynb` กด Run All ต้องเห็นคำว่า `LOT 775758`

#### ขั้นที่ 7 (ไม่บังคับ): ติดตั้ง pylon Viewer
เข้า https://www.baslerweb.com/en/downloads/software/ เลือก **pylon** รุ่นล่าสุด → **macOS** ดาวน์โหลดไฟล์ `.dmg` เปิดแล้วติดตั้งตามขั้นตอน
**Notebook ใช้กล้องจำลองได้แม้ไม่ติดตั้งข้อนี้**

---

### เช็กลิสต์ก่อนวันอบรม
- [ ] `python --version` (Mac: `python3 --version`) ขึ้นเลขเวอร์ชัน
- [ ] VS Code มีส่วนเสริม Python และ Jupyter
- [ ] `01_image_basics.ipynb` กด Run All แล้วเห็นภาพ
- [ ] `04_ocr.ipynb` กด Run All แล้วเห็น `LOT 775758`

ติดขั้นไหน ถ่ายภาพหน้าจอข้อความ error ไว้ แล้วนำมาให้ TA ดูในวันอบรม

---

## 2. วิธีใช้ Notebook

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

## 3. ภาพสดจากกล้อง (สำหรับสาธิต)

Notebook ไม่เหมาะกับภาพสดต่อเนื่อง ให้ใช้สคริปต์ใน `scripts/` แทน เปิด Terminal ที่โฟลเดอร์ `scripts` แล้วรัน

```
python workshop2_object_counting.py --source emu
python workshop2_object_counting.py --source basler
python workshop2b_ok_ng_inspection.py --source basler --expect 5
python workshop3_ocr.py --source basler
```
หน้าต่างมีแถบเลื่อนให้ปรับค่า  กด `q` เพื่อออก

---

## 4. สลับไปใช้กล้องจริง

1. **ปิด pylon Viewer ก่อน** กล้องเปิดได้ทีละโปรแกรม
2. เช็กสาย: กล้อง → PoE switch → คอมพิวเตอร์ (โน้ตบุ๊กไม่มีช่อง LAN ใช้อะแดปเตอร์ USB-C เป็น LAN แบบ Gigabit)
3. ตั้งการ์ดแลนเป็น Auto IP / DHCP (กล้องใช้ IP 169.254.x.x)
4. เปลี่ยนแหล่งภาพเป็น `"basler"` (Restart Kernel ก่อน)
5. **จูนค่าใหม่:** Exposure, ช่วง HSV, S min, V min, Min Area
   ค่าที่ใช้กับภาพไฟล์ใช้กับกล้องจริงไม่ได้ทันที เพราะแสงและสีต่างกัน ไม่ได้แปลว่าโค้ดผิด

---

## 5. ปัญหาที่พบบ่อย

| อาการ | วิธีแก้ |
|---|---|
| `python` ไม่รู้จักคำสั่ง / เปิด Microsoft Store | ติดตั้ง Python ใหม่ ติ๊ก **Add python.exe to PATH** |
| `pip` ไม่รู้จักคำสั่ง | ใช้ `python -m pip install -r requirements.txt` (Mac: `python3 -m pip ...`) |
| `No such file: requirements.txt` | Terminal ไม่ได้อยู่ในโฟลเดอร์ชุด Workshop ให้เปิดโฟลเดอร์ด้วย File → Open Folder ก่อน แล้วเปิด Terminal ใหม่ |
| Mac: `SSL: CERTIFICATE_VERIFY_FAILED` | ดับเบิลคลิก `Install Certificates.command` ในโฟลเดอร์ Applications → Python 3.x |
| Mac: `externally-managed-environment` | ใช้ Python จาก python.org แทน Python ของ Homebrew แล้วเลือก Kernel ให้ตรง |
| ไม่มีปุ่ม Select Kernel / Run All | ยังไม่ได้ติดตั้งส่วนเสริม Jupyter ใน VS Code |
| Mac: `import sklearn` แล้ว error เรื่อง scipy (`dlopen ... __thread_bss`) | `python3 -m pip install scipy==1.11.4` |
| `ModuleNotFoundError` | ยังไม่ได้ `pip install -r requirements.txt` หรือ Kernel ที่เลือกไม่ตรงกับ Python ที่ติดตั้ง |
| `ไม่พบกล้อง` | ปิด pylon Viewer / เช็กสาย PoE / ตั้งการ์ดแลนเป็น Auto IP |
| กล้องค้าง เปิดไม่ได้ | Restart Kernel ใน Notebook หรือปิดโปรแกรมที่ใช้กล้องอยู่ |
| สลับ emu กับ basler แล้วไม่เปลี่ยน | Restart Kernel ก่อนสลับ |
| `TesseractNotFoundError` | ยังไม่ติดตั้ง Tesseract หรือติดตั้งไว้คนละที่ |
| ตัวอักษรไทยในกราฟเป็นช่องสี่เหลี่ยม | เครื่องไม่มีฟอนต์ไทย ไม่กระทบผลลัพธ์ |
| ปรับ Exposure / Gain แล้วภาพไม่เปลี่ยน | ปกติของกล้องจำลอง ใช้กล้องจริงจึงเห็นผล |

---

## 6. License ของซอฟต์แวร์ที่ใช้

OpenCV และ Tesseract ใช้สัญญาอนุญาต Apache 2.0  Python และ VS Code ใช้ฟรี ทั้งหมดใช้ในเชิงพาณิชย์ได้
pypylon และ pylon ใช้ฟรีกับกล้อง Basler
ควรตรวจสอบเงื่อนไขล่าสุดของแต่ละซอฟต์แวร์ก่อนนำไปใช้ในการผลิตจริง
