"""
vision_camera.py : แหล่งภาพที่ใช้ร่วมกันทุก Workshop
  file   = ภาพจากไฟล์
  emu    = กล้อง Basler จำลอง (ไม่ต้องมีกล้องจริง)
  basler = กล้อง Basler จริง (GigE / USB)
  webcam = webcam สำรอง

ทุกแหล่งมีคำสั่งเหมือนกัน:  frame = src.read()   และ   src.close()
frame ที่ได้เป็นภาพสี BGR (numpy array) ส่งเข้า OpenCV ได้ทันที
"""
import os

import cv2
import numpy as np

# โฟลเดอร์ images/ ของ repo (ใช้ได้ไม่ว่าจะรันจากโฟลเดอร์ไหน)
IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")


def img_path(name):
    """คืน path เต็มของไฟล์ภาพตัวอย่างใน images/"""
    return name if os.path.exists(name) else os.path.join(IMG_DIR, name)


def imread_url(url, timeout=10):
    """โหลดภาพจาก URL (ต้องเป็นลิงก์ไฟล์ภาพโดยตรง เช่น .jpg / .png) คืนภาพ BGR เหมือน cv2.imread"""
    import ssl
    import urllib.request
    import numpy as np
    try:
        import certifi              # Python จาก python.org บน Mac ไม่มีใบรับรอง SSL ในตัว
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    # บางเว็บบล็อกโปรแกรมที่ไม่บอกชื่อ browser จึงใส่ User-Agent ไว้
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        data = r.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"ข้อมูลจาก {url} ไม่ใช่ไฟล์ภาพ (ลิงก์ต้องชี้ไปที่ไฟล์ภาพโดยตรง)")
    return img


def set_param(cam, names, value):
    """ชื่อ parameter ต่างกันตามรุ่นกล้อง เช่น ExposureTime (รุ่นใหม่) / ExposureTimeAbs (ace classic)"""
    for n in names:
        try:
            getattr(cam, n).Value = value
            print(f"  ตั้ง {n} = {value}")
            return True
        except Exception:
            continue
    print(f"  (ข้าม {names[0]} - กล้องรุ่นนี้ไม่มี parameter นี้)")
    return False


class FileSource:
    def __init__(self, path):
        self.img = cv2.imread(path)
        if self.img is None:
            raise FileNotFoundError(f"เปิดไฟล์ไม่ได้: {path}")

    def read(self):
        return self.img.copy()

    def close(self):
        pass


class WebcamSource:
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError("เปิด webcam ไม่ได้")

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        self.cap.release()


class BaslerSource:
    def __init__(self, emulate=False, image_path=None,
                 exposure_us=5000, gain=0):
        if emulate:
            os.environ["PYLON_CAMEMU"] = "1"     # ต้องตั้งก่อน import pypylon
        from pypylon import pylon
        self.pylon = pylon

        try:
            device = pylon.TlFactory.GetInstance().CreateFirstDevice()
        except Exception:
            raise RuntimeError(
                "ไม่พบกล้อง - เช็กสาย PoE / ปิด pylon Viewer ก่อน / ตั้งการ์ดแลนเป็น Auto IP")
        self.cam = pylon.InstantCamera(device)
        self.cam.Open()
        print("เชื่อมต่อกล้อง:", self.cam.GetDeviceInfo().GetModelName())

        if emulate:
            if image_path:
                # ให้กล้องจำลองส่งภาพจากไฟล์ แทน test pattern (ต้องปิด test image ก่อน)
                set_param(self.cam, ["TestImageSelector"], "Off")
                set_param(self.cam, ["ImageFilename"], os.path.abspath(image_path))
                set_param(self.cam, ["ImageFileMode"], "On")
            # กล้องจำลองเริ่มที่ Mono8 (ขาวดำ) -> เปลี่ยนเป็นสี
            set_param(self.cam, ["PixelFormat"], "BGR8Packed")

        # concept กล้องอุตสาหกรรม: กำหนดค่าเอง ไม่ปล่อย auto -> ผลซ้ำได้ทุกครั้ง
        set_param(self.cam, ["ExposureAuto"], "Off")
        set_param(self.cam, ["ExposureTime", "ExposureTimeAbs"], exposure_us)
        set_param(self.cam, ["GainAuto"], "Off")
        set_param(self.cam, ["Gain", "GainRaw"], gain)

        # แปลง Bayer / Mono จากกล้อง -> BGR สำหรับ OpenCV
        self.conv = pylon.ImageFormatConverter()
        self.conv.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.conv.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        # เอาเฉพาะภาพล่าสุด (สายการผลิตต้องการภาพปัจจุบัน ไม่ใช่ภาพค้างคิว)
        self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def read(self):
        res = self.cam.RetrieveResult(5000, self.pylon.TimeoutHandling_ThrowException)
        try:
            if res.GrabSucceeded():
                return self.conv.Convert(res).GetArray()
            return None
        finally:
            res.Release()

    def close(self):
        self.cam.StopGrabbing()
        self.cam.Close()


def add_source_args(ap, default_image):
    """เพิ่ม argument มาตรฐานให้ทุก Workshop"""
    ap.add_argument("--source", choices=["file", "emu", "basler", "webcam"],
                    default="file")
    ap.add_argument("--image", default=default_image)
    ap.add_argument("--exposure", type=int, default=5000, help="ไมโครวินาที")
    ap.add_argument("--gain", type=float, default=0)


def open_source(args):
    if args.source == "file":
        return FileSource(img_path(args.image))
    if args.source == "webcam":
        return WebcamSource()
    return BaslerSource(emulate=(args.source == "emu"), image_path=img_path(args.image),
                        exposure_us=args.exposure, gain=args.gain)


def fit_to_screen(img, max_w=1600):
    """ย่อเฉพาะตอนแสดงผล (ภาพกล้องจริง 2448x2048 ใหญ่เกินจอ)"""
    if img.shape[1] > max_w:
        s = max_w / img.shape[1]
        img = cv2.resize(img, None, fx=s, fy=s)
    return img


def use_thai_font():
    """ตั้งฟอนต์ไทยให้ matplotlib ทั้ง notebook (ทั้ง show() และกราฟที่วาดเอง)
    เลือกเฉพาะฟอนต์ที่มีในเครื่องจริง: Windows = Tahoma / Leelawadee UI, Mac = Thonburi / Tahoma"""
    import logging
    import warnings
    try:
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ImportError:
        return
    have = {f.name for f in font_manager.fontManager.ttflist}
    thai = [f for f in ("Tahoma", "Thonburi", "Leelawadee UI", "Noto Sans Thai",
                        "Sarabun", "Loma") if f in have]
    plt.rcParams["font.family"] = thai + ["DejaVu Sans"]
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message="Glyph .* missing from font")


use_thai_font()   # เรียกทันทีที่ import vision_camera


def show(*imgs, titles=None, size=6):
    """แสดงภาพใน Jupyter Notebook (แปลง BGR เป็น RGB ให้อัตโนมัติ)"""
    import logging
    import warnings
    import matplotlib.pyplot as plt
    # ฟอนต์ที่มีภาษาไทย: Windows = Tahoma / Leelawadee UI, Mac = Thonburi
    # เลือกเฉพาะฟอนต์ไทยที่มีในเครื่องนี้จริง จึงไม่มีข้อความ findfont ... not found
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    thai = [f for f in ("Tahoma", "Leelawadee UI", "Thonburi", "Noto Sans Thai") if f in have]
    plt.rcParams["font.family"] = thai + ["DejaVu Sans"]
    warnings.filterwarnings("ignore", message="Glyph")
    warnings.filterwarnings("ignore", message="findfont")
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # findfont มาทาง logging ไม่ใช่ warnings
    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(size * n, size * 0.75))
    axes = [axes] if n == 1 else axes
    for i, (ax, im) in enumerate(zip(axes, imgs)):
        if im.ndim == 2:
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        if titles:
            ax.set_title(titles[i])
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def grab_one(source="emu", image="shapes.png", exposure_us=5000, gain=0):
    """ถ่ายภาพ 1 ภาพแล้วปิดกล้องทันที (เหมาะกับ Notebook: กล้องไม่ค้าง)
    source = "file" | "emu" | "basler" | "webcam"
    """
    if source == "file":
        return cv2.imread(img_path(image))
    if source == "webcam":
        cam = WebcamSource()
    else:
        cam = BaslerSource(emulate=(source == "emu"), image_path=img_path(image),
                           exposure_us=exposure_us, gain=gain)
    try:
        frame = None
        for _ in range(5):          # ลองหลายครั้งเผื่อเฟรมแรกยังไม่พร้อม
            frame = cam.read()
            if frame is not None:
                break
        return frame
    finally:
        cam.close()                 # ปิดกล้องเสมอ แม้เกิด error


def put_label(img, text, org=(10, 30), scale=0.8, color=(0, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


# ---------------------------------------------------------------
# สำหรับ Notebook 02b: เล่นกับกล้องจำลอง
# ---------------------------------------------------------------
def grab_emu(image="shapes.png", pixel_format="BGR8Packed",
             width=None, height=None, offset_x=0, offset_y=0):
    """ถ่าย 1 ภาพจากกล้องจำลอง พร้อมตั้ง Pixel Format และ AOI (ตัดภาพ) แล้วปิดกล้อง"""
    os.environ["PYLON_CAMEMU"] = "1"
    from pypylon import pylon
    cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    cam.Open()
    try:
        cam.TestImageSelector.Value = "Off"
        cam.ImageFilename.Value = os.path.abspath(img_path(image))
        cam.ImageFileMode.Value = "On"
        cam.PixelFormat.Value = pixel_format
        # AOI: ต้องตั้ง Offset เป็น 0 ก่อน แล้วค่อยตั้งขนาด และค่อยเลื่อน Offset
        cam.OffsetX.Value = 0
        cam.OffsetY.Value = 0
        cam.Width.Value = width or cam.Width.Max
        cam.Height.Value = height or cam.Height.Max
        cam.OffsetX.Value = min(offset_x, cam.OffsetX.Max)
        cam.OffsetY.Value = min(offset_y, cam.OffsetY.Max)
        conv = pylon.ImageFormatConverter()
        conv.OutputPixelFormat = pylon.PixelType_BGR8packed
        res = cam.GrabOne(3000)
        try:
            return conv.Convert(res).GetArray()
        finally:
            res.Release()
    finally:
        cam.Close()


class EmuLine:
    """สายการผลิตจำลอง: กล้องจำลองส่งภาพจากโฟลเดอร์ทีละภาพ ทุกครั้งที่สั่ง Trigger
    ใช้แบบ:  line = EmuLine("line")  ->  frame = line.trigger()  ->  line.close()"""

    def __init__(self, folder="line"):
        os.environ["PYLON_CAMEMU"] = "1"
        from pypylon import pylon
        self.pylon = pylon
        self.cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.cam.Open()
        self.cam.TestImageSelector.Value = "Off"
        self.cam.ImageFilename.Value = os.path.abspath(img_path(folder))
        self.cam.ImageFileMode.Value = "On"
        self.cam.PixelFormat.Value = "BGR8Packed"
        # Trigger แบบ Software: กล้องรอคำสั่งก่อนถ่ายทุกภาพ เหมือนรอเซนเซอร์ในไลน์ผลิต
        self.cam.TriggerSelector.Value = "FrameStart"
        self.cam.TriggerMode.Value = "On"
        self.cam.TriggerSource.Value = "Software"
        self.conv = pylon.ImageFormatConverter()
        self.conv.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.cam.StartGrabbing()

    def trigger(self):
        p = self.pylon
        self.cam.WaitForFrameTriggerReady(2000, p.TimeoutHandling_ThrowException)
        self.cam.ExecuteSoftwareTrigger()
        res = self.cam.RetrieveResult(3000, p.TimeoutHandling_ThrowException)
        try:
            return self.conv.Convert(res).GetArray()
        finally:
            res.Release()

    def close(self):
        self.cam.StopGrabbing()
        self.cam.Close()


def simulate_real(img, exposure_ms=5.0, gain_db=0.0, light=1.0, speed=0.0, seed=0):
    """ทำให้ภาพจากกล้องจำลองเหมือนกล้องจริง (กล้องจำลองไม่เปลี่ยนภาพตาม Exposure / Gain)
    exposure_ms : เวลาเปิดรับแสง (มิลลิวินาที) มากขึ้น = สว่างขึ้น และเบลอถ้าชิ้นงานเคลื่อนที่
    gain_db     : ขยายสัญญาณ มากขึ้น = สว่างขึ้น แต่จุดรบกวนมากขึ้น
    light       : ความสว่างของไฟในห้อง (1.0 = ปกติ)
    speed       : ความเร็วชิ้นงาน (พิกเซล ต่อ มิลลิวินาที) 0 = อยู่นิ่ง
    """
    rng = np.random.default_rng(seed)
    g = 10 ** (gain_db / 20)
    x = img.astype(np.float32) * light * (exposure_ms / 5.0) * g
    blur = int(round(speed * exposure_ms))
    if blur > 1:
        k = np.zeros((1, blur), np.float32)
        k[0, :] = 1.0 / blur
        x = cv2.filter2D(x, -1, k)
    x += rng.normal(0, 1.5 + 2.5 * g, x.shape)      # จุดรบกวน ขยายตาม Gain
    return np.clip(x, 0, 255).astype(np.uint8)
