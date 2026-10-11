# สถานีตรวจชิ้นงาน (Web app)  รันด้วย:  python3 -m streamlit run app/inspection_app.py
import csv
import difflib
import os
import re
import shutil
import sys
import threading
import time
from collections import namedtuple
from datetime import datetime

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from vision_camera import (BaslerSource, WebcamSource, set_param,  # noqa: E402
                           simulate_real, img_path, put_label)

st.set_page_config(page_title='สถานีตรวจชิ้นงาน', page_icon='📷', layout='wide')
Q = chr(34)
IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp')
OK_REASON = 'ครบถ้วน'

# ---------------- วิธีตรวจ ----------------
METHODS = {
    'color': 'นับวัตถุตามสี (HSV)',
    'reference': 'เทียบภาพอ้างอิง',
    'label': 'ตรวจฉลาก (Template + OCR)',
    'parts': 'นับและแยกชนิดชิ้นงาน (Watershed)',
    'measure': 'วัดขนาดเป็นมิลลิเมตร',
    'ai': 'AI จำแนกถาด (SVM)',
    'rack_zones': 'Rack server: แผนที่โซน',
    'rack_parts': 'Rack server: หาส่วนประกอบอัตโนมัติ',
    'zones': 'แผนที่โซน เทียบภาพอ้างอิง',
    'count': 'นับชิ้นงาน (Otsu)',
}
ALIGN_MODES = {'none': 'ไม่จัดตำแหน่ง (มีจิ๊กยึดตำแหน่ง)',
               'shift': 'หาระยะเลื่อน (Template matching)',
               'ecc': 'หมุน เลื่อน ย่อขยาย (ECC รูปมือถือ)'}
ALIGNABLE = ('reference', 'zones', 'count')      # วิธีที่จัดตำแหน่งภาพให้ตรงภาพอ้างอิงก่อนตรวจได้
NEEDS_REF = ('reference', 'label', 'zones')      # วิธีที่ต้องมีภาพอ้างอิง / ต้นแบบเสมอ
ALIGN_MIN = 0.6                                  # ความเหมือนขั้นต่ำหลังจัดตำแหน่ง ต่ำกว่านี้ให้ถ่ายใหม่


def needs_ref(s):
    return s['method'] in NEEDS_REF or (s['method'] in ALIGNABLE and s['align'] != 'none')


def _w10_images():
    # Workshop 10: ถ้ามีรูปของคุณใน images/my_project/ (ต้องมี good.jpg) ใช้รูปนั้น  ไม่งั้นใช้รูปจำลองจากมือถือ
    d = img_path('my_project')
    if os.path.isfile(os.path.join(d, 'good.jpg')):
        others = sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXT) and f != 'good.jpg')
        return 'my_project/good.jpg', ('my_project/' + others[0]) if others else 'my_project/good.jpg'
    return 'phone_demo/good.jpg', 'phone_demo/photo_01_clean.jpg'


W10_REF, W10_IMG = _w10_images()
W10_ZONES = '\n'.join(f'slot{i + 1}: {x}, {y}, 120, 120' for i, (x, y) in
                      enumerate([(90, 90), (340, 90), (590, 90), (90, 365), (340, 365), (590, 365)]))

# ---------------- สูตรงาน (Recipe): ค่าตั้งสำเร็จรูปตาม Workshop ----------------
RECIPES = {
    'Workshop 2: นับวัตถุสีแดง': dict(
        desc='ภาพ shapes.png  นับชิ้นแดงใหญ่ ต้องได้ 5 ชิ้น',
        source='ภาพตัวอย่าง', image='shapes.png', h_range=(0, 10), s_min=50, v_min=50,
        min_area=1000, expect=5, chk_chip=False, chk_foreign=False, sim=False),
    'Workshop 2: นับวัตถุสีน้ำเงิน': dict(
        desc='ภาพเดียวกัน เปลี่ยนช่วงสีเป็นน้ำเงิน (H 95–125)',
        source='ภาพตัวอย่าง', image='shapes.png', h_range=(95, 125), s_min=50, v_min=50,
        min_area=1000, expect=3, chk_chip=False, chk_foreign=False, sim=False),
    'Workshop 2B: ตรวจถาดสายการผลิต': dict(
        desc='ถาด 6 ใบจากกล้องจำลอง  ตรวจครบ 5 ชิ้น + เศษแตก + สิ่งแปลกปลอมสีส้ม',
        source='กล้อง Basler จำลอง', image='line/tray_01.png', h_range=(0, 10), s_min=50, v_min=50,
        min_area=1000, expect=5, chk_chip=True, chk_foreign=True, sim=False),
    'จำลองกะการผลิต: แสงน้อย': dict(
        desc='ถาดจากกะการผลิต แสงรอบไลน์เหลือ 20%  ลองแก้ด้วย Exposure หรือ V ต่ำสุด',
        source='ภาพตัวอย่าง', image='factory/part_001.png', h_range=(0, 10), s_min=50, v_min=50,
        min_area=1000, expect=5, chk_chip=True, chk_foreign=True, sim=True, light=0.2, speed=0.0),
    'จำลองกะการผลิต: สายพานเร็ว': dict(
        desc='ชิ้นงานวิ่งเร็ว ภาพเบลอ  ลองลด Exposure แล้วเพิ่ม Gain',
        source='ภาพตัวอย่าง', image='factory/part_001.png', h_range=(0, 10), s_min=50, v_min=50,
        min_area=1000, expect=5, chk_chip=True, chk_foreign=True, sim=True, light=1.0, speed=2.0,
        exposure=10.0),
    'กล้องจริง: ชิ้นงานสีแดง': dict(
        desc='ตั้งภาพใน pylon Viewer ก่อน  ภาพจริงใหญ่ 16 เท่า พื้นที่ขั้นต่ำจึงสูงขึ้น',
        source='กล้อง Basler จริง', image='shapes.png', h_range=(0, 10), s_min=80, v_min=60,
        min_area=15000, expect=5, chk_chip=False, chk_foreign=False, sim=False),

    # ----- Workshop 5: จำลองกะการผลิต 40 ถาด (notebook 05) -----
    'Workshop 5 ภารกิจ 1: เดินไลน์ 1 กะ ค่าเริ่มต้น': dict(
        desc='แท็บ 🏭 เดินไลน์ 1 กะ กด ▶  ควรได้ ตรวจถูก 28/40 เสียหาย 8,160 บาท  '
             'แล้วลอง Gain 12 (ภารกิจ 2) หรือเปิดกล่องครอบ (ภารกิจ 3)',
        image='factory/part_001.png', sim=True, light=1.0, speed=1.0, exposure=5.0, gain=0.0),
    'Workshop 5 ภารกิจ 4: ครอบกล่อง + กฎตรวจครบ': dict(
        desc='แสงคงที่ + ตรวจเศษแตกและสิ่งแปลกปลอม  ควรได้ ตรวจถูก 40/40 เสียหาย 0 บาท',
        image='factory/part_001.png', chk_chip=True, chk_foreign=True,
        sim=True, light=1.0, speed=1.0, exposure=5.0, gain=0.0, enclosure=True),
    'Workshop 5 ภารกิจ 5: ไม่ครอบกล่อง + Auto Exposure': dict(
        desc='สายพานเร็ว 2  ช่วงเย็นแสงน้อย Exposure นานจนภาพเบลอ ผิด 1 ถาด  '
             'ลดความเร็วทีละ 0.5 หาความเร็วสูงสุดที่ยังถูกครบ (ควรได้ 1.5)',
        image='factory/part_029.png', chk_chip=True, chk_foreign=True,
        sim=True, light=0.2, speed=2.0, exposure=5.0, gain=0.0, auto_exposure=True),

    # ----- Workshop 6: งานตรวจ 4 แบบที่ไม่ได้ดูแค่สี (notebook 06) -----
    'Workshop 6.1: เทียบภาพอ้างอิง หาสิ่งแปลกปลอม/ชิ้นหาย': dict(
        desc='ถาดต้องเหมือนภาพอ้างอิง  ลอง tray_01–05 (สกรู เส้นผม ชิ้นหาย เศษพลาสติก)  '
             'แล้วลอง Threshold 8 และ 60',
        method='reference', image='industrial/tray_01_clean.png', ref_image='industrial/tray_reference.png',
        diff_threshold=20, min_area=120),
    'Workshop 6.2: ตรวจฉลากสินค้า': dict(
        desc='หาฉลาก → เทียบการพิมพ์กับต้นแบบ → OCR ชื่อสินค้า + EXP  ลอง label_01–05  '
             '(ต้องติดตั้ง Tesseract)',
        method='label', image='industrial/label_01_good.png', ref_image='industrial/label_golden.png'),
    'Workshop 6.3: นับและแยกแหวน/น็อต': dict(
        desc='แยกชิ้นที่ติดกันด้วย Watershed แยกชนิดด้วยขนาดรู  ชุดต้องมีแหวน 7 น็อต 5  '
             'ลอง parts_02 และ peak 0.3 / 0.8',
        method='parts', image='industrial/parts_01.png'),
    'Workshop 6.4: วัดขนาดเป็นมิลลิเมตร': dict(
        desc='สอบเทียบด้วยเหรียญ 20 มม. แล้ววัดชิ้นงาน  สเปก 25.5 x 16.0 มม. ± 0.3',
        method='measure', image='industrial/measure_01.png'),

    # ----- Workshop 7: AI (notebook 07) -----
    'Workshop 7: AI จำแนกถาด (SVM)': dict(
        desc='สอนโมเดลจากถาดจำลอง 300 ภาพ (ขยับ ±10 px แสง 60–130%) ครั้งแรกรอไม่กี่วินาที  '
             'ลองภาพ ai_tray/* ทุกประเภท',
        method='ai', image='ai_tray/missing_1.png'),

    # ----- Workshop 8: ส่วนประกอบของสินค้า Rack server (notebook 08) -----
    'Workshop 8.1: Rack server แผนที่โซน': dict(
        desc='แต่ละโซนมีกฎของตัวเอง ไดรฟ์ ไฟสถานะ USB ฉลาก  ลอง rack_02–04  '
             'และเกณฑ์ไดรฟ์ 90%',
        method='rack_zones', image='rack/rack_01_good.png'),
    'Workshop 8.2: Rack server หาส่วนประกอบอัตโนมัติ': dict(
        desc='หาก้อนทั้งหมดแล้วจำแนกด้วยขนาด/รูปร่าง ไม่ต้องกำหนดกรอบ  ต้องมีไดรฟ์ 8 พอร์ต 2 ไฟ 2 ฉลาก 1',
        method='rack_parts', image='rack/rack_01_good.png'),

    # ----- Workshop 9: Challenge (notebook 09) -----
    'Workshop 9 ข้อ 1: นับสีน้ำเงิน': dict(
        desc='shapes.png ช่วง H น้ำเงิน 95–125  ต้องได้ 3 ชิ้น',
        image='shapes.png', h_range=(95, 125), expect=3),
    'Workshop 9 ข้อ 2–3: สีส้มปน / เศษชิ้นแตก': dict(
        desc='tray_05 มีสีส้มปน ต้อง NG  tray_03 มีเศษแตก 2 ชิ้น ต้อง NG  tray_01 ต้อง OK',
        image='line/tray_05.png', chk_chip=True, chk_foreign=True),
    'Workshop 9 ข้อ 4: ฉลากต้องมีเลข LOT': dict(
        desc='เพิ่มกฎ LOT ตามด้วยตัวเลข 6 หลัก  label_01 ต้อง OK  label_05 (รอยเปื้อนทับ LOT) ต้อง NG',
        method='label', image='industrial/label_01_good.png', ref_image='industrial/label_golden.png',
        need_lot=True),
    'Workshop 9 ข้อ 5: ไฟสถานะไดรฟ์': dict(
        desc='ตรวจไฟเขียวใต้ไดรฟ์ทั้ง 8 ตัว  rack_05 ต้องเจอไดรฟ์ 3 ไฟดับ',
        method='rack_zones', image='rack/rack_05_drive_led_off.png', chk_drive_led=True),
    'Workshop 9 ข้อ 6: เหรียญสอบเทียบ 25 มม.': dict(
        desc='เปลี่ยนขนาดเหรียญเป็น 25 มม.  ชิ้นงานต้องกว้าง 31.9 มม.',
        method='measure', image='industrial/measure_01.png', coin_mm=25.0, spec_w=31.9, spec_h=20.0),
    'Workshop 9 ข้อ 7: ถาดเลื่อน จัดตำแหน่งก่อนเทียบ': dict(
        desc='ถาดเลื่อน 14 px ขวา 9 px ขึ้น  หาระยะเลื่อนด้วย Template matching แล้วต้องเหลือความต่าง 1 จุด (สกรู)  '
             'ลองเปลี่ยนเป็น "ไม่จัดตำแหน่ง"',
        method='reference', image='industrial/tray_06_shifted_screw.png',
        ref_image='industrial/tray_reference.png', align='shift', diff_threshold=20, min_area=120),
    'Workshop 9 ข้อ 8: AI 4 ประเภท (+ชิ้นเกิน)': dict(
        desc='เพิ่มประเภท extra (ชิ้นเกินกลางถาด) แล้วสอนใหม่  ความแม่นต้อง ≥ 85%',
        method='ai', image='ai_tray/extra_1.png', ai_extra=True),

    # ----- Workshop 10: โครงการแรกของฉัน รูปจากมือถือ (notebook 10) -----
    'Workshop 10: รูปมือถือ เทียบภาพอ้างอิง': dict(
        desc='ใช้รูปของคุณ: แหล่งภาพ "อัปโหลดภาพ" แล้วอัปโหลดรูปชิ้นงานดี + รูปที่จะตรวจ '
             '(หรือใส่รูปใน images/my_project/ ตั้งชื่อรูปดีว่า good.jpg)  ไม่มีรูปใช้ phone_demo/* ก่อน',
        method='reference', image=W10_IMG, ref_image=W10_REF, align='ecc', diff_threshold=25, min_area=150),
    'Workshop 10: รูปมือถือ แผนที่โซน': dict(
        desc='ตรวจทีละโซนว่าเหมือนรูปชิ้นงานดีไหม  ดูพิกัดจากภาพเส้นตาราง แล้วแก้โซนในแท็บปรับเงื่อนไข',
        method='zones', image=W10_IMG, ref_image=W10_REF, align='ecc'),
    'Workshop 10: รูปมือถือ นับชิ้นงาน': dict(
        desc='นับชิ้นงานที่ตัดกับพื้นหลัง ต้องมี 6 ชิ้น  ชิ้นงานเข้มกว่าพื้นให้ติ๊ก "กลับสี"',
        method='count', image=W10_IMG, ref_image=W10_REF, align='ecc', expect=6, min_area=500),
}
SOURCES = {'ภาพตัวอย่าง': 'file', 'อัปโหลดภาพ': 'upload', 'กล้อง Basler จำลอง': 'emu',
           'กล้อง Basler จริง': 'basler', 'Webcam': 'webcam'}
DEFAULTS = dict(
    method='color', source='ภาพตัวอย่าง', image='shapes.png', ref_image='industrial/tray_reference.png',
    # นับตามสี (Workshop 2, 5, 9)  min_area / expect ใช้ร่วมกับวิธีเทียบภาพอ้างอิงและนับชิ้นงาน
    h_range=(0, 10), s_min=50, v_min=50, min_area=1000, expect=5, chk_chip=False, chk_foreign=False,
    # กล้อง + จำลองสภาพโรงงาน (Workshop 5)
    sim=False, light=1.0, speed=0.0, exposure=5.0, gain=0.0, enclosure=False, auto_exposure=False,
    cost_fr=20, cost_esc=2000,
    # จัดตำแหน่ง + เทียบภาพอ้างอิง (Workshop 6.1, 9.7, 10)
    align='none', diff_threshold=20,
    # ฉลาก (Workshop 6.2, 9.4)
    product='PRODUCT A', match_min=0.6, diff_max=0.01, name_min=0.85, need_exp=True, need_lot=False,
    # แหวน / น็อต (Workshop 6.3)
    peak_frac=0.5, hole_min=280, expect_washer=7, expect_nut=5,
    # วัดขนาด (Workshop 6.4, 9.6)
    coin_mm=20.0, spec_w=25.5, spec_h=16.0, tol_mm=0.3,
    # AI (Workshop 7, 9.8)
    ai_extra=False,
    # Rack server (Workshop 8, 9.5)
    drive_min=0.6, chk_drive_led=False,
    # แผนที่โซน / นับชิ้นงาน รูปมือถือ (Workshop 10)
    zones_text=W10_ZONES, zone_tol=25, invert=False,
)
PROJECT = dict(p_name='ตรวจถาดบรรจุชิ้นส่วน', p_product='ถาดใส่แหวนโลหะ 6 ช่อง',
               p_defect='มีสิ่งแปลกปลอม หรือแหวนหาย', p_now='พนักงานดูด้วยตา 1 คน ต่อกะ',
               p_goal='ไม่ให้ของเสียหลุดถึงลูกค้า')

# คงค่าทุกช่องไว้ แม้ช่องนั้นไม่ได้แสดงในรอบนี้ (เช่น เปลี่ยนวิธีตรวจแล้วกลับมา)
for k, v in {**DEFAULTS, **PROJECT}.items():
    st.session_state[k] = st.session_state.get(k, v)
for k, v in (('log', []), ('last', None), ('shift', None), ('recipe', list(RECIPES)[0])):
    st.session_state.setdefault(k, v)


def apply_recipe():
    r = RECIPES[st.session_state.recipe]
    for k, v in DEFAULTS.items():
        st.session_state[k] = r.get(k, v)
    st.session_state.last = None
    st.session_state.shift = None


# ---------------- กล้อง: เปิดครั้งเดียวแล้วค้างไว้ ----------------
@st.cache_resource(max_entries=1)
def open_camera(kind, image, keep=False):
    if kind == 'webcam':
        cam = WebcamSource()
    else:
        cam = BaslerSource(emulate=(kind == 'emu'), keep_settings=keep, image_path=img_path(image))
    cam.lock = threading.Lock()
    cam.settings = None
    return cam


# ---------------- ภาพตัวอย่าง (ไฟล์ใน images/ + ภาพที่สร้างในโปรแกรม) ----------------
def make_tray(kind, rng):
    """ถาดจำลอง 6 ช่องสำหรับสอน AI (notebook 07)  ถาดขยับ แสงเปลี่ยน และมีจุดรบกวนทุกภาพ"""
    img = np.full((240, 320, 3), (70, 72, 75), np.uint8)
    dx, dy = rng.integers(-10, 11, 2)                              # ถาดขยับ (จิ๊กหลวม)
    centers = [(60 + dx, 60 + dy), (160 + dx, 60 + dy), (260 + dx, 60 + dy),
               (60 + dx, 170 + dy), (160 + dx, 170 + dy), (260 + dx, 170 + dy)]
    if kind == 'extra':                                             # ชิ้นเกินกลางถาด (notebook 09 ข้อ 8)
        centers.append((160 + dx, 115 + dy))
    skip = rng.integers(0, 6) if kind == 'missing' else -1
    for i, (x, y) in enumerate(centers):
        if i == skip:
            continue
        cv2.circle(img, (int(x), int(y)), 26, (190, 190, 195), -1)
        cv2.circle(img, (int(x), int(y)), 8, (80, 82, 85), -1)
    if kind == 'foreign':                                           # สิ่งแปลกปลอม สุ่มชนิด ตำแหน่ง ขนาด
        fx, fy = int(rng.integers(30, 290)), int(rng.integers(30, 210))
        f = rng.integers(0, 3)
        if f == 0:                                                   # สกรู
            cv2.rectangle(img, (fx, fy), (fx + int(rng.integers(40, 70)), fy + 12), (30, 30, 35), -1)
        elif f == 1:                                                 # เศษพลาสติกสี
            cv2.circle(img, (fx, fy), int(rng.integers(14, 22)), (60, 200, 230), -1)
        else:                                                        # เส้นด้าย
            pts = np.array([[fx - 30, fy], [fx, fy - 12], [fx + 30, fy + 10], [fx + 60, fy - 5]], np.int32)
            cv2.polylines(img, [pts], False, (20, 20, 20), 3)
    light = rng.uniform(0.6, 1.3)                                   # แสงเปลี่ยน
    return np.clip(img.astype(np.float32) * light + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)


def make_demo_photos():
    """รูปจำลองการถ่ายด้วยมือถือ (notebook 10): เอียง เลื่อน ย่อขยาย และแสงต่างกันนิดหน่อย"""
    rng = np.random.default_rng(3)
    names = ['tray_reference.png', 'tray_01_clean.png', 'tray_02_screw.png',
             'tray_03_hair.png', 'tray_04_missing.png', 'tray_05_plastic.png']
    photos = {}
    for k, n in enumerate(names):
        img = cv2.imread(img_path('industrial/' + n))
        if img is None:
            continue
        if k == 0:
            photos['good.jpg'] = img
            continue
        h, w = img.shape[:2]
        ang, sc = rng.uniform(-4, 4), rng.uniform(0.95, 1.05)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
        M[:, 2] += rng.uniform(-15, 15, 2)
        warped = cv2.warpAffine(img, M, (w, h), borderValue=(70, 72, 75))
        warped = np.clip(warped.astype(np.float32) * rng.uniform(0.9, 1.1), 0, 255).astype(np.uint8)
        photos['photo_' + n.split('_', 1)[1].replace('.png', '.jpg')] = warped
    return photos


@st.cache_resource
def virtual_images():
    imgs = {}
    rng = np.random.default_rng(2024)
    for kind in ('ok', 'missing', 'foreign', 'extra'):
        for k in range(1, 4):
            imgs[f'ai_tray/{kind}_{k}.png'] = make_tray(kind, rng)
    for name, img in make_demo_photos().items():
        imgs['phone_demo/' + name] = img
    return imgs


def load_image(name):
    img = virtual_images().get(name)
    return cv2.imread(img_path(name)) if img is None else img.copy()


def sample_images(virtual=True):
    names = ['shapes.png']
    for folder in ('line', 'factory', 'industrial', 'rack', 'my_project'):
        d = img_path(folder)
        if os.path.isdir(d):
            names += [folder + '/' + f for f in sorted(os.listdir(d)) if f.lower().endswith(IMG_EXT)]
    if virtual:                     # กล้องจำลองต้องใช้ไฟล์จริง ภาพที่สร้างในโปรแกรมจึงใช้ได้เฉพาะแหล่ง "ภาพตัวอย่าง"
        names += list(virtual_images())
    return names


# ---------------- วิธีตรวจ: ทุกวิธีคืน Result แบบเดียวกัน ----------------
Result = namedtuple('Result', 'verdict reason metric_label metric out aux aux_caption')


def finish(reasons, metric_label, metric, out, aux=None, aux_caption=''):
    reason = '  |  '.join(reasons) if reasons else OK_REASON
    return Result('NG' if reasons else 'OK', reason, metric_label, metric, out, aux, aux_caption)


def line_width(img):
    return max(2, img.shape[1] // 200)


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def count_color(hsv, lo, hi, s_min, v_min):
    mask = cv2.inRange(hsv, np.array([lo, s_min, v_min]), np.array([hi, 255, 255]))
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return mask, cs


def inspect_color(img, s, ref=None, valid=None):
    """Workshop 2, 5, 9 ข้อ 1–3: นับชิ้นตามช่วงสี + กฎเศษแตก + สิ่งแปลกปลอมสีส้ม"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lo, hi = s['h_range']
    mask, cs = count_color(hsv, lo, hi, s['s_min'], s['v_min'])
    out = img.copy()
    thick = line_width(img)
    big = [c for c in cs if cv2.contourArea(c) > s['min_area']]
    chips = [c for c in cs if s['min_area'] * 0.05 < cv2.contourArea(c) <= s['min_area']]
    for c in big:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(out, (x, y), (x + w, y + h), (40, 180, 40), thick)
    foreign = []
    if s['chk_foreign']:
        _, cs_o = count_color(hsv, 12, 22, s['s_min'], s['v_min'])
        foreign = [c for c in cs_o if cv2.contourArea(c) > s['min_area']]
    if s['chk_chip']:
        for c in chips:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), thick)
    for c in foreign:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), thick)
    n = len(big)
    if n < s['expect']:
        reason = 'MISSING  ขาด ' + str(s['expect'] - n) + ' ชิ้น'
    elif n > s['expect']:
        reason = 'EXTRA  เกิน ' + str(n - s['expect']) + ' ชิ้น'
    elif s['chk_chip'] and chips:
        reason = 'CHIP  มีเศษชิ้นแตก ' + str(len(chips))
    elif foreign:
        reason = 'FOREIGN  มีสิ่งแปลกปลอม ' + str(len(foreign))
    else:
        reason = OK_REASON
    verdict = 'OK' if reason == OK_REASON else 'NG'
    return Result(verdict, reason, 'พบ / ต้องมี', f"{n} / {s['expect']}", out, mask, 'Mask (สีที่เลือก)')


def width_to(img, w):
    return cv2.resize(img, (w, int(img.shape[0] * w / img.shape[1])))


def align_to_ref(img, ref, mode):
    """จัดภาพให้ทับภาพอ้างอิง  คืน (ภาพที่จัดแล้ว, ภาพอ้างอิง, mask ส่วนที่มีข้อมูล, ความเหมือน, ข้อความ)"""
    if mode == 'ecc':
        # Workshop 10: รูปมือถือเอียง/เลื่อนไม่เท่ากัน ย่อให้กว้าง 800 แล้วหมุน เลื่อน ย่อขยาย จนทับรูปอ้างอิงพอดีที่สุด
        img, ref = width_to(img, 800), width_to(ref, 800)
        g0 = cv2.GaussianBlur(to_gray(ref), (5, 5), 0).astype(np.float32)
        g1 = cv2.GaussianBlur(to_gray(img), (5, 5), 0).astype(np.float32)
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            score, warp = cv2.findTransformECC(g0, g1, warp, cv2.MOTION_AFFINE,
                                               (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5), None, 5)
        except cv2.error:
            return img, ref, None, 0.0, 'ECC จัดตำแหน่งไม่สำเร็จ'
        h, w = ref.shape[:2]
        flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        out = cv2.warpAffine(img, warp, (w, h), flags=flags, borderMode=cv2.BORDER_REPLICATE)
        valid = cv2.warpAffine(np.full(img.shape[:2], 255, np.uint8), warp, (w, h), flags=flags, borderValue=0)
        valid = cv2.erode(valid, np.ones((15, 15), np.uint8))          # ตัดขอบที่เกิดจากการหมุนทิ้ง
        return out, ref, valid, float(score), f'ECC ความเหมือน {score:.2f}'
    # Workshop 9 ข้อ 7: ถาดเลื่อน  หาว่าส่วนกลางของภาพอ้างอิงไปอยู่ตรงไหนในภาพใหม่ แล้วเลื่อนภาพกลับ
    h, w = ref.shape[:2]
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h))
    y0, x0 = h // 5, w // 4
    template = to_gray(ref)[y0:h - y0, x0:w - x0]
    _, score, _, (x, y) = cv2.minMaxLoc(cv2.matchTemplate(to_gray(img), template, cv2.TM_CCOEFF_NORMED))
    dx, dy = x - x0, y - y0
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    valid = cv2.warpAffine(np.full((h, w), 255, np.uint8), M, (w, h), borderValue=0)
    return out, ref, valid, float(score), f'เลื่อน dx={dx} dy={dy}'


def inspect_reference(img, s, ref, valid=None):
    """Workshop 6.1, 9 ข้อ 7, 10: ส่วนที่ต่างจากภาพอ้างอิง = สิ่งที่ไม่ควรมี หรือสิ่งที่หายไป"""
    phone = s['align'] == 'ecc'
    k = 7 if phone else 5           # รูปมือถือผ่านการหมุน/ย่อขยาย มีจุดรบกวนมากกว่า จึงเบลอและตัดจุดเล็กมากขึ้น
    g1 = cv2.GaussianBlur(to_gray(img), (k, k), 0)
    g0 = cv2.GaussianBlur(to_gray(ref), (k, k), 0)
    _, mask = cv2.threshold(cv2.absdiff(g1, g0), s['diff_threshold'], 255, cv2.THRESH_BINARY)
    if valid is not None:
        mask = cv2.bitwise_and(mask, valid)                         # ไม่นับขอบภาพที่ไม่มีข้อมูลหลังจัดตำแหน่ง
    if phone:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((k, k), np.uint8))              # รวมพิกเซลที่ต่างให้เป็นชิ้นเดียวกัน
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out, found = img.copy(), {}
    for c in cs:
        if cv2.contourArea(c) < s['min_area']:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if phone:
            code = 'DIFF'
        else:   # ภาพอ้างอิงตรงนี้เคยมีชิ้นงาน (สว่าง) = หายไป  ถ้าเดิมเป็นพื้นถาด = มีสิ่งแปลกปลอมเพิ่มมา
            code = 'MISSING' if g0[y:y + h, x:x + w].mean() > 120 else 'FOREIGN'
        found[code] = found.get(code, 0) + 1
        color = (0, 165, 255) if code == 'MISSING' else (0, 0, 255)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, line_width(img))
    text = {'FOREIGN': 'สิ่งแปลกปลอม', 'MISSING': 'ชิ้นหาย', 'DIFF': 'ต่างจากภาพอ้างอิง'}
    reasons = [f'{code}  {text[code]} {n} จุด' for code, n in found.items()]
    return finish(reasons, 'จุดที่ต่าง', str(sum(found.values())), out, mask, 'ความต่างจากภาพอ้างอิง')


@st.cache_resource
def ocr_engine():
    """pytesseract ที่พร้อมใช้ หรือ None ถ้ายังไม่ได้ติดตั้ง"""
    try:
        import pytesseract
    except ImportError:
        return None
    win = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if shutil.which('tesseract') is None and os.path.exists(win):
        pytesseract.pytesseract.tesseract_cmd = win
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        return None
    return pytesseract


def inspect_label(img, s, ref, valid=None):
    """Workshop 6.2, 9 ข้อ 4: หาฉลาก → เทียบการพิมพ์กับต้นแบบ → อ่านข้อความด้วย OCR"""
    gi, gg = to_gray(img), to_gray(ref)
    h, w = gg.shape
    out = img.copy()
    if gi.shape[0] < h or gi.shape[1] < w:
        return finish(['NO_LABEL  ภาพเล็กกว่าฉลากต้นแบบ'], 'Match score', '-', out, ref, 'ฉลากต้นแบบ')
    # 1) หาตำแหน่งฉลาก
    _, score, _, (x, y) = cv2.minMaxLoc(cv2.matchTemplate(gi, gg, cv2.TM_CCOEFF_NORMED))
    if score < s['match_min']:
        put_label(out, f'label not found ({score:.2f})', org=(20, 40), color=(0, 0, 255))
        return finish([f'NO_LABEL  ไม่พบฉลาก / ฉลากเอียง-เลื่อน (match {score:.2f})'],
                      'Match score', f'{score:.2f}', out, ref, 'ฉลากต้นแบบ')
    reasons = []
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 0), 3)
    crop = gi[y:y + h, x:x + w]
    # 2) เทียบการพิมพ์กับต้นแบบ
    _, m = cv2.threshold(cv2.absdiff(crop, gg), 60, 255, cv2.THRESH_BINARY)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ratio = m.mean() / 255
    if ratio > s['diff_max']:
        reasons.append(f'PRINT  การพิมพ์ต่างจากต้นแบบ {ratio * 100:.1f}% ของพื้นที่')
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            if cv2.contourArea(c) > 80:
                bx, by, bw, bh = cv2.boundingRect(c)
                cv2.rectangle(out, (x + bx, y + by), (x + bx + bw, y + by + bh), (0, 0, 255), 2)
    # 3) อ่านข้อความ
    ocr, text = ocr_engine(), ''
    if ocr is None:
        reasons.append('OCR  อ่านข้อความไม่ได้ ติดตั้ง Tesseract ก่อน')
    else:
        text = ocr.image_to_string(crop, config='--psm 6').upper()
        first = text.split('\n')[0]
        want = s['product'].strip().upper()
        n = len(want)
        # OCR อ่านตัวอักษรเพี้ยนได้บ้าง (เช่น D เป็น V) จึงใช้ "ความคล้าย" แทนการเทียบตรงๆ
        sim = max((difflib.SequenceMatcher(None, want, first[i:i + n]).ratio()
                   for i in range(max(1, len(first) - n + 1))), default=0)
        if sim < s['name_min']:
            reasons.append(f"NAME  ชื่อสินค้าไม่ตรง (อ่านได้ '{first.strip()}' คล้าย {sim:.0%})")
        if s['need_exp'] and 'EXP' not in text:
            reasons.append('NO_EXP  ไม่มีวันหมดอายุ')
        if s['need_lot'] and not re.search(r'LOT\s*\d{6}\b', text):
            reasons.append('NO_LOT  ไม่มีเลข LOT 6 หลัก')
    lines = [t.strip() for t in text.split('\n') if t.strip()]
    caption = 'ส่วนที่พิมพ์ต่างจากต้นแบบ' + ('   OCR อ่านได้: ' + ' / '.join(lines) if lines else '')
    return finish(reasons, 'Match score', f'{score:.2f}', out, m, caption)


def inspect_parts(img, s, ref=None, valid=None):
    """Workshop 6.3: แยกโลหะด้วย Otsu  แยกชิ้นที่ติดกันด้วย Watershed  แยกชนิดด้วยขนาดรู"""
    gray = to_gray(img)
    _, bw = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cs, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(bw)
    cv2.drawContours(filled, cs, -1, 255, -1)                                   # ชิ้นงานแบบอุดรู
    holes = cv2.bitwise_and(filled, cv2.bitwise_not(bw))                        # เฉพาะรูตรงกลาง
    # Distance transform หา "ใจกลาง" ของแต่ละชิ้น แล้ว Watershed แบ่งเขตชิ้นที่ติดกัน
    dist = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    _, peaks = cv2.threshold(dist, s['peak_frac'] * dist.max(), 255, 0)
    peaks = peaks.astype(np.uint8)
    sure_bg = cv2.dilate(filled, np.ones((9, 9), np.uint8))
    n_markers, markers = cv2.connectedComponents(peaks)
    markers = markers + 1
    markers[(sure_bg == 255) & (peaks == 0)] = 0
    markers = cv2.watershed(img, markers)

    out, counts = img.copy(), {'washer': 0, 'nut': 0}
    out[markers == -1] = (0, 0, 255)                                            # เส้นแบ่งของ Watershed
    for label_id in range(2, n_markers + 1):
        region = markers == label_id
        if region.sum() < 300:
            continue
        hole_area = int((holes[region] > 0).sum())
        kind = 'washer' if hole_area > s['hole_min'] else 'nut'
        counts[kind] += 1
        ys, xs = np.where(region)
        color = (0, 255, 0) if kind == 'washer' else (0, 200, 255)
        put_label(out, f'{kind[0].upper()} hole {hole_area}', org=(int(xs.min()), int(ys.min()) - 4),
                  scale=0.45, color=color)
    reasons = []
    if counts['washer'] != s['expect_washer'] or counts['nut'] != s['expect_nut']:
        reasons.append(f"COUNT  แหวน {counts['washer']}/{s['expect_washer']}  น็อต {counts['nut']}/{s['expect_nut']}")
    return finish(reasons, 'แหวน / น็อต', f"{counts['washer']} / {counts['nut']}", out, bw,
                  'หลัง Threshold (Otsu)   เส้นแดงในผลตรวจ = รอยตัดของ Watershed')


def inspect_measure(img, s, ref=None, valid=None):
    """Workshop 6.4, 9 ข้อ 6: สอบเทียบด้วยเหรียญที่รู้ขนาด → px/mm → วัดชิ้นงาน"""
    _, bw = cv2.threshold(to_gray(img), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)   # วัตถุมืดบนพื้นสว่าง
    cs, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = sorted((c for c in cs if cv2.contourArea(c) > 100), key=lambda c: cv2.boundingRect(c)[0])  # ซ้ายไปขวา: เหรียญก่อน
    out = img.copy()
    if len(cs) < 2:
        return finish(['NO_PART  ไม่พบเหรียญสอบเทียบหรือชิ้นงาน'], 'ขนาด (มม.)', '-', out, bw, 'หลัง Threshold (Otsu)')
    (cx, cy), r = cv2.minEnclosingCircle(cs[0])
    px_per_mm = 2 * r / s['coin_mm']
    cv2.circle(out, (int(cx), int(cy)), int(r), (0, 200, 0), 2)
    reasons, sizes = [], []
    for c in cs[1:]:
        rect = cv2.minAreaRect(c)                                    # กรอบที่หมุนตามชิ้นงาน
        W, H = sorted([rect[1][0] / px_per_mm, rect[1][1] / px_per_mm], reverse=True)
        ok = abs(W - s['spec_w']) <= s['tol_mm'] and abs(H - s['spec_h']) <= s['tol_mm']
        box = cv2.boxPoints(rect).astype(int)
        cv2.drawContours(out, [box], 0, (0, 200, 0) if ok else (0, 0, 255), 2)
        put_label(out, f'{W:.1f} x {H:.1f} mm', org=(int(box[:, 0].min()), int(box[:, 1].min()) - 10), scale=0.7)
        sizes.append(f'{W:.1f} x {H:.1f}')
        if not ok:
            reasons.append(f"SIZE  {W:.1f} x {H:.1f} มม. (สเปก {s['spec_w']:.1f} x {s['spec_h']:.1f} ± {s['tol_mm']:.1f})")
    caption = (f"หลัง Threshold (Otsu)   เหรียญ {s['coin_mm']:.1f} มม. = {2 * r:.0f} px  →  {px_per_mm:.2f} px/mm"
               f"  (1 px = {1 / px_per_mm:.2f} มม.)")
    return finish(reasons, 'ขนาด (มม.)', ', '.join(sizes), out, bw, caption)


# ----- AI (Workshop 7, 9 ข้อ 8) -----
AI_CLASSES = ('ok', 'missing', 'foreign')
_HOG = cv2.HOGDescriptor((64, 48), (16, 16), (8, 8), (8, 8), 9) if hasattr(cv2, 'HOGDescriptor') else None


def hog_features(gray):
    """HOG: สรุปว่าขอบในภาพหันทางไหน  OpenCV 5 ไม่มี HOGDescriptor จึงคำนวณแบบเดียวกันด้วย numpy แทน"""
    if _HOG is not None:
        return _HOG.compute(gray).flatten()
    g = gray.astype(np.float32)
    mag, ang = cv2.cartToPolar(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=1), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=1),
                               angleInDegrees=True)
    bins = (ang % 180 // 20).astype(np.int32) % 9                     # 9 ทิศทาง 0–180 องศา
    ch, cw = g.shape[0] // 8, g.shape[1] // 8                         # cell 8 x 8 พิกเซล
    cells = np.stack([np.where(bins == b, mag, 0)[:ch * 8, :cw * 8].reshape(ch, 8, cw, 8).sum(axis=(1, 3))
                      for b in range(9)], axis=2)
    blocks = np.concatenate([cells[:-1, :-1], cells[1:, :-1], cells[:-1, 1:], cells[1:, 1:]], axis=2)  # block 2 x 2 cell
    blocks /= np.linalg.norm(blocks, axis=2, keepdims=True) + 1e-6
    return blocks.ravel()


def ai_features(img):
    small_img = cv2.resize(img, (64, 48))
    gray = to_gray(small_img)
    thumb = cv2.resize(gray, (32, 24)).astype(np.float32)
    f_thumb = (thumb / (thumb.mean() + 1e-6)).flatten()                # ภาพย่อ ตัดผลของแสง
    hist = cv2.calcHist([cv2.cvtColor(small_img, cv2.COLOR_BGR2HSV)], [0, 1], None, [8, 4], [0, 180, 0, 256]).flatten()
    f_color = 10 * hist / (hist.sum() + 1e-6)                          # สัดส่วนสี ไม่ขึ้นกับความสว่าง
    return np.concatenate([hog_features(gray), f_thumb, f_color])


@st.cache_resource(show_spinner='กำลังสอนโมเดล AI จากถาดจำลอง ...')
def train_ai(classes):
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ImportError:
        raise RuntimeError('ต้องติดตั้ง scikit-learn ก่อน:  pip install scikit-learn')
    rng = np.random.default_rng(0)
    X, y = [], []
    for ci, kind in enumerate(classes):
        for _ in range(100):
            X.append(ai_features(make_tray(kind, rng)))
            y.append(ci)
    X_tr, X_te, y_tr, y_te = train_test_split(np.array(X), np.array(y), test_size=0.3, random_state=0, stratify=y)
    model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=10)).fit(X_tr, y_tr)
    return model, float((model.predict(X_te) == y_te).mean())


def inspect_ai(img, s, ref=None, valid=None):
    """Workshop 7, 9 ข้อ 8: ภาพ → Features → โมเดล SVM ทายประเภทถาด"""
    classes = AI_CLASSES + (('extra',) if s['ai_extra'] else ())
    model, acc = train_ai(classes)
    view = cv2.resize(img, (320, 240))
    kind = classes[int(model.predict([ai_features(view)])[0])]
    out = img.copy()
    put_label(out, 'AI: ' + kind, color=(0, 255, 0) if kind == 'ok' else (0, 0, 255))
    reasons = [] if kind == 'ok' else [f'{kind.upper()}  AI ทายว่า {kind}']
    return finish(reasons, 'AI ทาย', kind, out, view,
                  f'ภาพที่โมเดลเห็น 320 x 240   {len(classes)} ประเภท  ความแม่นกับภาพสอบ {acc:.0%}')


# ----- Rack server (Workshop 8, 9 ข้อ 5) -----
RACK_ZONES = {f'bay{i + 1}': ('drive', (50 + i * 88, 80, 80, 170)) for i in range(8)}     # (x, y, w, h)
RACK_ZONES.update(led_power=('led', (48, 28, 24, 24)), led_status=('led', (88, 28, 24, 24)),
                  usb1=('port', (640, 28, 36, 24)), usb2=('port', (690, 28, 36, 24)),
                  label=('label', (300, 22, 260, 36)))
RACK_EXPECT = {'drive': 8, 'port': 2, 'led': 2, 'label': 1}
RACK_SIZE = (800, 300)                    # โซนกำหนดจากภาพอ้างอิงขนาดนี้ (กว้าง x สูง)


def check_rack_zone(kind, roi, drive_min):
    gray, hsv = to_gray(roi), cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    if kind == 'drive':                   # ไดรฟ์ = พื้นที่สว่าง (>90) มากพอ
        r = (gray > 90).mean()
        return r > drive_min, f'ไม่มีไดรฟ์ (สว่าง {r:.0%})'
    if kind == 'led':                     # ไฟ = สีเขียว H 40-80  ส้ม H 10-25 = fault
        green = cv2.inRange(hsv, (40, 80, 80), (80, 255, 255)).mean() / 255
        orange = cv2.inRange(hsv, (10, 80, 80), (25, 255, 255)).mean() / 255
        if green > 0.3:
            return True, 'สีเขียว'
        return False, 'สีส้ม = fault' if orange > 0.3 else 'ไฟดับ'
    if kind == 'port':                    # พอร์ต = มีขอบสว่างล้อมช่องมืด
        r = (gray > 150).mean()
        return r > 0.25, f'ไม่มีพอร์ต (สว่าง {r:.0%})'
    r = (gray > 200).mean()               # ฉลาก = พื้นที่ขาว > 50%
    return r > 0.5, f'ไม่มีฉลาก (สีขาว {r:.0%})'


def inspect_rack_zones(img, s, ref=None, valid=None):
    """Workshop 8.1, 9 ข้อ 5: แต่ละโซนมีกฎตรวจของตัวเอง และบอกได้ว่าส่วนไหนผิด"""
    if img.shape[1::-1] != RACK_SIZE:
        img = cv2.resize(img, RACK_SIZE)
    out, reasons, passed, no_drive = img.copy(), [], 0, set()
    for name, (kind, (x, y, w, h)) in RACK_ZONES.items():
        ok, info = check_rack_zone(kind, img[y:y + h, x:x + w], s['drive_min'])
        passed += int(ok)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 0) if ok else (0, 0, 255), 2)
        if not ok:
            put_label(out, name, org=(x, y - 4), scale=0.5, color=(0, 0, 255))
            reasons.append(f'{kind.upper()}  {name} {info}')
            if kind == 'drive':
                no_drive.add(name)
    total = len(RACK_ZONES)
    if s['chk_drive_led']:
        # ไฟสถานะสีเขียวใต้ไดรฟ์แต่ละตัว แถบ y 233–245  (ช่องที่ไม่มีไดรฟ์รายงานไปแล้ว จึงข้าม)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for i in range(8):
            name, x = f'bay{i + 1}', 50 + i * 88
            if name in no_drive:
                continue
            total += 1
            ok = cv2.inRange(hsv[233:245, x + 8:x + 72], (40, 80, 80), (80, 255, 255)).mean() / 255 >= 0.3
            passed += int(ok)
            cv2.rectangle(out, (x + 8, 233), (x + 72, 245), (0, 200, 0) if ok else (0, 0, 255), 2)
            if not ok:
                reasons.append(f'DRIVE_LED  {name} ไฟสถานะไดรฟ์ดับ')
    return finish(reasons, 'โซนที่ผ่าน', f'{passed} / {total}', out)


def inspect_rack_parts(img, s, ref=None, valid=None):
    """Workshop 8.2: หาก้อนทั้งหมดในภาพ แล้วจำแนกด้วยขนาดและรูปร่าง ไม่ต้องรู้ตำแหน่งล่วงหน้า"""
    gray = to_gray(img)
    _, bw = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)                    # ส่วนประกอบสว่างกว่าตัวเครื่อง
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    colors = {'drive': (0, 200, 0), 'port': (255, 150, 0), 'led': (0, 255, 255), 'label': (255, 0, 200)}
    out, found = img.copy(), dict.fromkeys(RACK_EXPECT, 0)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < 150 or w > 700:                                               # ตัดจุดรบกวน และกรอบตัวเครื่อง
            continue
        aspect, mean = w / h, gray[y:y + h, x:x + w].mean()
        if area > 8000 and aspect < 0.7:
            kind = 'drive'
        elif mean > 180 and aspect > 4:
            kind = 'label'
        elif 500 < area < 1500 and aspect > 1.2:
            kind = 'port'
        elif aspect > 0.7 and area < 500:
            kind = 'led'
        else:
            continue
        found[kind] += 1
        cv2.rectangle(out, (x, y), (x + w, y + h), colors[kind], 2)
        put_label(out, kind, org=(x, y - 4), scale=0.45, color=colors[kind])
    missing = {k: RACK_EXPECT[k] - v for k, v in found.items() if v < RACK_EXPECT[k]}
    reasons = ['MISSING  ขาด ' + ', '.join(f'{k} {n}' for k, n in missing.items())] if missing else []
    return finish(reasons, 'ไดรฟ์ / พอร์ต / ไฟ / ฉลาก', ' / '.join(str(found[k]) for k in RACK_EXPECT), out, bw,
                  'หลัง Threshold (ส่วนประกอบสว่างกว่าตัวเครื่อง)')


# ----- รูปจากมือถือ (Workshop 10) -----
def parse_zones(text):
    zones = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        name, _, rest = line.partition(':')
        nums = [int(v) for v in re.findall(r'\d+', rest)]
        if not name.strip() or len(nums) != 4:
            raise ValueError('โซน "' + line.strip() + '" ต้องเขียนแบบ  ชื่อ: x, y, w, h')
        zones[name.strip()] = tuple(nums)
    if not zones:
        raise ValueError('ยังไม่ได้กำหนดโซนในแท็บปรับเงื่อนไขตรวจ')
    return zones


def grid_image(img, zones, step=50):
    g = img.copy()
    for x in range(0, g.shape[1], step):
        cv2.line(g, (x, 0), (x, g.shape[0]), (0, 255, 255), 1)
        put_label(g, str(x), org=(x + 2, 14), scale=0.4)
    for y in range(0, g.shape[0], step):
        cv2.line(g, (0, y), (g.shape[1], y), (0, 255, 255), 1)
        put_label(g, str(y), org=(2, y + 14), scale=0.4)
    for x, y, w, h in zones.values():
        cv2.rectangle(g, (x, y), (x + w, y + h), (255, 0, 255), 2)
    return g


def inspect_zones(img, s, ref, valid=None):
    """Workshop 10: แต่ละโซนต้องสว่างและมีลวดลายใกล้เคียงรูปชิ้นงานดี"""
    zones = parse_zones(s['zones_text'])
    g_img, g_ref = to_gray(img), to_gray(ref)
    H, W = g_ref.shape
    out, bad = img.copy(), []
    for name, (x, y, w, h) in zones.items():
        if w == 0 or h == 0 or x + w > W or y + h > H:
            raise ValueError(f'โซน {name} อยู่นอกภาพ (ภาพขนาด {W} x {H})')
        a, b = g_img[y:y + h, x:x + w].astype(float), g_ref[y:y + h, x:x + w].astype(float)
        same = abs(a.mean() - b.mean()) < s['zone_tol'] and abs(a.std() - b.std()) < s['zone_tol']
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 0) if same else (0, 0, 255), 3)
        if not same:
            bad.append(name)
            put_label(out, name, org=(x, y - 6), scale=0.6, color=(0, 0, 255))
    reasons = ['ZONE  ส่วนที่ผิด ' + ', '.join(bad)] if bad else []
    return finish(reasons, 'โซนที่ผ่าน', f'{len(zones) - len(bad)} / {len(zones)}', out,
                  grid_image(ref, zones), 'ภาพอ้างอิงพร้อมพิกัด (ใช้กำหนดโซน)')


def inspect_count(img, s, ref=None, valid=None):
    """Workshop 10: นับชิ้นงานที่ตัดกับพื้นหลังด้วย Otsu"""
    gray = cv2.GaussianBlur(to_gray(img), (5, 5), 0)
    flag = cv2.THRESH_BINARY_INV if s['invert'] else cv2.THRESH_BINARY
    _, bw = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)
    cs, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    parts = [c for c in cs if cv2.contourArea(c) > s['min_area']]
    out = img.copy()
    cv2.drawContours(out, parts, -1, (0, 200, 0), 3)
    n = len(parts)
    if n < s['expect']:
        reasons = [f"MISSING  ขาด {s['expect'] - n} ชิ้น"]
    elif n > s['expect']:
        reasons = [f"EXTRA  เกิน {n - s['expect']} ชิ้น"]
    else:
        reasons = []
    return finish(reasons, 'พบ / ต้องมี', f"{n} / {s['expect']}", out, bw, 'หลัง Threshold (Otsu)')


INSPECTORS = {'color': inspect_color, 'reference': inspect_reference, 'label': inspect_label,
              'parts': inspect_parts, 'measure': inspect_measure, 'ai': inspect_ai,
              'rack_zones': inspect_rack_zones, 'rack_parts': inspect_rack_parts,
              'zones': inspect_zones, 'count': inspect_count}


def inspect(img, s, ref=None):
    method = s['method']
    if needs_ref(s) and ref is None:
        raise ValueError('ยังไม่มีภาพอ้างอิง: เลือกหรืออัปโหลดภาพอ้างอิงทางซ้าย')
    valid, note = None, ''
    if method in ALIGNABLE and s['align'] != 'none':
        img, ref, valid, score, note = align_to_ref(img, ref, s['align'])
        if score < ALIGN_MIN:
            return Result('NG', f'ALIGN  จัดตำแหน่งไม่ได้ (ความเหมือน {score:.2f}) ถ่ายใหม่ให้มุมและระยะใกล้เคียงภาพอ้างอิง',
                          'ความเหมือนหลังจัดตำแหน่ง', f'{score:.2f}', img, ref, 'ภาพอ้างอิง')
    elif method in ('reference', 'zones') and img.shape != ref.shape:
        img = cv2.resize(img, (ref.shape[1], ref.shape[0]))
    r = INSPECTORS[method](img, s, ref, valid)
    return r._replace(aux_caption=(r.aux_caption + '   ' + note).strip()) if note else r


# ---------------- จำลองกะการผลิต 40 ถาด (Workshop 5) ----------------
def light_at(i):
    """แสงรอบไลน์ตามช่วงเวลาของกะ: ถาด 1–15 เช้า  16–28 บ่ายเมฆบัง  29–40 เย็นไฟดับหนึ่งดวง"""
    return 1.0 if i <= 15 else (0.5 if i <= 28 else 0.2)


@st.cache_data
def factory_truth():
    with open(img_path('factory_labels.csv'), encoding='utf-8') as f:
        return {int(r['part']): (r['truth'], r['defect']) for r in csv.DictReader(f)}


def simulate_station(frame, s, light, seed=0):
    """ภาพที่กล้องได้ตามสภาพหน้างาน  คืน (ภาพ, แสงที่ใช้, Exposure ที่ใช้)"""
    L = 1.0 if s['enclosure'] else light                        # กล่องครอบ + ไฟของตัวเอง = แสงคงที่
    exp = s['exposure']
    img = simulate_real(frame, exp, s['gain'], L, s['speed'], seed=seed)
    if s['auto_exposure']:                                      # ปรับ Exposure ตามความสว่างจริง
        exp = min(20.0, exp * 235.0 / max(float(np.median(img)), 1.0))
        img = simulate_real(frame, exp, s['gain'], L, s['speed'], seed=seed)
    return img, L, exp


def run_shift(s):
    rows = []
    t0 = time.perf_counter()
    for i, (truth, defect) in sorted(factory_truth().items()):
        frame = cv2.imread(img_path(f'factory/part_{i:03d}.png'))
        if frame is None:
            raise FileNotFoundError(f'ไม่พบภาพ images/factory/part_{i:03d}.png')
        img, L, exp = simulate_station(frame, s, light_at(i), seed=i)
        r = inspect_color(img, s)
        mistake = ('คัดทิ้งผิด' if truth == 'OK' and r.verdict == 'NG' else
                   'หลุดรอด' if truth == 'NG' and r.verdict == 'OK' else '')
        rows.append(dict(i=i, light=L, exp=round(exp, 1), truth=truth, defect=defect, verdict=r.verdict,
                         reason=r.reason, mistake=mistake, img=r.out if mistake else None))
    on = {True: 'เปิด', False: 'ปิด'}
    settings = (f"Exposure {s['exposure']} ms  Gain {s['gain']} dB  สายพาน {s['speed']}  "
                f"กล่องครอบ {on[s['enclosure']]}  Auto Exposure {on[s['auto_exposure']]}  "
                f"กฎเศษแตก {on[s['chk_chip']]}  กฎสิ่งแปลกปลอม {on[s['chk_foreign']]}")
    return dict(rows=rows, ms=(time.perf_counter() - t0) * 1000 / len(rows), settings=settings)


# ---------------- แสดงผล ----------------
def small(img, width=960):
    if img.shape[1] > width:
        img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])))
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def hue_bar(lo, hi, s_min, v_min):
    # แถบสี H 0-179  ส่วนที่เลือกสว่าง ส่วนอื่นจาง
    h = np.tile(np.arange(180, dtype=np.uint8), (24, 1))
    hsv = np.dstack([h, np.full_like(h, max(s_min, 160)), np.full_like(h, max(v_min, 200))])
    bar = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    sel = (h >= lo) & (h <= hi)
    bar[~sel] = (bar[~sel] * 0.25 + 190).astype(np.uint8)
    return cv2.resize(bar, (720, 28), interpolation=cv2.INTER_NEAREST)


def badge(verdict, text):
    color = '#1E7E34' if verdict == 'OK' else '#C62828'
    return ('<div style=' + Q + 'background:' + color + ';color:white;border-radius:14px;'
            'padding:10px 20px;text-align:center' + Q + '>'
            '<div style=' + Q + 'font-size:44px;font-weight:800;line-height:1.1' + Q + '>' + verdict + '</div>'
            '<div style=' + Q + 'font-size:18px;opacity:0.95' + Q + '>' + text + '</div></div>')


def show_shift(res, cost_fr, cost_esc):
    rows = res['rows']
    fr = [r for r in rows if r['mistake'] == 'คัดทิ้งผิด']
    esc = [r for r in rows if r['mistake'] == 'หลุดรอด']
    m = st.columns(4)
    m[0].metric('ตรวจถูก', f'{len(rows) - len(fr) - len(esc)} / {len(rows)}')
    m[1].metric('คัดทิ้งผิด (ทิ้งของดี)', len(fr))
    m[2].metric('หลุดรอด (ถึงลูกค้า)', len(esc))
    m[3].metric('ความเสียหายโดยประมาณ', f'{len(fr) * cost_fr + len(esc) * cost_esc:,} บาท')
    st.caption(f"เวลาตรวจเฉลี่ย {res['ms']:.0f} ms/ถาด   ค่าที่ใช้: {res['settings']}")

    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'img'} for r in rows])
    kinds, colors = ['คัดทิ้งผิด', 'หลุดรอด'], ['#F28E2B', '#D62728']
    line = alt.Chart(df).mark_line(color='gray').encode(
        x=alt.X('i:Q', title='ถาดที่'),
        y=alt.Y('light:Q', title='แสงรอบไลน์', scale=alt.Scale(domain=[0, 1.2])))
    points = alt.Chart(df[df['mistake'] != '']).mark_point(size=160, filled=True).encode(
        x='i:Q', y='light:Q',
        color=alt.Color('mistake:N', title='ตรวจผิด', scale=alt.Scale(domain=kinds, range=colors)),
        shape=alt.Shape('mistake:N', title='ตรวจผิด', scale=alt.Scale(domain=kinds, range=['circle', 'cross'])),
        tooltip=[alt.Tooltip('i:Q', title='ถาด'), alt.Tooltip('truth:N', title='จริง'),
                 alt.Tooltip('defect:N', title='ของเสีย'), alt.Tooltip('verdict:N', title='ระบบบอก')])
    st.altair_chart((line + points).properties(height=230, width='container'))

    bad = (esc + fr)[:3]                                        # หลุดรอดร้ายแรงที่สุด แสดงก่อน
    for col, r in zip(st.columns(3), bad):
        col.image(small(r['img'], 640), width='stretch',
                  caption=f"ถาด {r['i']}: จริง {r['truth']} ({r['defect']}) ระบบบอก {r['verdict']}")
    with st.expander('ผลทั้ง ' + str(len(rows)) + ' ถาด'):
        st.dataframe(df.rename(columns={'i': 'ถาด', 'light': 'แสง', 'exp': 'Exposure (ms)', 'truth': 'เฉลย',
                                        'defect': 'ของเสีย', 'verdict': 'ระบบบอก', 'reason': 'สาเหตุ',
                                        'mistake': 'ตรวจผิด'}),
                     width='stretch', hide_index=True)


# ================= แถบซ้าย =================
with st.sidebar:
    st.markdown('### 1. สูตรงาน')
    st.selectbox('เลือกสูตรตาม Workshop', list(RECIPES), key='recipe', on_change=apply_recipe,
                 label_visibility='collapsed')
    st.caption(RECIPES[st.session_state.recipe]['desc'])
    st.button('↺ คืนค่าตามสูตร', on_click=apply_recipe, width='stretch')
    st.selectbox('วิธีตรวจ', list(METHODS), key='method', format_func=METHODS.get)

    st.markdown('### 2. แหล่งภาพ')
    source_label = st.selectbox('ภาพมาจาก', list(SOURCES), key='source')
    source = SOURCES[source_label]
    is_camera = source in ('emu', 'basler', 'webcam')
    uploaded = ref_upload = None
    if source in ('file', 'emu'):
        imgs = sample_images(virtual=(source == 'file'))
        if st.session_state.image not in imgs:
            st.session_state.image = imgs[0]
        st.selectbox('ภาพชิ้นงาน', imgs, key='image')
    if source == 'upload':
        uploaded = st.file_uploader('เลือกไฟล์ภาพ', type=['png', 'jpg', 'jpeg', 'bmp'])
    if needs_ref(st.session_state):
        if source == 'upload':
            ref_upload = st.file_uploader('ภาพชิ้นงานดี (อ้างอิง / ต้นแบบ)', type=['png', 'jpg', 'jpeg', 'bmp'],
                                          key='ref_upload')
        refs = sample_images()
        if st.session_state.ref_image not in refs:
            st.session_state.ref_image = refs[0]
        st.selectbox('ภาพอ้างอิง / ต้นแบบ', refs, key='ref_image', disabled=ref_upload is not None,
                     help='ภาพชิ้นงานที่ถูกต้อง ใช้เทียบกับภาพที่ตรวจ  งานฉลากใช้ label_golden.png')

    keep = False
    if is_camera or source == 'file':
        with st.expander('3. ตั้งค่ากล้อง', expanded=(source == 'basler')):
            if source == 'basler':
                keep = st.toggle('ใช้ค่าที่ตั้งใน pylon Viewer', value=True,
                                 help='แนะนำ: ตั้งโฟกัส รูรับแสง Exposure White Balance ใน pylon Viewer ก่อน แล้วปิด Viewer')
            st.slider('Exposure (มิลลิวินาที)', 0.5, 200.0, key='exposure', step=0.5, disabled=keep,
                      help='เวลาเปิดรับแสง  นานขึ้น = สว่างขึ้น แต่ชิ้นงานที่เคลื่อนที่จะเบลอ')
            st.slider('Gain', 0.0, 24.0, key='gain', step=1.0, disabled=keep,
                      help='ขยายสัญญาณ  สูงขึ้น = สว่างขึ้น แต่จุดรบกวนมากขึ้น')
            if source == 'basler':
                if st.button('White Balance อัตโนมัติ 1 ครั้ง', width='stretch'):
                    try:
                        cam = open_camera('basler', st.session_state.image, keep)
                        with cam.lock:
                            set_param(cam.cam, ['BalanceWhiteAuto'], 'Once')
                        st.success('ปรับแล้ว ใช้กับภาพถัดไป')
                    except Exception as e:
                        st.error('ปรับไม่ได้: ' + str(e))
            if is_camera and st.button('ปิดกล้อง (คืนให้โปรแกรมอื่น)', width='stretch'):
                open_camera.clear()
                st.success('ปิดกล้องแล้ว')

    if source in ('file', 'emu', 'upload'):
        with st.expander('4. จำลองสภาพโรงงาน', expanded=st.session_state.sim):
            st.toggle('เปิดการจำลอง', key='sim',
                      help='กล้องจำลองและภาพไฟล์ไม่เปลี่ยนตาม Exposure/Gain ส่วนนี้จำลองผลแบบกล้องจริง')
            sim_off = not st.session_state.sim
            st.slider('แสงรอบไลน์', 0.1, 1.5, key='light', step=0.1, disabled=sim_off)
            st.slider('ความเร็วสายพาน', 0.0, 4.0, key='speed', step=0.5, disabled=sim_off,
                      help='พิกเซลต่อมิลลิวินาที  ยิ่งเร็ว + Exposure นาน = ยิ่งเบลอ')
            st.toggle('ครอบกล่อง + ไฟของตัวเอง', key='enclosure', disabled=sim_off,
                      help='แสงคงที่เสมอ ไม่ขึ้นกับแสงรอบไลน์')
            st.toggle('Auto Exposure', key='auto_exposure', disabled=sim_off,
                      help='ปรับ Exposure เองตามความสว่างของภาพ (นานสุด 20 ms)')

S = {k: st.session_state[k] for k in DEFAULTS}


def get_frame():
    if source == 'upload':
        if uploaded is None:
            return None
        return cv2.imdecode(np.frombuffer(uploaded.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if source == 'file':
        return load_image(st.session_state.image)
    cam = open_camera(source, st.session_state.image, keep)
    try:
        with cam.lock:
            if isinstance(cam, BaslerSource) and not keep:
                want = (int(st.session_state.exposure * 1000), st.session_state.gain)
                if cam.settings != want:
                    set_param(cam.cam, ['ExposureTime', 'ExposureTimeAbs'], want[0])
                    set_param(cam.cam, ['Gain', 'GainRaw'], want[1])
                    cam.settings = want
            for _ in range(5):
                f = cam.read()
                if f is not None:
                    return f
    except Exception:
        open_camera.clear()
        raise RuntimeError('การเชื่อมต่อกล้องหลุด กำลังเชื่อมต่อใหม่ ลองอีกครั้ง')
    raise RuntimeError('ภาพจากกล้องมาไม่ครบ ลองใหม่ หรือเช็กสาย LAN / PoE')


def get_ref():
    if not needs_ref(S):
        return None
    if ref_upload is not None:
        return cv2.imdecode(np.frombuffer(ref_upload.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    return load_image(S['ref_image'])


REF = get_ref()


def prepare(frame):
    if S['sim'] and source != 'basler':
        return simulate_station(frame, S, S['light'])[0]
    return frame


def log_result(r, ms):
    st.session_state.log.append({
        'เวลา': datetime.now().strftime('%H:%M:%S'), 'สูตรงาน': st.session_state.recipe,
        'วิธีตรวจ': METHODS[S['method']],
        'ภาพ': st.session_state.image if source in ('file', 'emu') else source_label,
        'ค่าที่วัด': r.metric_label + ': ' + r.metric, 'ผล': r.verdict, 'สาเหตุ': r.reason,
        'เวลาตรวจ (ms)': round(ms, 1)})


# ================= หน้าหลัก =================
st.markdown('## 📷 สถานีตรวจชิ้นงาน')
top = st.columns([1.1, 1.1, 1.2, 2.6])
trigger = top[0].button('📷  Trigger', type='primary', width='stretch',
                        help='ชิ้นงานมาถึง: ถ่าย 1 ภาพ ตรวจ และบันทึกผล')
live = top[1].toggle('ภาพสด', disabled=not is_camera, help='ตรวจภาพจากกล้องต่อเนื่อง (เฉพาะแหล่งภาพที่เป็นกล้อง)')
badge_slot = top[3].empty()
stat_slot = top[2].empty()

tab_view, tab_tune, tab_shift, tab_log = st.tabs(['🔍 ผลตรวจ', '🎚️ ปรับเงื่อนไขตรวจ', '🏭 เดินไลน์ 1 กะ', '📋 บันทึกผล'])


# ---------- แท็บปรับเงื่อนไข (ปรับแล้วผลเปลี่ยนทันทีกับภาพล่าสุด) ----------
def align_input(col):
    col.selectbox('จัดตำแหน่งก่อนตรวจ', list(ALIGN_MODES), key='align', format_func=ALIGN_MODES.get,
                  help='ถาดเลื่อน (Workshop 9 ข้อ 7) ใช้ Template matching  รูปจากมือถือ (Workshop 10) ใช้ ECC')


def tune_color():
    c1, c2 = st.columns(2)
    with c1:
        st.slider('ช่วงสี H (Hue)', 0, 179, key='h_range', help='แดง 0–10  ส้ม 12–22  เหลือง 22–35  เขียว 40–80  น้ำเงิน 95–125')
        lo, hi = st.session_state.h_range
        st.image(hue_bar(lo, hi, st.session_state.s_min, st.session_state.v_min),
                 caption='สีที่สว่างในแถบ = ช่วงที่เลือก', width='stretch')
        st.slider('S ต่ำสุด (ความสด)', 0, 255, key='s_min', help='ลดลงถ้าสีซีด  เพิ่มขึ้นถ้าจับพื้นหลังสีเทาผิด')
        st.slider('V ต่ำสุด (ความสว่าง)', 0, 255, key='v_min', help='ลดลงถ้าภาพมืด')
    with c2:
        st.number_input('พื้นที่ขั้นต่ำของชิ้นงาน (พิกเซล)', 0, 500000, key='min_area', step=100,
                        help='ชิ้นที่เล็กกว่านี้ไม่นับ  ภาพจากกล้องจริงใหญ่กว่า ต้องเพิ่มค่านี้')
        st.number_input('จำนวนชิ้นที่ต้องมี', 0, 100, key='expect', step=1)
        st.markdown('**กฎตรวจเพิ่มเติม**')
        st.checkbox('NG ถ้ามีเศษชิ้นแตก (ชิ้นสีเดียวกันแต่เล็ก)', key='chk_chip')
        st.checkbox('NG ถ้ามีสิ่งแปลกปลอมสีส้ม (H 12–22)', key='chk_foreign')


def tune_reference():
    c1, c2 = st.columns(2)
    c1.slider('Threshold ความต่าง', 1, 100, key='diff_threshold',
              help='ต่ำ = ไวขึ้น จับเส้นผมบางได้ แต่จุดรบกวนมาก  สูง = จับเฉพาะสิ่งที่ต่างชัด')
    c1.number_input('ขนาดเล็กสุดของจุดที่ต่าง (พิกเซล)', 0, 500000, key='min_area', step=10)
    align_input(c2)
    c2.caption('จับสิ่งแปลกปลอมได้ทุกสีทุกรูปร่าง แต่กล้องกับชิ้นงานต้องอยู่ตำแหน่งเดิมและแสงคงที่  '
               'ถ้าชิ้นงานเลื่อน ให้จัดตำแหน่งก่อน')


def tune_label():
    c1, c2 = st.columns(2)
    c1.text_input('ชื่อสินค้าที่ต้องเป็น', key='product')
    c1.slider('คะแนน Match ขั้นต่ำ (หาฉลาก)', 0.0, 1.0, key='match_min', step=0.05,
              help='Template matching ไม่ทนการหมุน ฉลากเอียงจะได้คะแนนต่ำ')
    c1.number_input('การพิมพ์ต่างจากต้นแบบได้ไม่เกิน (สัดส่วนพื้นที่)', 0.0, 1.0, key='diff_max', step=0.001,
                    format='%.3f', help='ลองลดเหลือ 0.001 ดูว่าฉลากที่พิมพ์ชื่อผิดถูกจับจากการเทียบภาพด้วยไหม')
    c2.slider('ความคล้ายของชื่อสินค้าขั้นต่ำ', 0.0, 1.0, key='name_min', step=0.05,
              help='OCR อ่านเพี้ยนได้บ้าง (เช่น D เป็น V) จึงใช้ความคล้ายแทนการเทียบตรงๆ')
    c2.checkbox('ต้องมีวันหมดอายุ (EXP)', key='need_exp')
    c2.checkbox('ต้องมีเลข LOT 6 หลัก', key='need_lot')
    if ocr_engine() is None:
        c2.warning('ไม่พบ Tesseract: ขั้นอ่านข้อความจะตัดสิน NG ทุกภาพ  ติดตั้ง Tesseract ตาม README ก่อน')


def tune_parts():
    c1, c2 = st.columns(2)
    c1.slider('Peak (แยกชิ้นที่ติดกัน)', 0.1, 0.9, key='peak_frac', step=0.05,
              help='ต่ำไป = ชิ้นติดกันไม่แยก  สูงไป = ชิ้นเดียวถูกตัดเป็นหลายชิ้น')
    c1.number_input('รูใหญ่กว่านี้ = แหวน (พิกเซล)', 0, 100000, key='hole_min', step=10)
    c2.number_input('แหวนที่ต้องมี', 0, 100, key='expect_washer', step=1)
    c2.number_input('น็อตที่ต้องมี', 0, 100, key='expect_nut', step=1)


def tune_measure():
    c1, c2 = st.columns(2)
    c1.number_input('ขนาดจริงของเหรียญสอบเทียบ (มม.)', 1.0, 200.0, key='coin_mm', step=0.5)
    c1.caption('ชิ้นงานต้องแบนและอยู่ระยะเดียวกับเหรียญ  วัดได้ละเอียดราว 1 พิกเซล')
    c2.number_input('สเปกด้านยาว (มม.)', 0.0, 1000.0, key='spec_w', step=0.1, format='%.1f')
    c2.number_input('สเปกด้านสั้น (มม.)', 0.0, 1000.0, key='spec_h', step=0.1, format='%.1f')
    c2.number_input('ยอมคลาดเคลื่อน ± (มม.)', 0.0, 100.0, key='tol_mm', step=0.1, format='%.1f')


def tune_ai():
    st.checkbox('เพิ่มประเภท extra: ชิ้นเกินกลางถาด (Workshop 9 ข้อ 8)', key='ai_extra')
    st.caption('โมเดลเรียนจากถาดจำลอง 100 ภาพต่อประเภท (สอน 70% สอบ 30%)  ภาพที่ไม่ใช่ถาดแบบนี้ผลที่ทายไม่มีความหมาย  '
               'กับกล้องจริงต้องสร้างชุดข้อมูลจากภาพจริงก่อน')


def tune_rack_zones():
    c1, c2 = st.columns(2)
    c1.slider('เกณฑ์ไดรฟ์: พื้นที่สว่างขั้นต่ำ', 0.05, 1.0, key='drive_min', step=0.05)
    c2.checkbox('ตรวจไฟสถานะสีเขียวใต้ไดรฟ์ทุกตัว (Workshop 9 ข้อ 5)', key='chk_drive_led')
    c2.caption('โซนกำหนดจากภาพอ้างอิง 800 x 300  ถ้าเครื่องเลื่อน กรอบจะคลาด')


def tune_rack_parts():
    st.caption('ต้องมี ไดรฟ์ 8  พอร์ต 2  ไฟ 2  ฉลาก 1   กฎจำแนกด้วยขนาด/รูปร่างต้องเขียนใหม่เมื่อสินค้าเปลี่ยนรุ่น')


def tune_zones():
    c1, c2 = st.columns(2)
    c1.text_area('โซน บรรทัดละ 1 โซน  ชื่อ: x, y, w, h', key='zones_text', height=180,
                 help='พิกัดดูจากภาพเส้นตารางในแท็บผลตรวจ  รูปมือถือถูกย่อให้กว้าง 800 ก่อน')
    c2.slider('ความสว่างต่างจากภาพอ้างอิงได้ไม่เกิน', 1, 100, key='zone_tol')
    align_input(c2)


def tune_count():
    c1, c2 = st.columns(2)
    c1.number_input('จำนวนชิ้นที่ต้องมี', 0, 100, key='expect', step=1)
    c1.number_input('พื้นที่ขั้นต่ำของชิ้นงาน (พิกเซล)', 0, 500000, key='min_area', step=100)
    c2.checkbox('กลับสี: ชิ้นงานเข้มกว่าพื้นหลัง', key='invert')
    align_input(c2)


TUNERS = {'color': tune_color, 'reference': tune_reference, 'label': tune_label, 'parts': tune_parts,
          'measure': tune_measure, 'ai': tune_ai, 'rack_zones': tune_rack_zones,
          'rack_parts': tune_rack_parts, 'zones': tune_zones, 'count': tune_count}

with tab_tune:
    st.caption('ปรับแล้วภาพล่าสุดถูกตรวจใหม่ทันที  ไม่ต้องกด Trigger ซ้ำ')
    TUNERS[S['method']]()

# ---------- แท็บเดินไลน์ 1 กะ (Workshop 5) ----------
with tab_shift:
    st.caption('ไลน์บรรจุอะไหล่ 1 กะ = 40 ถาดจาก images/factory  ทุกถาดต้องมีชิ้นแดงใหญ่ 5 ชิ้น  แสงเปลี่ยนตามเวลา: '
               'ถาด 1–15 เช้า  16–28 บ่ายเมฆบัง  29–40 เย็นไฟดับหนึ่งดวง  เทียบผลกับเฉลยใน factory_labels.csv')
    if S['method'] != 'color':
        st.info('ใช้กับวิธีตรวจ "' + METHODS['color'] + '"  เลือกสูตร Workshop 5 ทางซ้าย')
    else:
        st.caption('ใช้ Exposure / Gain และการจำลองสภาพโรงงาน (ความเร็วสายพาน กล่องครอบ Auto Exposure) จากแถบซ้าย '
                   'กับเงื่อนไขตรวจจากแท็บปรับเงื่อนไข')
        c = st.columns([1, 1, 1.2], vertical_alignment='bottom')
        c[0].number_input('ทิ้งของดี (บาท/ถาด)', 0, 1_000_000, key='cost_fr', step=10)
        c[1].number_input('ลูกค้าเคลม (บาท/ถาด)', 0, 100_000_000, key='cost_esc', step=100)
        if c[2].button('▶  เดินไลน์ 1 กะ', type='primary', width='stretch'):
            try:
                with st.spinner('กำลังเดินไลน์ 40 ถาด...'):
                    st.session_state.shift = run_shift(S)
            except Exception as e:
                st.error('เดินไลน์ไม่สำเร็จ: ' + str(e))
        if st.session_state.shift:
            show_shift(st.session_state.shift, S['cost_fr'], S['cost_esc'])

# ---------- ได้ภาพ ----------
frame, grab_ms, new_shot = None, 0.0, False
if trigger:
    try:
        t0 = time.perf_counter()
        with st.spinner('กำลังถ่ายภาพ...'):
            raw = get_frame()
        grab_ms = (time.perf_counter() - t0) * 1000
        if raw is None:
            st.warning('ยังไม่ได้เลือกไฟล์ภาพทางซ้าย')
        else:
            st.session_state.last = prepare(raw)
            new_shot = True
    except Exception as e:
        st.error('ถ่ายภาพไม่สำเร็จ: ' + str(e))
elif st.session_state.last is None and source == 'file':
    st.session_state.last = prepare(load_image(st.session_state.image))
frame = st.session_state.last


def render(frame, slot_view, logging=False):
    t0 = time.perf_counter()
    try:
        r = inspect(frame, S, REF)
    except Exception as e:
        badge_slot.error('ตรวจไม่สำเร็จ: ' + str(e))
        return None
    ms = (time.perf_counter() - t0) * 1000
    badge_slot.markdown(badge(r.verdict, r.reason), unsafe_allow_html=True)
    stat_slot.metric(r.metric_label, r.metric, help='เวลาตรวจ ' + str(round(ms)) + ' ms')
    with slot_view:
        a, b = st.columns([1.6, 1])
        a.image(small(r.out), caption='ผลการตรวจ  ' + str(frame.shape[1]) + ' x ' + str(frame.shape[0])
                + '   ความสว่างเฉลี่ย ' + str(int(frame.mean())), width='stretch')
        if r.aux is not None:
            b.image(small(r.aux, 640), caption=r.aux_caption, width='stretch')
        b.image(small(frame, 640), caption='ภาพต้นฉบับ', width='stretch')
    if logging:
        log_result(r, ms)
    return r


@st.fragment(run_every=(1.0 if source == 'basler' else 0.3) if live else None)
def live_view():
    if not live:
        return
    try:
        f = prepare(get_frame())
        st.session_state.last = f
        render(f, st.container())
    except Exception as e:
        st.warning('ภาพนี้ไม่สำเร็จ: ' + str(e))


with tab_view:
    if live:
        live_view()
        if new_shot:
            shot = render(st.session_state.last, st.container(), logging=True)
            if shot:
                st.toast('บันทึกผล: ' + shot.verdict + '  ' + shot.reason)
    elif frame is not None:
        render(frame, st.container(), logging=new_shot)
    else:
        badge_slot.info('เลือกสูตรงานหรือแหล่งภาพทางซ้าย แล้วกด Trigger')

# ---------- แท็บบันทึกผล ----------
with tab_log:
    if st.session_state.log:
        df = pd.DataFrame(st.session_state.log)
        ok = int((df['ผล'] == 'OK').sum())
        m = st.columns(4)
        m[0].metric('ตรวจทั้งหมด', len(df))
        m[1].metric('OK', ok)
        m[2].metric('NG', len(df) - ok)
        m[3].metric('Yield', str(round(ok / len(df) * 100)) + '%')
        ng = df[df['ผล'] == 'NG']
        if len(ng):
            st.markdown('**NG แยกตามสาเหตุ**')
            st.bar_chart(ng['สาเหตุ'].str.split().str[0].value_counts())
        st.dataframe(df.iloc[::-1], width='stretch', hide_index=True)
        d1, d2 = st.columns(2)
        d1.download_button('⬇️ ดาวน์โหลด CSV', df.to_csv(index=False).encode('utf-8-sig'),
                           'inspection_log.csv', 'text/csv', width='stretch')
        if d2.button('ล้างบันทึก', width='stretch'):
            st.session_state.log = []
            st.rerun()
    else:
        st.caption('ยังไม่มีการตรวจ  กด Trigger เพื่อบันทึกผล')

    # Workshop 10 ข้อ 1 + 6: สรุปโครงการ 1 หน้า จากผลที่บันทึกไว้
    with st.expander('📝 สรุปโครงการ 1 หน้า (Workshop 10)'):
        c = st.columns(2)
        c[0].text_input('ชื่อโครงการ', key='p_name')
        c[0].text_input('ชิ้นงานคืออะไร', key='p_product')
        c[0].text_input('ของเสียที่ต้องการจับ', key='p_defect')
        c[1].text_input('ตอนนี้ตรวจอย่างไร', key='p_now')
        c[1].text_input('เป้าหมาย', key='p_goal')
        log = st.session_state.log
        n_ng = sum(row['ผล'] == 'NG' for row in log)
        summary = [
            'สรุปโครงการ: ' + st.session_state.p_name, '',
            'ชิ้นงาน: ' + st.session_state.p_product,
            'ของเสียที่ต้องการจับ: ' + st.session_state.p_defect,
            'วิธีตรวจปัจจุบัน: ' + st.session_state.p_now,
            'เป้าหมาย: ' + st.session_state.p_goal,
            'วิธีที่ทดลอง: ' + METHODS[S['method']],
            f'ผลทดลอง: {len(log)} รูป  ตัดสิน NG {n_ng} รูป  OK {len(log) - n_ng} รูป',
            f'ผลที่ถูกต้องเมื่อตรวจด้วยตา: ___ / {len(log)} รูป', '',
            'ขั้นต่อไป:',
            '1. ถ่ายรูปเพิ่มให้ครบทุกแบบของเสีย อย่างน้อย 20 รูป',
            '2. ทำขาตั้งยึดมือถือหรือกล้อง ให้ระยะคงที่ และคุมแสง',
            '3. ถ้าแม่นพอ ทดลองกล้องอุตสาหกรรม + จิ๊กที่หน้างาน 1 จุด',
        ]
        st.caption('คัดลอกไปใส่อีเมลหรือรายงานถึงหัวหน้าได้  ช่อง "ผลที่ถูกต้อง" ให้ตรวจด้วยตาว่าโปรแกรมตัดสินถูกกี่รูป')
        st.code('\n'.join(summary), language=None)
