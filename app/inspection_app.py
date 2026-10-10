# สถานีตรวจชิ้นงาน (Web app)  รันด้วย:  python3 -m streamlit run app/inspection_app.py
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from vision_camera import (BaslerSource, WebcamSource, set_param,  # noqa: E402
                           simulate_real, img_path)

st.set_page_config(page_title='สถานีตรวจชิ้นงาน', page_icon='📷', layout='wide')

for key, val in (('log', []), ('last', None)):
    if key not in st.session_state:
        st.session_state[key] = val


# ---------------- กล้อง: เปิดครั้งเดียวแล้วค้างไว้ (ถ่ายเร็ว) ----------------
@st.cache_resource(max_entries=1)
def open_camera(kind, image):
    # เปิดกล้องครั้งเดียว ใช้ต่อทุกครั้งที่ Trigger / Live  (เปลี่ยนแหล่งภาพ = ปิดตัวเก่า เปิดตัวใหม่)
    if kind == 'webcam':
        return WebcamSource()
    return BaslerSource(emulate=(kind == 'emu'), image_path=img_path(image))


def apply_camera_settings(cam, exposure_us, gain):
    if isinstance(cam, BaslerSource):
        set_param(cam.cam, ['ExposureTime', 'ExposureTimeAbs'], exposure_us)
        set_param(cam.cam, ['Gain', 'GainRaw'], gain)


def sample_images():
    names = ['shapes.png']
    for folder in ('line', 'factory'):
        d = img_path(folder)
        if os.path.isdir(d):
            names += [folder + '/' + f for f in sorted(os.listdir(d)) if f.endswith('.png')]
    return names


def inspect(img, s):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([s['h_low'], s['s_min'], s['v_min']]),
                       np.array([s['h_high'], 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = img.copy()
    n = 0
    thick = max(2, img.shape[1] // 200)
    for c in contours:
        if cv2.contourArea(c) > s['min_area']:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(out, (x, y), (x + w, y + h), (200, 0, 100), thick)
            n += 1
    verdict = 'OK' if n == s['expect'] else 'NG'
    return verdict, n, mask, out


def small(img, width=960):
    # ย่อเฉพาะตอนแสดงผล ให้เบราว์เซอร์แสดงเร็ว
    if img.shape[1] > width:
        img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])))
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def badge(verdict, n, expect, extra=''):
    color = '#1E7E34' if verdict == 'OK' else '#C62828'
    return ('<div style=' + chr(34) + 'background:' + color + ';color:white;border-radius:12px;'
            'padding:8px 24px;font-size:40px;font-weight:700;text-align:center' + chr(34) + '>'
            + verdict + '  ' + str(n) + ' / ' + str(expect) + extra + '</div>')


# ---------------- แถบด้านซ้าย ----------------
st.sidebar.title('ตั้งค่าสถานีตรวจ')
st.sidebar.subheader('1. แหล่งภาพ')
LABELS = {'ภาพตัวอย่าง': 'file', 'อัปโหลดภาพ': 'upload', 'กล้อง Basler จำลอง': 'emu',
          'กล้อง Basler จริง': 'basler', 'Webcam': 'webcam'}
source_label = st.sidebar.radio('ภาพมาจาก', list(LABELS))
source = LABELS[source_label]
is_camera = source in ('emu', 'basler', 'webcam')

image_name = 'shapes.png'
uploaded = None
if source in ('file', 'emu'):
    image_name = st.sidebar.selectbox('ภาพชิ้นงาน', sample_images())
if source == 'upload':
    uploaded = st.sidebar.file_uploader('เลือกไฟล์ภาพ', type=['png', 'jpg', 'jpeg', 'bmp'])

st.sidebar.subheader('2. กล้อง')
exposure_ms = st.sidebar.slider('Exposure (มิลลิวินาที)', 0.5, 200.0, 5.0, 0.5)
gain = st.sidebar.slider('Gain', 0.0, 24.0, 0.0, 1.0)
if source == 'basler':
    st.sidebar.caption('ปิด pylon Viewer ก่อนใช้  ภาพมืด: เปิดฝาเลนส์ / หมุนรูรับแสงให้กว้าง / เพิ่มแสง')
if source == 'webcam':
    st.sidebar.caption('Mac: ครั้งแรกต้องอนุญาตให้ Terminal ใช้กล้อง (System Settings > Privacy > Camera)')
if is_camera and st.sidebar.button('ปิดกล้อง (คืนกล้องให้โปรแกรมอื่น)'):
    open_camera.clear()
    st.sidebar.success('ปิดกล้องแล้ว')

simulate = False
light, speed = 1.0, 0.0
if source in ('file', 'emu', 'upload'):
    with st.sidebar.expander('จำลองสภาพโรงงาน (แสง / สายพาน)'):
        simulate = st.checkbox('เปิดการจำลอง', value=False)
        light = st.slider('แสงรอบไลน์', 0.1, 1.5, 1.0, 0.1)
        speed = st.slider('ความเร็วสายพาน (พิกเซล/มิลลิวินาที)', 0.0, 4.0, 0.0, 0.5)

st.sidebar.subheader('3. เงื่อนไขตรวจ')
h_low, h_high = st.sidebar.slider('ช่วงสี H (แดง = 0–10)', 0, 179, (0, 10))
S = dict(h_low=h_low, h_high=h_high,
         s_min=st.sidebar.slider('S ต่ำสุด (ความสด)', 0, 255, 50),
         v_min=st.sidebar.slider('V ต่ำสุด (ความสว่าง)', 0, 255, 50),
         min_area=st.sidebar.number_input('พื้นที่ขั้นต่ำ (พิกเซล)', 0, 500000, 1000, 100),
         expect=st.sidebar.number_input('จำนวนชิ้นที่ต้องมี', 0, 100, 5, 1))


def get_frame():
    # ได้ภาพ 1 ภาพจากแหล่งที่เลือก
    if source == 'upload':
        if uploaded is None:
            return None
        return cv2.imdecode(np.frombuffer(uploaded.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if source == 'file':
        return cv2.imread(img_path(image_name))
    cam = open_camera(source, image_name)
    apply_camera_settings(cam, int(exposure_ms * 1000), gain)
    for _ in range(5):
        f = cam.read()
        if f is not None:
            return f
    raise RuntimeError('ภาพจากกล้องมาไม่ครบ ลองใหม่อีกครั้ง หรือเช็กสาย LAN / PoE')


def prepare(frame):
    if simulate and source != 'basler':
        return simulate_real(frame, exposure_ms, gain, light, speed)
    return frame


# ---------------- หน้าหลัก ----------------
st.title('สถานีตรวจชิ้นงาน')
live = False
if is_camera:
    live = st.toggle('🔴 ภาพสด (Live)  ตรวจทุกภาพแบบต่อเนื่อง', value=False)
st.caption('กด Trigger = ชิ้นงานมาถึง กล้องถ่าย 1 ภาพ ตรวจ แล้วบันทึกผล')


@st.fragment(run_every=0.3 if live else None)
def live_view():
    if not live:
        return
    try:
        t0 = time.perf_counter()
        frame = prepare(get_frame())
        verdict, n, mask, out = inspect(frame, S)
        ms = (time.perf_counter() - t0) * 1000
    except Exception as e:
        st.error('กล้องมีปัญหา: ' + str(e))
        return
    st.session_state.last = frame
    st.markdown(badge(verdict, n, S['expect'], '   (' + str(round(ms)) + ' ms)'), unsafe_allow_html=True)
    a, b = st.columns(2)
    a.image(small(out), caption='ภาพสด + ผลการตรวจ  ' + str(frame.shape[1]) + ' x ' + str(frame.shape[0]),
            width='stretch')
    b.image(small(mask), caption='Mask (สีที่เลือก)', width='stretch')


c1, c2, c3 = st.columns([1, 1, 2])
trigger = c1.button('📷 Trigger', type='primary', width='stretch')
recheck = c2.button('ตรวจภาพเดิมซ้ำ', width='stretch', disabled=st.session_state.last is None)

live_view()

frame = None
grab_ms = 0.0
if trigger:
    try:
        t0 = time.perf_counter()
        with st.spinner('กำลังถ่ายภาพ...'):
            raw = get_frame()
        grab_ms = (time.perf_counter() - t0) * 1000
        if raw is None:
            st.warning('ยังไม่ได้เลือกไฟล์ภาพทางซ้าย')
        else:
            frame = prepare(raw)
            st.session_state.last = frame
    except Exception as e:
        st.error('ถ่ายภาพไม่สำเร็จ: ' + str(e))
elif recheck:
    frame = st.session_state.last

if frame is not None:
    t0 = time.perf_counter()
    verdict, n, mask, out = inspect(frame, S)
    ms = (time.perf_counter() - t0) * 1000
    c3.markdown(badge(verdict, n, S['expect']), unsafe_allow_html=True)
    st.session_state.log.append({
        'เวลา': datetime.now().strftime('%H:%M:%S'), 'แหล่งภาพ': source_label,
        'ภาพ': image_name if source in ('file', 'emu') else '-',
        'พบ': n, 'ต้องมี': S['expect'], 'ผล': verdict,
        'ถ่ายภาพ (ms)': round(grab_ms), 'ประมวลผล (ms)': round(ms, 1)})
    if not live:
        a, b, c = st.columns(3)
        a.image(small(frame), caption='ภาพจากกล้อง  ' + str(frame.shape[1]) + ' x ' + str(frame.shape[0]),
                width='stretch')
        b.image(small(mask), caption='Mask (สีที่เลือก)', width='stretch')
        c.image(small(out), caption='ผลการตรวจ', width='stretch')
elif not live:
    st.info('ตั้งค่าทางซ้าย แล้วกด Trigger  (กล้องจะเปิดค้างไว้ ครั้งแรกช้า ครั้งต่อไปเร็ว)')

# ---------------- บันทึกผล ----------------
st.subheader('บันทึกผลการตรวจ')
if st.session_state.log:
    df = pd.DataFrame(st.session_state.log)
    ok = int((df['ผล'] == 'OK').sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('ตรวจทั้งหมด', len(df))
    m2.metric('OK', ok)
    m3.metric('NG', len(df) - ok)
    m4.metric('อัตรา OK', str(round(ok / len(df) * 100)) + '%')
    st.dataframe(df.iloc[::-1], width='stretch', hide_index=True)
    d1, d2 = st.columns(2)
    d1.download_button('ดาวน์โหลด CSV', df.to_csv(index=False).encode('utf-8-sig'),
                       'inspection_log.csv', 'text/csv')
    if d2.button('ล้างบันทึก'):
        st.session_state.log = []
        st.rerun()
else:
    st.caption('ยังไม่มีการตรวจ')
