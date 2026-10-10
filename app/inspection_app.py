# สถานีตรวจชิ้นงาน (Web app)  รันด้วย:  streamlit run app/inspection_app.py
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from vision_camera import grab_one, simulate_real, img_path  # noqa: E402

st.set_page_config(page_title='สถานีตรวจชิ้นงาน', page_icon='📷', layout='wide')

if 'log' not in st.session_state:
    st.session_state.log = []
if 'last' not in st.session_state:
    st.session_state.last = None


def sample_images():
    names = ['shapes.png']
    for folder in ('line', 'factory'):
        d = img_path(folder)
        if os.path.isdir(d):
            names += [folder + '/' + f for f in sorted(os.listdir(d)) if f.endswith('.png')]
    return names


def inspect(img, h_low, h_high, s_min, v_min, min_area, expect):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([h_low, s_min, v_min]), np.array([h_high, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = img.copy()
    n = 0
    thick = max(2, img.shape[1] // 200)
    for c in contours:
        if cv2.contourArea(c) > min_area:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(out, (x, y), (x + w, y + h), (200, 0, 100), thick)
            n += 1
    verdict = 'OK' if n == expect else 'NG'
    return verdict, n, mask, out


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------- แถบด้านซ้าย: ตั้งค่า ----------------
st.sidebar.title('ตั้งค่าสถานีตรวจ')

st.sidebar.subheader('1. แหล่งภาพ')
source_label = st.sidebar.radio('ภาพมาจาก', ['ภาพตัวอย่าง', 'อัปโหลดภาพ', 'กล้อง Basler จำลอง',
                                           'กล้อง Basler จริง', 'Webcam'])
source = {'ภาพตัวอย่าง': 'file', 'อัปโหลดภาพ': 'upload', 'กล้อง Basler จำลอง': 'emu',
          'กล้อง Basler จริง': 'basler', 'Webcam': 'webcam'}[source_label]

image_name = 'shapes.png'
uploaded = None
if source in ('file', 'emu'):
    image_name = st.sidebar.selectbox('ภาพชิ้นงาน', sample_images())
if source == 'upload':
    uploaded = st.sidebar.file_uploader('เลือกไฟล์ภาพ', type=['png', 'jpg', 'jpeg', 'bmp'])

st.sidebar.subheader('2. กล้อง')
exposure_ms = st.sidebar.slider('Exposure (มิลลิวินาที)', 0.5, 100.0, 5.0, 0.5)
gain_db = st.sidebar.slider('Gain (dB)', 0.0, 24.0, 0.0, 1.0)
if source == 'basler':
    st.sidebar.caption('ปิด pylon Viewer ก่อนกด Trigger  ถ่ายแต่ละภาพใช้เวลาประมาณ 5 วินาที')

simulate = False
light, speed = 1.0, 0.0
if source in ('file', 'emu', 'upload'):
    with st.sidebar.expander('จำลองสภาพโรงงาน (แสง / สายพาน)'):
        simulate = st.checkbox('เปิดการจำลอง', value=False)
        light = st.slider('แสงรอบไลน์', 0.1, 1.5, 1.0, 0.1)
        speed = st.slider('ความเร็วสายพาน (พิกเซล/มิลลิวินาที)', 0.0, 4.0, 0.0, 0.5)
        st.caption('กล้องจำลองไม่เปลี่ยนภาพตาม Exposure / Gain  ส่วนนี้จึงจำลองผลให้')

st.sidebar.subheader('3. เงื่อนไขตรวจ')
h_low, h_high = st.sidebar.slider('ช่วงสี H (แดง = 0–10)', 0, 179, (0, 10))
s_min = st.sidebar.slider('S ต่ำสุด (ความสด)', 0, 255, 50)
v_min = st.sidebar.slider('V ต่ำสุด (ความสว่าง)', 0, 255, 50)
min_area = st.sidebar.number_input('พื้นที่ขั้นต่ำ (พิกเซล)', 0, 500000, 1000, 100)
expect = st.sidebar.number_input('จำนวนชิ้นที่ต้องมี', 0, 100, 5, 1)

# ---------------- หน้าหลัก ----------------
st.title('สถานีตรวจชิ้นงาน')
st.caption('กด Trigger = ชิ้นงานมาถึง กล้องถ่าย 1 ภาพ แล้วตรวจตามเงื่อนไขทางซ้าย')

c1, c2, c3 = st.columns([1, 1, 2])
trigger = c1.button('📷 Trigger', type='primary', width='stretch')
recheck = c2.button('ตรวจภาพเดิมซ้ำ', width='stretch',
                    disabled=st.session_state.last is None)

frame = None
if trigger:
    try:
        with st.spinner('กำลังถ่ายภาพ...'):
            if source == 'upload':
                if uploaded is None:
                    st.warning('ยังไม่ได้เลือกไฟล์ภาพทางซ้าย')
                    st.stop()
                data = np.frombuffer(uploaded.getvalue(), np.uint8)
                frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            else:
                frame = grab_one(source, image=image_name, exposure_us=int(exposure_ms * 1000),
                                 gain=gain_db)
        st.session_state.last = frame
    except Exception as e:
        st.error('ถ่ายภาพไม่สำเร็จ: ' + str(e))
        st.stop()
elif recheck:
    frame = st.session_state.last

if frame is not None:
    t0 = time.perf_counter()
    img = frame
    if simulate and source != 'basler':
        img = simulate_real(frame, exposure_ms, gain_db, light, speed)
    verdict, n, mask, out = inspect(img, h_low, h_high, s_min, v_min, min_area, expect)
    ms = (time.perf_counter() - t0) * 1000

    color = '#1E7E34' if verdict == 'OK' else '#C62828'
    c3.markdown('<div style="background:' + color + ';color:white;border-radius:12px;'
                'padding:8px 24px;font-size:40px;font-weight:700;text-align:center">'
                + verdict + '  ' + str(n) + ' / ' + str(expect) + '</div>', unsafe_allow_html=True)

    st.session_state.log.append({
        'เวลา': datetime.now().strftime('%H:%M:%S'),
        'แหล่งภาพ': source_label,
        'ภาพ': image_name if source in ('file', 'emu') else '-',
        'พบ': n, 'ต้องมี': expect, 'ผล': verdict,
        'ประมวลผล (ms)': round(ms, 1),
    })

    a, b, c = st.columns(3)
    a.image(rgb(img), caption='ภาพจากกล้อง  ' + str(img.shape[1]) + ' x ' + str(img.shape[0]),
            width='stretch')
    b.image(mask, caption='Mask (สีที่เลือก)', width='stretch')
    c.image(rgb(out), caption='ผลการตรวจ', width='stretch')
else:
    st.info('ตั้งค่าทางซ้าย แล้วกด Trigger')

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
