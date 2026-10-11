# สถานีตรวจชิ้นงาน (Web app)  รันด้วย:  python3 -m streamlit run app/inspection_app.py
import os
import sys
import threading
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
Q = chr(34)

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
}
SOURCES = {'ภาพตัวอย่าง': 'file', 'อัปโหลดภาพ': 'upload', 'กล้อง Basler จำลอง': 'emu',
           'กล้อง Basler จริง': 'basler', 'Webcam': 'webcam'}
DEFAULTS = dict(source='ภาพตัวอย่าง', image='shapes.png', h_range=(0, 10), s_min=50, v_min=50,
                min_area=1000, expect=5, chk_chip=False, chk_foreign=False, sim=False,
                light=1.0, speed=0.0, exposure=5.0, gain=0.0)

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
for k, v in (('log', []), ('last', None), ('recipe', list(RECIPES)[0])):
    st.session_state.setdefault(k, v)


def apply_recipe():
    r = RECIPES[st.session_state.recipe]
    for k, v in DEFAULTS.items():
        st.session_state[k] = r.get(k, v)
    st.session_state.last = None


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


def sample_images():
    names = ['shapes.png']
    for folder in ('line', 'factory'):
        d = img_path(folder)
        if os.path.isdir(d):
            names += [folder + '/' + f for f in sorted(os.listdir(d)) if f.endswith('.png')]
    return names


def count_color(hsv, lo, hi, s_min, v_min):
    mask = cv2.inRange(hsv, np.array([lo, s_min, v_min]), np.array([hi, 255, 255]))
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return mask, cs


def inspect(img, s):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lo, hi = s['h_range']
    mask, cs = count_color(hsv, lo, hi, s['s_min'], s['v_min'])
    out = img.copy()
    thick = max(2, img.shape[1] // 200)
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
        reason = 'ครบถ้วน'
    verdict = 'OK' if reason == 'ครบถ้วน' else 'NG'
    return verdict, reason, n, mask, out


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


# ================= แถบซ้าย =================
with st.sidebar:
    st.markdown('### 1. สูตรงาน')
    st.selectbox('เลือกสูตรตาม Workshop', list(RECIPES), key='recipe', on_change=apply_recipe,
                 label_visibility='collapsed')
    st.caption(RECIPES[st.session_state.recipe]['desc'])
    st.button('↺ คืนค่าตามสูตร', on_click=apply_recipe, width='stretch')

    st.markdown('### 2. แหล่งภาพ')
    source_label = st.selectbox('ภาพมาจาก', list(SOURCES), key='source')
    source = SOURCES[source_label]
    is_camera = source in ('emu', 'basler', 'webcam')
    uploaded = None
    if source in ('file', 'emu'):
        imgs = sample_images()
        if st.session_state.image not in imgs:
            st.session_state.image = imgs[0]
        st.selectbox('ภาพชิ้นงาน', imgs, key='image')
    if source == 'upload':
        uploaded = st.file_uploader('เลือกไฟล์ภาพ', type=['png', 'jpg', 'jpeg', 'bmp'])

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
            st.slider('แสงรอบไลน์', 0.1, 1.5, key='light', step=0.1, disabled=not st.session_state.sim)
            st.slider('ความเร็วสายพาน', 0.0, 4.0, key='speed', step=0.5, disabled=not st.session_state.sim,
                      help='พิกเซลต่อมิลลิวินาที  ยิ่งเร็ว + Exposure นาน = ยิ่งเบลอ')

S = {k: st.session_state[k] for k in ('h_range', 's_min', 'v_min', 'min_area', 'expect', 'chk_chip', 'chk_foreign')}


def get_frame():
    if source == 'upload':
        if uploaded is None:
            return None
        return cv2.imdecode(np.frombuffer(uploaded.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if source == 'file':
        return cv2.imread(img_path(st.session_state.image))
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


def prepare(frame):
    if st.session_state.sim and source != 'basler':
        return simulate_real(frame, st.session_state.exposure, st.session_state.gain,
                             st.session_state.light, st.session_state.speed)
    return frame


def log_result(verdict, reason, n, ms):
    st.session_state.log.append({
        'เวลา': datetime.now().strftime('%H:%M:%S'), 'สูตรงาน': st.session_state.recipe,
        'ภาพ': st.session_state.image if source in ('file', 'emu') else source_label,
        'พบ': n, 'ต้องมี': S['expect'], 'ผล': verdict, 'สาเหตุ': reason, 'เวลาตรวจ (ms)': round(ms, 1)})


# ================= หน้าหลัก =================
st.markdown('## 📷 สถานีตรวจชิ้นงาน')
top = st.columns([1.1, 1.1, 1.2, 2.6])
trigger = top[0].button('📷  Trigger', type='primary', width='stretch',
                        help='ชิ้นงานมาถึง: ถ่าย 1 ภาพ ตรวจ และบันทึกผล')
live = top[1].toggle('ภาพสด', disabled=not is_camera, help='ตรวจภาพจากกล้องต่อเนื่อง (เฉพาะแหล่งภาพที่เป็นกล้อง)')
badge_slot = top[3].empty()
stat_slot = top[2].empty()

tab_view, tab_tune, tab_log = st.tabs(['🔍 ผลตรวจ', '🎚️ ปรับเงื่อนไขตรวจ', '📋 บันทึกผล'])

# ---------- แท็บปรับเงื่อนไข (ปรับแล้วผลเปลี่ยนทันทีกับภาพล่าสุด) ----------
with tab_tune:
    st.caption('ปรับแล้วภาพล่าสุดถูกตรวจใหม่ทันที  ไม่ต้องกด Trigger ซ้ำ')
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
    st.session_state.last = prepare(cv2.imread(img_path(st.session_state.image)))
frame = st.session_state.last


def render(frame, slot_view, logging=False):
    t0 = time.perf_counter()
    verdict, reason, n, mask, out = inspect(frame, S)
    ms = (time.perf_counter() - t0) * 1000
    badge_slot.markdown(badge(verdict, reason), unsafe_allow_html=True)
    stat_slot.metric('พบ / ต้องมี', str(n) + ' / ' + str(S['expect']), help='เวลาตรวจ ' + str(round(ms)) + ' ms')
    with slot_view:
        a, b = st.columns([1.6, 1])
        a.image(small(out), caption='ผลการตรวจ  ' + str(frame.shape[1]) + ' x ' + str(frame.shape[0])
                + '   ความสว่างเฉลี่ย ' + str(int(frame.mean())), width='stretch')
        b.image(small(mask, 640), caption='Mask (สีที่เลือก)', width='stretch')
        b.image(small(frame, 640), caption='ภาพต้นฉบับ', width='stretch')
    if logging:
        log_result(verdict, reason, n, ms)
    return verdict, reason


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
            v, r = render(st.session_state.last, st.container(), logging=True)
            st.toast('บันทึกผล: ' + v + '  ' + r)
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
