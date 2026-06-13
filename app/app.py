import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import folium
from streamlit_folium import st_folium
from pathlib import Path
import sys

# إضافة المسار للمشروع
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.feature_engineering import feature_engineering
from src.preprocessing import encode_categorical
from src.predict import load_model, load_features, align_features, predict

# 1. تهيئة الصفحة بأعلى معايير الـ Premium UX
st.set_page_config(
    page_title="NYC Taxi Trip Duration Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. هندسة الواجهة الأمامية بالـ Custom CSS الفاتح والمريح
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cabinet+Grotesk:wght@700&family=Satoshi:wght@400;500;700&display=swap');
    
    /* النظام اللوني الشامل - مريح، فاتح، وواضح */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #F8FAFC !important;
        font-family: 'Satoshi', sans-serif;
    }
    
    .leaflet-control-attribution { display: none !important; }
    
    /* العناوين الرئيسية بجاذبية بصرية عالية */
    .main-title { 
        font-family: 'Cabinet Grotesk', sans-serif;
        font-size: 2.6rem; 
        font-weight: 700; 
        color: #0F172A; 
        margin-bottom: 0.1rem; 
        letter-spacing: -0.03em; 
    }
    .subtitle { 
        font-size: 1.1rem; 
        color: #475569; 
        margin-bottom: 2.5rem; 
        font-weight: 400; 
    }
    
    /* كروت البيانات الفاخرة (Stripe-Style Cards) */
    .ux-card {
        background: #FFFFFF; 
        border: 1px solid #E2E8F0; 
        padding: 1.4rem;
        border-radius: 20px; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02), 0 2px 4px -1px rgba(0, 0, 0, 0.01);
        margin-bottom: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .ux-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.05), 0 10px 10px -5px rgba(0, 0, 0, 0.02);
        border-color: #CBD5E1;
    }
    .ux-card-title { 
        font-size: 0.75rem; 
        text-transform: uppercase; 
        color: #64748B; 
        font-weight: 700; 
        letter-spacing: 0.06em; 
    }
    .ux-card-value { 
        font-size: 1.5rem; 
        font-weight: 700; 
        color: #0F172A; 
        margin-top: 0.4rem; 
    }
    
    /* صناديق الإرشادات سهلة القراءة وفاتحة جداً */
    .instruction-box {
        padding: 1.2rem; 
        border-radius: 16px; 
        font-size: 1rem;
        font-weight: 500; 
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    .ins-blue { background-color: #EFF6FF; color: #1E40AF; border: 1px solid #BFDBFE; }
    .ins-green { background-color: #ECFDF5; color: #065F46; border: 1px solid #A7F3D0; }
    .ins-red { background-color: #FEF2F2; color: #991B1B; border: 1px solid #FEE2E2; }
    
    /* أزرار تفاعلية مبهجة وفاتحة وسهلة الضغط */
    div.stButton > button:first-child {
        border-radius: 14px; 
        font-weight: 700; 
        padding: 0.8rem 1.5rem; 
        font-size: 1rem;
        background: #0284C7; /* الأزرق السماوي الواضح */
        border: none; 
        color: white; 
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.3);
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background: #0369A1;
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.4);
        transform: translateY(-1px);
    }
    
    hr { border-color: #E2E8F0; margin: 2rem 0; }
    </style>
""", unsafe_allow_html=True)

# ترويسة الصفحة الواضحة
st.markdown('<div class="main-title">📍 NYC Taxi Trip Duration Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predictive Machine Learning Service & Route Optimization Dashboard</div>', unsafe_allow_html=True)

# حدود نيويورك الصارمة لمنع الـ Zoom Out خارجها تماماً
NYC_CENTER = [40.7128, -74.0060]
NYC_BOUNDS = [[40.4774, -74.2591], [40.9176, -73.7004]]

@st.cache_resource
def get_model_and_features():
    return load_model(), load_features()

model, features = get_model_and_features()

# تهيئة الـ Session State
if "pickup_coords" not in st.session_state:
    st.session_state.pickup_coords = None
if "dropoff_coords" not in st.session_state:
    st.session_state.dropoff_coords = None
if "selection_mode" not in st.session_state:
    st.session_state.selection_mode = "pickup"

# بناء الخريطة التفاعلية الفاتحة المحدودة بنيويورك
def build_master_map():
    m = folium.Map(
        location=NYC_CENTER,
        zoom_start=12,
        min_zoom=11,
        max_bounds=True,
        min_lat=40.4774, max_lat=40.9176,
        min_lon=-74.2591, max_lon=-73.7004,
        tiles="CartoDB positron" # خلفية بيضاء مريحة ونظيفة جداً للخرائط
    )
    m.fit_bounds(NYC_BOUNDS)

    # إظهار نقطة الركوب فوراً إذا وُجدت
    if st.session_state.pickup_coords:
        folium.Marker(
            location=st.session_state.pickup_coords,
            popup="🟢 Pickup Point",
            icon=folium.Icon(color="green", icon="play", prefix="fa")
        ).add_to(m)

    # إظهار نقطة النزول فوراً إذا وُجدت
    if st.session_state.dropoff_coords:
        folium.Marker(
            location=st.session_state.dropoff_coords,
            popup="🔴 Dropoff Point",
            icon=folium.Icon(color="red", icon="stop", prefix="fa")
        ).add_to(m)

    # رسم خط الرحلة فور اكتمال النقطتين
    if st.session_state.pickup_coords and st.session_state.dropoff_coords:
        folium.PolyLine(
            locations=[st.session_state.pickup_coords, st.session_state.dropoff_coords],
            color="#0284C7", weight=5, opacity=0.85
        ).add_to(m)
        m.fit_bounds([st.session_state.pickup_coords, st.session_state.dropoff_coords])

    return m

# --- القائمة الجانبية (Sidebar UI) ---
with st.sidebar:
    st.markdown("### 🎛️ Ride Configurations")
    
    vendor_id = st.selectbox(
        "Vendor Dispatcher", options=[1, 2],
        format_func=lambda x: f"Creative Mobile Tech (Vendor {x})" if x == 1 else f"VeriFone Inc (Vendor {x})"
    )
    passenger_count = st.selectbox("Passenger Count", options=[1, 2, 3, 4, 5, 6], index=0)
    pickup_date = st.date_input("Pickup Date")
    pickup_time = st.time_input("Pickup Time")

    st.markdown("---")
    
    # الـ UX Switcher: العنصر الأساسي لتوجيه نقرات الخريطة بلغة واضحة
    st.markdown("### 🗺️ Map Mode Selection")
    st.session_state.selection_mode = st.radio(
        "Clicking on map will set:",
        options=["pickup", "dropoff"],
        format_func=lambda x: "🟢 Set Pickup Location" if x == "pickup" else "🔴 Set Dropoff Location"
    )
    
    if st.button("🔄 Clear & Reset Map", use_container_width=True):
        st.session_state.pickup_coords = None
        st.session_state.dropoff_coords = None
        st.rerun()

# --- تخطيط الصفحة الرئيسي (Main Dashboard Layout) ---
col_left, col_right = st.columns([2.5, 1], gap="large")

with col_left:
    # نظام إرشادات ديناميكي مبهج يتغير فوراً مع كل حركة يقوم بها المستخدم
    if not st.session_state.pickup_coords:
        st.markdown('<div class="instruction-box ins-blue">👋 <b>Welcome! Step 1:</b> Click anywhere on the map to set your <b>Pickup Location (🟢)</b>.</div>', unsafe_allow_html=True)
    elif not st.session_state.dropoff_coords:
        st.markdown('<div class="instruction-box ins-blue">🎯 <b>Pickup Set! Step 2:</b> Now change the mode on the sidebar to <b>"Set Dropoff Location"</b> and click your destination.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="instruction-box ins-green">✨ <b>Perfect!</b> Your route is ready. Click the <b>Run Model Prediction</b> button on the right to start calculation.</div>', unsafe_allow_html=True)

    # عرض الخريطة الرئيسية بـ Key جديد تماماً لكسر كاش المتصفح القديم
    main_map = build_master_map()
    map_result = st_folium(
        main_map,
        width=920,
        height=540,
        key="master_nyc_map" # مفتاح جديد يجبر المتصفح على قراءة التحديث حياً
    )

    # منطق التقاط الإحداثيات وتحديث الواجهة فوراً
    if map_result and map_result.get("last_clicked"):
        clicked = [map_result["last_clicked"]["lat"], map_result["last_clicked"]["lng"]]
        
        if st.session_state.selection_mode == "pickup" and st.session_state.pickup_coords != clicked:
            st.session_state.pickup_coords = clicked
            st.rerun()
        elif st.session_state.selection_mode == "dropoff" and st.session_state.dropoff_coords != clicked:
            st.session_state.dropoff_coords = clicked
            st.rerun()

with col_right:
    st.markdown('<h3 style="font-size:1.2rem; font-weight:700; color:#0F172A; margin-bottom:1rem;">📋 Summary & Results</h3>', unsafe_allow_html=True)

    p_ok = st.session_state.pickup_coords is not None
    d_ok = st.session_state.dropoff_coords is not None

    # كروت العرض الفاتحة والواضحة للملخص الرقمي للرحلة
    st.markdown(f"""
        <div class="ux-card">
            <div class="ux-card-title">🛫 Pickup Location</div>
            <div class="ux-card-value" style="font-size:0.95rem; color: {'#059669' if p_ok else '#94A3B8'}; font-weight: 600;">
                {f"{st.session_state.pickup_coords[0]:.4f}, {st.session_state.pickup_coords[1]:.4f}" if p_ok else "Not Set"}
            </div>
        </div>
        <div class="ux-card">
            <div class="ux-card-title">🛬 Dropoff Location</div>
            <div class="ux-card-value" style="font-size:0.95rem; color: {'#059669' if d_ok else '#94A3B8'}; font-weight: 600;">
                {f"{st.session_state.dropoff_coords[0]:.4f}, {st.session_state.dropoff_coords[1]:.4f}" if d_ok else "Not Set"}
            </div>
        </div>
        <div class="ux-card">
            <div class="ux-card-title">👥 Passengers</div>
            <div class="ux-card-value">{passenger_count}</div>
        </div>
    """, unsafe_allow_html=True)

    predict_btn = st.button("⚡ Run Model Prediction", use_container_width=True)

    if predict_btn:
        if not st.session_state.pickup_coords:
            st.markdown('<div class="instruction-box ins-red">⚠️ Please drop a pickup pin first.</div>', unsafe_allow_html=True)
        elif not st.session_state.dropoff_coords:
            st.markdown('<div class="instruction-box ins-red">⚠️ Please drop a dropoff pin first.</div>', unsafe_allow_html=True)
        else:
            with st.spinner("AI is calculating optimal route dynamics..."):
                # دمج واستدعاء خط إنتاج الموديل الذكي التنبؤي
                pickup_datetime = f"{pickup_date} {pickup_time}"
                sample = pd.DataFrame([{
                    "pickup_datetime": pickup_datetime,
                    "pickup_latitude": st.session_state.pickup_coords[0],
                    "pickup_longitude": st.session_state.pickup_coords[1],
                    "dropoff_latitude": st.session_state.dropoff_coords[0],
                    "dropoff_longitude": st.session_state.dropoff_coords[1],
                    "passenger_count": passenger_count,
                    "store_and_fwd_flag": "N",
                    "vendor_id": vendor_id
                }])

                df = feature_engineering(sample)
                df = encode_categorical(df)
                df = align_features(df, features)

                result = predict(model, df)
                duration_seconds = result[0]
                duration_minutes = duration_seconds / 60
                duration_hours = duration_minutes / 60

                # كارت النتيجة النهائي الكبير والواضح جداً للمستخدم بالألوان الفاتحة والمريحة
                st.markdown(f"""
                    <div class="ux-card" style="border: 2px solid #0284C7; background-color: #F0F9FF;">
                        <div class="ux-card-title" style="color: #0369A1; font-weight:700;">Predicted Travel Time</div>
                        <div class="ux-card-value" style="color: #0369A1; font-size: 2.3rem; letter-spacing:-0.04em;">{duration_minutes:.1f} <span style="font-size:1.2rem; font-weight:500;">mins</span></div>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                        <div class="ux-card">
                            <div class="ux-card-title">In Seconds</div>
                            <div class="ux-card-value" style="font-size:1.15rem;">{duration_seconds:.0f}s</div>
                        </div>
                        <div class="ux-card">
                            <div class="ux-card-title">In Hours</div>
                            <div class="ux-card-value" style="font-size:1.15rem;">{duration_hours:.2f}h</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)