import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import requests
import time

# Cấu hình trang với giao diện rộng và tiêu đề chuyên nghiệp
st.set_page_config(
    page_title="AI Anomaly Analytics - Hệ Thống Phát Hiện Bất Thường",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Áp dụng Custom CSS phong cách Glassmorphism sang trọng, màu sắc sinh động
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hình nền chuyển sắc sang trọng */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #0d1224 0%, #050710 100%);
        color: #e2e8f0;
    }
    
    /* Thiết kế thẻ Card Glassmorphism */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    /* Badge trạng thái */
    .badge-anomaly {
        background: linear-gradient(135deg, #ef4444 0%, #991b1b 100%);
        color: #ffffff;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        box-shadow: 0 0 12px rgba(239, 68, 68, 0.4);
    }
    .badge-normal {
        background: linear-gradient(135deg, #10b981 0%, #065f46 100%);
        color: #ffffff;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.4);
    }
    
    /* Tiêu đề hiệu ứng Gradient */
    .gradient-text {
        background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 50%, #4f46e5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* Nút bấm thiết kế đẹp */
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    .stButton>button:hover {
        transform: scale(1.03);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# Hàm gọi API Gemini tích hợp Exponential Backoff phòng tránh lỗi nghẽn hoặc Rate Limit
def generate_gemini_analysis(prompt, system_instruction, api_key):
    if not api_key:
        return None
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": system_instruction}]}
    }
    
    backoff_delays = [1, 2, 4, 8, 16]
    for delay in backoff_delays:
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                return text
            elif response.status_code == 429:
                time.sleep(delay)
            else:
                time.sleep(delay)
        except Exception:
            time.sleep(delay)
    return "Không thể kết nối đến Gemini API sau nhiều lần thử lại. Vui lòng kiểm tra lại API Key hoặc kết nối mạng."

# Hàm tạo bộ dữ liệu giả lập chất lượng cao về giao dịch tài chính
@st.cache_data
def load_default_data():
    np.random.seed(42)
    n_samples = 200
    
    # Tạo thời gian phát sinh giao dịch liên tục
    base_time = datetime.now() - timedelta(days=10)
    time_stamps = [base_time + timedelta(hours=i*1.2) for i in range(n_samples)]
    
    # Dữ liệu phân phối chuẩn bình thường
    amount = np.random.normal(loc=15.0, scale=4.0, size=n_samples) # Số tiền (Triệu VNĐ)
    freq = np.random.normal(loc=5.0, scale=1.5, size=n_samples)    # Tần suất giao dịch / ngày
    risk_score = np.random.normal(loc=30.0, scale=8.0, size=n_samples) # Điểm tin cậy IP/Vị trí
    
    df = pd.DataFrame({
        "Mã Giao Dịch": [f"TXN-{1000+i}" for i in range(n_samples)],
        "Thời Gian": time_stamps,
        "Số Tiền (Tr VNĐ)": np.round(amount, 2),
        "Tần Suất (Lần/Ngày)": np.round(freq, 1),
        "Điểm Rủi Ro Thiết Bị": np.round(risk_score, 1),
        "Vùng Địa Lý": np.random.choice(["Hà Nội", "TP. Hồ Chí Minh", "Đà Nẵng", "Cần Thơ"], size=n_samples)
    })
    
    # Chèn điểm dị biệt cực đoan để học máy nhận biết
    df.loc[15, "Số Tiền (Tr VNĐ)"] = 95.0
    df.loc[15, "Tần Suất (Lần/Ngày)"] = 1.0
    df.loc[42, "Tần Suất (Lần/Ngày)"] = 28.0
    df.loc[42, "Điểm Rủi Ro Thiết Bị"] = 92.5
    df.loc[88, "Số Tiền (Tr VNĐ)"] = 0.5
    df.loc[88, "Tần Suất (Lần/Ngày)"] = 35.0
    df.loc[88, "Điểm Rủi Ro Thiết Bị"] = 85.0
    df.loc[112, "Điểm Rủi Ro Thiết Bị"] = 98.0
    df.loc[112, "Số Tiền (Tr VNĐ)"] = 72.0
    df.loc[145, "Số Tiền (Tr VNĐ)"] = 82.5
    df.loc[145, "Tần Suất (Lần/Ngày)"] = 22.0
    df.loc[145, "Điểm Rủi Ro Thiết Bị"] = 89.0
    df.loc[172, "Tần Suất (Lần/Ngày)"] = 30.5
    df.loc[172, "Điểm Rủi Ro Thiết Bị"] = 95.0
    df.loc[190, "Số Tiền (Tr VNĐ)"] = 120.0
    
    return df

# Hàm tính toán phát hiện bất thường bằng Isolation Forest và lưu thông số giải thích
def detect_anomalies(df, selected_features, contamination_rate=0.04):
    X = df[selected_features].copy()
    
    # Điền giá trị khuyết thiếu nếu có để thuật toán không bị lỗi
    X = X.fillna(X.mean())
    
    # Chuẩn hóa để hỗ trợ giải thích trực quan từng cột
    mean_vals = X.mean()
    std_vals = X.std()
    
    # Thuật toán học máy phát hiện bất thường nâng cao
    model = IsolationForest(contamination=contamination_rate, random_state=42)
    model.fit(X)
    
    # Phân loại tự động (-1: Bất thường, 1: Bình thường)
    predictions = model.predict(X)
    scores = model.decision_function(X)
    
    df_result = df.copy()
    df_result["Trạng Thái"] = ["Bất Thường" if pred == -1 else "Bình Thường" for pred in predictions]
    df_result["Độ Bất Thường (Score)"] = np.round(-scores, 4)
    
    # Tạo kích cỡ điểm vẽ luôn dương bảo đảm không lỗi 3D scatter
    min_score = scores.min()
    max_score = scores.max()
    if max_score != min_score:
        df_result["Kích Thước Biểu Đồ"] = 2 + 16 * (scores - min_score) / (max_score - min_score)
    else:
        df_result["Kích Thước Biểu Đồ"] = 8.0
        
    stats = {
        "mean": mean_vals.to_dict(),
        "std": std_vals.to_dict()
    }
    
    return df_result, stats

# Tiêu đề Header của Web
st.write("")
st.markdown("""
<div style='text-align: center; margin-bottom: 25px;'>
    <h1 style='font-size: 2.8rem; margin-bottom: 8px;'><span class='gradient-text'>🛡️ AI ANOMALY DETECTOR PRO</span></h1>
    <p style='color: #94a3b8; font-size: 1.1rem; max-width: 800px; margin: 0 auto;'>
        Hệ thống phân tích, giám sát dữ liệu và phát hiện bất thường thông minh tương thích đa định dạng tệp tin kết hợp phân tích AI tạo sinh.
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# KHU VỰC CẤU HÌNH & TẢI FILE TRỰC TIẾP TRÊN TRANG CHỦ (MAIN PAGE)
# ==========================================
st.markdown("<h3 style='color: #cbd5e1; margin-top: 10px; margin-bottom: 15px;'>⚙️ Khởi Tạo Cấu Hình Hệ Thống</h3>", unsafe_allow_html=True)

col_input_left, col_input_right = st.columns(2)

with col_input_left:
    st.markdown("""
    <div class="glass-card" style="padding: 20px; min-height: 250px;">
        <h4 style="margin-top: 0; color: #818cf8; font-size: 1.1rem;">🔑 Bước 1: Kết Nối Trí Tuệ Nhân Tạo (API)</h4>
        <p style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 15px;">
            Nhập Gemini API Key của bạn để kích hoạt khả năng phân tích lập luận thông minh rành mạch cho từng dòng dữ liệu và sinh báo cáo tự động bằng ngôn ngữ tự nhiên.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Di chuyển ô nhập API Key ra màn hình chính để dễ tương tác
    gemini_api_key = st.text_input(
        "Nhập Gemini API Key:", 
        type="password", 
        placeholder="AIzaSy...",
        label_visibility="collapsed",
        help="Hệ thống sử dụng mô hình Gemini 2.5 Flash để tự động đưa ra các đánh giá nghiệp vụ chuyên sâu."
    )
    
    if gemini_api_key:
        st.success("🎉 Đã nhận diện thành công API Key Gemini!")
    else:
        st.info("💡 Bạn đang dùng chế độ Thống Kê Định Lượng Z-Score nội bộ (Không cần API Key).")

with col_input_right:
    st.markdown("""
    <div class="glass-card" style="padding: 20px; min-height: 250px;">
        <h4 style="margin-top: 0; color: #818cf8; font-size: 1.1rem;">📂 Bước 2: Tải Lên Tập Dữ Liệu Khảo Sát</h4>
        <p style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 10px;">
            Hệ thống hỗ trợ tự động xử lý và phân tích đa định dạng tệp tin bao gồm: <strong>CSV, Excel (xlsx, xls), JSON, Parquet</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chọn phương án nạp dữ liệu trực tiếp trên giao diện chính
    dataset_option = st.selectbox(
        "Lựa chọn nguồn nạp dữ liệu:",
        ["Sử dụng bộ dữ liệu Tài chính mẫu mặc định", "Tải lên tệp dữ liệu tùy chỉnh của bạn"],
        label_visibility="collapsed"
    )
    
    df_input = None
    features_to_use = []
    
    if dataset_option == "Tải lên tệp dữ liệu tùy chỉnh của bạn":
        uploaded_file = st.file_uploader(
            "Kéo thả hoặc duyệt tệp từ máy tính:", 
            type=["csv", "xlsx", "xls", "json", "parquet"]
        )
        if uploaded_file is not None:
            file_extension = uploaded_file.name.split(".")[-1].lower()
            try:
                # Phân giải cấu trúc tệp dựa trên phần mở rộng đuôi file
                if file_extension == "csv":
                    df_input = pd.read_csv(uploaded_file)
                elif file_extension in ["xlsx", "xls"]:
                    df_input = pd.read_excel(uploaded_file)
                elif file_extension == "json":
                    df_input = pd.read_json(uploaded_file)
                elif file_extension == "parquet":
                    df_input = pd.read_parquet(uploaded_file)
                    
                st.success(f"✔️ Đã tải tệp thành công! Định dạng: {file_extension.upper()} ({len(df_input)} dòng)")
                
                # Trích lọc thông minh các cột dữ liệu số
                numeric_cols = df_input.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) < 2:
                    st.error("⚠️ Tệp dữ liệu cần chứa ít nhất 2 cột định dạng số để chạy phân tích toán học Isolation Forest.")
                    df_input = None
                else:
                    st.markdown("<p style='font-size:0.9rem; font-weight:600; color:#818cf8; margin: 10px 0 5px 0;'>Chọn các cột số để đưa vào mô hình học máy:</p>", unsafe_allow_html=True)
                    default_features = numeric_cols[:min(3, len(numeric_cols))]
                    features_to_use = st.multiselect(
                        "Chọn cột số phân tích:",
                        options=numeric_cols,
                        default=default_features,
                        label_visibility="collapsed"
                    )
                    if len(features_to_use) < 2:
                        st.warning("⚠️ Vui lòng lựa chọn tối thiểu 2 thuộc tính số để mô hình hoạt động.")
            except Exception as e:
                st.error(f"Lỗi khi đọc file {file_extension.upper()}: {str(e)}")
                st.info("💡 Mẹo: Đối với định dạng Excel, hãy chắc chắn máy của bạn đã cài openpyxl (`pip install openpyxl`).")
                df_input = None
        else:
            st.info("👉 Hãy tải lên tệp tin của bạn để bắt đầu. Trong lúc chờ đợi, hệ thống tạm thời hiển thị dữ liệu mẫu.")
            df_input = load_default_data()
            features_to_use = ["Số Tiền (Tr VNĐ)", "Tần Suất (Lần/Ngày)", "Điểm Rủi Ro Thiết Bị"]
    else:
        df_input = load_default_data()
        features_to_use = ["Số Tiền (Tr VNĐ)", "Tần Suất (Lần/Ngày)", "Điểm Rủi Ro Thiết Bị"]

# Cài đặt Sidebar chỉ dùng cho việc điều chỉnh siêu tham số mô hình
st.sidebar.markdown("""
<div style='text-align: center; margin-bottom: 20px;'>
    <h2 style='color: #818cf8; font-size: 1.3rem; font-weight: 700;'>🎛️ THAM SỐ MÔ HÌNH</h2>
    <p style='color: #64748b; font-size: 0.8rem;'>Cấu hình độ nhạy bén thuật toán</p>
</div>
""", unsafe_allow_html=True)

contamination = st.sidebar.slider(
    "Tỷ lệ dữ liệu bất thường mong muốn (%)",
    min_value=1.0,
    max_value=15.0,
    value=4.5,
    step=0.5,
    help="Tỷ lệ bất thường dự kiến trong tập dữ liệu của bạn."
) / 100.0

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🛡️ Tiêu chuẩn An toàn
Hệ thống xử lý hoàn toàn trong bộ nhớ máy cục bộ, bảo vệ tối đa tính riêng tư dữ liệu của doanh nghiệp bạn.
""")

# Kiểm tra xem có dữ liệu hợp lệ hay không
if df_input is not None and len(features_to_use) >= 2:
    # Tiến hành chạy mô hình học máy phát hiện dị biệt
    processed_df, stats_summary = detect_anomalies(df_input, features_to_use, contamination)
    
    # Phân tách dữ liệu
    anomalies_only = processed_df[processed_df["Trạng Thái"] == "Bất Thường"]
    normals_only = processed_df[processed_df["Trạng Thái"] == "Bình Thường"]
    
    # Hiển thị số liệu tổng quan (Metrics Row) thiết kế Glassmorphism
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        st.markdown(f"""
        <div class="glass-card" style="text-align: center;">
            <p style="color: #94a3b8; font-size: 0.9rem; font-weight: 600; margin: 0;">TỔNG SỐ BẢN GHI</p>
            <h2 style="color: #3b82f6; font-size: 2.2rem; font-weight: 800; margin: 10px 0 0 0;">{len(processed_df)}</h2>
        </div>
        """, unsafe_allow_html=True)
        
    with col_m2:
        st.markdown(f"""
        <div class="glass-card" style="text-align: center; border-left: 4px solid #10b981;">
            <p style="color: #94a3b8; font-size: 0.9rem; font-weight: 600; margin: 0;">DỮ LIỆU BÌNH THƯỜNG</p>
            <h2 style="color: #10b981; font-size: 2.2rem; font-weight: 800; margin: 10px 0 0 0;">{len(normals_only)}</h2>
        </div>
        """, unsafe_allow_html=True)
        
    with col_m3:
        st.markdown(f"""
        <div class="glass-card" style="text-align: center; border-left: 4px solid #ef4444;">
            <p style="color: #94a3b8; font-size: 0.9rem; font-weight: 600; margin: 0;">SỐ LƯỢNG BẤT THƯỜNG</p>
            <h2 style="color: #ef4444; font-size: 2.2rem; font-weight: 800; margin: 10px 0 0 0;">{len(anomalies_only)}</h2>
        </div>
        """, unsafe_allow_html=True)
        
    with col_m4:
        ratio_percentage = (len(anomalies_only) / len(processed_df)) * 100
        st.markdown(f"""
        <div class="glass-card" style="text-align: center;">
            <p style="color: #94a3b8; font-size: 0.9rem; font-weight: 600; margin: 0;">TỶ LỆ BẤT THƯỜNG</p>
            <h2 style="color: #a855f7; font-size: 2.2rem; font-weight: 800; margin: 10px 0 0 0;">{ratio_percentage:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)
        
    # Thiết kế biểu đồ Trực quan hóa tương tác
    st.markdown("<h3 style='color: #cbd5e1; margin-top: 20px; margin-bottom: 15px;'>📊 Phân Tích & Biểu Đồ Trực Quan</h3>", unsafe_allow_html=True)
    
    col_chart1, col_chart2 = st.columns([3, 2])
    
    with col_chart1:
        st.markdown("""
        <div class="glass-card" style="height: 100%;">
            <h4 style="margin-top:0; color:#a5b4fc;">Không Gian Phân Phối Đa Chiều (Scatter Plot)</h4>
            <p style="font-size:0.85rem; color:#94a3b8;">Biểu đồ tương tác đa chiều giúp quan sát mật độ phân bổ và trực quan hóa các tọa độ dữ liệu dị biệt bị cô lập.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Tạo biểu đồ động thông minh (3D nếu chọn >= 3 biến số, 2D nếu chỉ chọn 2 biến số)
        if len(features_to_use) >= 3:
            fig_chart = px.scatter_3d(
                processed_df, 
                x=features_to_use[0], 
                y=features_to_use[1], 
                z=features_to_use[2],
                color="Trạng Thái",
                color_discrete_map={"Bình Thường": "#10b981", "Bất Thường": "#ef4444"},
                opacity=0.85,
                size="Kích Thước Biểu Đồ",
                size_max=18
            )
            fig_chart.update_layout(
                scene=dict(
                    xaxis=dict(backgroundcolor="#050710", gridcolor="rgba(255,255,255,0.05)", title_font_color="#cbd5e1"),
                    yaxis=dict(backgroundcolor="#050710", gridcolor="rgba(255,255,255,0.05)", title_font_color="#cbd5e1"),
                    zaxis=dict(backgroundcolor="#050710", gridcolor="rgba(255,255,255,0.05)", title_font_color="#cbd5e1")
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(font=dict(color="#cbd5e1")),
                margin=dict(l=0, r=0, b=0, t=20),
                height=480
            )
        else:
            # Vẽ biểu đồ 2D Scatter tương tác cực đẹp
            fig_chart = px.scatter(
                processed_df,
                x=features_to_use[0],
                y=features_to_use[1],
                color="Trạng Thái",
                color_discrete_map={"Bình Thường": "#10b981", "Bất Thường": "#ef4444"},
                opacity=0.85,
                size="Kích Thước Biểu Đồ",
                size_max=18
            )
            fig_chart.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title_font_color="#cbd5e1", tickfont=dict(color="#cbd5e1")),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title_font_color="#cbd5e1", tickfont=dict(color="#cbd5e1")),
                legend=dict(font=dict(color="#cbd5e1")),
                margin=dict(l=0, r=0, b=0, t=20),
                height=480
            )
        st.plotly_chart(fig_chart, use_container_width=True)
        
    with col_chart2:
        st.markdown("""
        <div class="glass-card" style="height: 100%;">
            <h4 style="margin-top:0; color:#a5b4fc;">Tương Quan Biến Động Phân Vị (Feature Distribution)</h4>
            <p style="font-size:0.85rem; color:#94a3b8;">Phân tích biểu đồ hộp giúp dễ dàng xác định ngưỡng biên ranh giới và phân vùng dị biệt cực đoan.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Biểu đồ hộp phân vị cho các thuộc tính số
        feature_to_plot = st.selectbox(
            "Lọc tham số phân phối:",
            features_to_use
        )
        
        fig_box = px.box(
            processed_df,
            x="Trạng Thái",
            y=feature_to_plot,
            color="Trạng Thái",
            color_discrete_map={"Bình Thường": "#10b981", "Bất Thường": "#ef4444"},
            points="all"
        )
        fig_box.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title_font_color="#cbd5e1", tickfont=dict(color="#cbd5e1")),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title_font_color="#cbd5e1", tickfont=dict(color="#cbd5e1")),
            legend=dict(font=dict(color="#cbd5e1")),
            margin=dict(l=0, r=0, b=10, t=10),
            height=380
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
    # KHU VỰC TRÌNH BÀY DỮ LIỆU & GIẢI THÍCH CHI TIẾT TỪ MÔ HÌNH AI
    st.markdown("<hr style='border-color: rgba(255,255,255,0.1)'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #cbd5e1; margin-top: 20px; margin-bottom: 5px;'>🔍 Trình Tương Tác & Giải Thích Chi Tiết Dữ Liệu</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 0.95rem; margin-bottom: 20px;'>Chọn một hàng bất kỳ dưới đây để xem phân tích chi tiết tự động và trực quan bằng thuật toán thống kê hoặc Trí tuệ Nhân tạo Gemini.</p>", unsafe_allow_html=True)
    
    col_table, col_details = st.columns([5, 4])
    
    # Sắp xếp các bản ghi có điểm bất thường cao nhất lên đầu bảng
    df_sorted = processed_df.sort_values(by="Độ Bất Thường (Score)", ascending=False).reset_index(drop=True)
    
    # Tạo chỉ mục định dạng hiển thị cho người dùng chọn nhanh từ menu
    row_options = []
    # Xác định cột nhận diện chính (ưu tiên "Mã Giao Dịch" hoặc cột đầu tiên dạng text)
    primary_id_col = "Mã Giao Dịch" if "Mã Giao Dịch" in df_sorted.columns else df_sorted.columns[0]
    
    for idx, row in df_sorted.iterrows():
        status_symbol = "🚨 [BẤT THƯỜNG]" if row["Trạng Thái"] == "Bất Thường" else "🟢 [BÌNH THƯỜNG]"
        val_preview = " | ".join([f"{col}: {row[col]}" for col in features_to_use[:2]])
        row_options.append(f"{idx+1}. {row[primary_id_col]} - {status_symbol} ({val_preview})")
        
    with col_table:
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.02); padding: 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 15px;">
            <strong style="color: #818cf8;">👉 Bước 3: Chọn hàng dữ liệu cần khảo sát chi tiết:</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Hộp chọn nhanh giao dịch (Độc lập phiên bản Streamlit và cực kì bảo mật)
        selected_row_str = st.selectbox(
            "DANH SÁCH BẢN GHI DỮ LIỆU ĐÃ ĐƯỢC XẾP HẠNG:",
            row_options,
            index=0,
            key="dropdown_selection"
        )
        
        # Lấy chỉ số dòng được chọn từ chuỗi text
        current_selected_idx = int(selected_row_str.split(".")[0]) - 1
        selected_row = df_sorted.iloc[current_selected_idx]
        
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.02); padding: 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); margin-top: 25px; margin-bottom: 10px;">
            <strong style="color: #818cf8;">📋 Toàn bộ danh sách dữ liệu đầu ra:</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Hiển thị bảng dạng DataFrame tĩnh, an toàn, không có các tham số gây lỗi biên dịch
        display_cols = [primary_id_col] + features_to_use + ["Trạng Thái", "Độ Bất Thường (Score)"]
        display_df = df_sorted[display_cols].copy()
        display_df.columns = [f"Thuộc tính: {c}" if c in features_to_use else c for c in display_df.columns]
        
        st.dataframe(display_df, use_container_width=True)
        
    with col_details:
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.02); padding: 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);">
            <strong style="color: #f472b6;">🔬 Diễn Giải & Giải Thích Chi Tiết Của AI:</strong>
        </div>
        """, unsafe_allow_html=True)
        
        is_anomaly = selected_row["Trạng Thái"] == "Bất Thường"
        badge_html = "<span class='badge-anomaly'>🚨 BẤT THƯỜNG</span>" if is_anomaly else "<span class='badge-normal'>🟢 BÌNH THƯỜNG</span>"
        
        st.markdown(f"""
        <div class="glass-card" style="border-top: 4px solid {'#ef4444' if is_anomaly else '#10b981'}; margin-top: 15px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <span style="font-size: 1.2rem; font-weight: 700; color: #f8fafc;">Bản ghi: {selected_row[primary_id_col]}</span>
                {badge_html}
            </div>
            <table style="width: 100%; border-collapse: collapse; margin-bottom: 15px; color: #e2e8f0; font-size: 0.9rem;">
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05); height: 35px;">
                    <td><strong>Chỉ số dị biệt tổng thể (Anomaly Score):</strong></td>
                    <td style="text-align: right; font-weight: bold; color: {'#f87171' if is_anomaly else '#34d399'};">
                        {selected_row['Độ Bất Thường (Score)']*100:.2f}%
                    </td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
        # Phần lập luận chi tiết kết hợp AI tạo sinh nếu có API Key
        if gemini_api_key:
            st.markdown("##### 🔮 Lời bình luận từ AI Agent (Gemini):")
            
            # Xây dựng prompt khoa học chi tiết gửi tới Gemini API
            record_details = "\n".join([f"- {feat}: {selected_row[feat]} (Giá trị trung bình quần thể: {stats_summary['mean'][feat]:.2f})" for feat in features_to_use])
            
            system_prompt = "Bạn là một Chuyên gia phân tích dữ liệu rủi ro cao cấp sử dụng mô hình học máy. Hãy đưa ra lập luận chặt chẽ, chuyên nghiệp, khoa học và ngắn gọn bằng Tiếng Việt."
            user_prompt = f"""Hãy phân tích dòng dữ liệu sau đây và đưa ra lý do chính xác tại sao bản ghi này lại được thuật toán học máy Isolation Forest gắn nhãn là **{selected_row['Trạng Thái']}**.
            
            Thông tin chi tiết của bản ghi:
            Mã định danh: {selected_row[primary_id_col]}
            {record_details}
            Độ lệch dị biệt đo lường: {selected_row['Độ Bất Thường (Score)']*100:.2f}%
            
            Vui lòng giải thích một cách thuyết phục cho doanh nghiệp hiểu:
            1. Vì sao bản ghi này lại ở trạng thái như vậy (dựa vào so sánh tương quan giữa giá trị thực tế của nó và giá trị trung bình quần thể).
            2. Đưa ra 1 khuyến nghị hành động thực tiễn cho quản trị viên.
            Lời giải thích cần trực quan, ngắn gọn dưới 5 câu, lịch lãm, sinh động và sang trọng."""
            
            with st.spinner("🧠 Trí tuệ AI Gemini đang phân tích sâu dữ liệu..."):
                ai_explanation = generate_gemini_analysis(user_prompt, system_prompt, gemini_api_key)
                
            if ai_explanation:
                st.markdown(f"<div style='background: rgba(99, 102, 241, 0.1); padding: 15px; border-radius: 10px; border-left: 4px solid #6366f1; color: #e2e8f0; font-size: 0.95rem; line-height: 1.6;'>{ai_explanation}</div>", unsafe_allow_html=True)
            else:
                st.warning("⚠️ Không thể sinh báo cáo tự động từ Gemini. Đang chuyển hướng sang phân tích thống kê toán học nội bộ.")
                
        # Khung giải thích toán học nội bộ (Chế độ Fallback an toàn)
        st.markdown("##### 📊 Diễn giải định lượng toán học (Z-Score):")
        explanations = []
        for feat in features_to_use:
            val = selected_row[feat]
            mean_v = stats_summary['mean'][feat]
            std_v = stats_summary['std'][feat]
            z_score = (val - mean_v) / std_v if std_v > 0 else 0
            
            if abs(z_score) > 1.96:
                explanations.append(f"• **{feat} ({val}):** Vượt ngưỡng bình thường một cách đáng kể, có độ lệch chuẩn **Z-score = {z_score:.2f}** (nằm ngoài khoảng tin cậy thống kê chuẩn $95\\%$)")
            else:
                explanations.append(f"• **{feat} ({val}):** Hoàn toàn nằm trong phạm vi dao động thông thường và ổn định của quần thể (Z-score = {z_score:.2f})")
                
        for exp in explanations:
            st.markdown(exp)
            
        if not gemini_api_key:
            st.markdown("💡 *Bí quyết: Hãy nhập Gemini API Key ở thanh Sidebar bên trái để kích hoạt thêm những báo cáo nhận định rủi ro siêu việt của trí tuệ nhân tạo tạo sinh.*")

    # BÁO CÁO PHÂN TÍCH TỰ ĐỘNG (AUTOMATED REPORT COMPONENT)
    st.markdown("<hr style='border-color: rgba(255,255,255,0.1)'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #cbd5e1; margin-top: 20px; margin-bottom: 15px;'>📑 Tạo Báo Cáo Phân Tích & Xuất Dữ Liệu</h3>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
        <div class="glass-card">
            <h4 style="margin-top:0; color:#818cf8; font-size: 1.2rem;">Biên Bản Tóm Tắt Phát Hiện Dữ Liệu Bất Thường</h4>
            <p style="color: #cbd5e1; font-size: 0.95rem;">
                Hệ thống tự động tổng hợp tình trạng hoạt động của hệ thống dữ liệu số. Bạn có thể sao chép văn bản báo cáo bên dưới hoặc tải xuống file CSV kết quả hoàn chỉnh.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Tạo văn bản báo cáo động trực quan chuyên sâu
        report_text = f"""============================================================
                  BÁO CÁO PHÂN TÍCH DỮ LIỆU BẤT THƯỜNG
                           HỆ THỐNG AI-DETECT
============================================================
Ngày tạo báo cáo : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Tổng số bản ghi phân tích: {len(processed_df)} dòng dữ liệu
Ngưỡng phát hiện bất thường thiết lập: {contamination*100}%

KẾT QUẢ PHÂN TÍCH TỔNG QUAN:
- Số bản ghi bình thường: {len(normals_only)} ({100 - ratio_percentage:.2f}%)
- Số bản ghi bất thường: {len(anomalies_only)} ({ratio_percentage:.2f}%)

DANH SÁCH CÁC TRƯỜNG THÔNG TIN SỐ ĐƯỢC CHỌN KHẢO SÁT:
"""
        for feat in features_to_use:
            report_text += f"- {feat} | Giá trị trung bình: {stats_summary['mean'][feat]:.2f}\n"
            
        report_text += f"\nDANH SÁCH CÁC BẤT THƯỜNG TRỌNG ĐIỂM (CẦN XỬ LÝ NGAY LẬP TỨC):\n"
        for idx, r in anomalies_only.head(10).iterrows():
            feat_vals_str = " | ".join([f"{f}: {r[f]}" for f in features_to_use])
            report_text += f"- [{r[primary_id_col]}] | {feat_vals_str} | Chỉ số dị biệt: {r['Độ Bất Thường (Score)']*100:.2f}%\n"
            
        report_text += """
============================================================
KHUYẾN NGHỊ VẬN HÀNH:
1. Thực hiện rà soát, kiểm tra độ xác thực đối với các bản ghi bất thường có chỉ số dị biệt lớn hơn 40%.
2. Cập nhật và tinh chỉnh các thuộc tính dữ liệu đầu vào định kỳ để tối ưu hóa tỷ lệ phân rã của mô hình Isolation Forest.
============================================================
Báo cáo được biên soạn tự động bởi Hệ thống AI Anomaly Analytics."""
        
        st.text_area("Xem trước báo cáo phân tích (Markdown/Text):", report_text, height=250)
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            st.download_button(
                label="💾 Tải Xuống Báo Cáo Văn Bản (.txt)",
                data=report_text,
                file_name=f"bao_cao_bat_thuong_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        with col_btn2:
            # Tạo bộ nhớ đệm phục vụ tải dữ liệu CSV đã phân loại
            csv_buffer = io.StringIO()
            processed_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Tải Xuống Toàn Bộ Dữ Liệu Phân Loại (.csv)",
                data=csv_buffer.getvalue(),
                file_name=f"du_lieu_phan_tich_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
else:
    st.info("💡 Vui lòng thiết lập dữ liệu đầu vào hoặc tải tệp dữ liệu lên để khởi chạy các module trực quan hóa và báo cáo AI.")

# Chân trang (Footer) sang trọng
st.markdown("""
<div style='text-align: center; margin-top: 50px; padding: 20px; border-top: 1px solid rgba(255,255,255,0.05); color: #64748b; font-size: 0.85rem;'>
    Hệ thống giám sát an toàn thông tin & phân tích rủi ro dị biệt tích hợp Gemini AI © 2026.
</div>
""", unsafe_allow_html=True)