import streamlit as st
import pandas as pd
import datetime

# --- CẤU HÌNH TRANG WEB CHUẨN ---
st.set_page_config(
    page_title="Hệ Thống Phân Tích Dữ Liệu & Phát Hiện Bất Thường",
    page_icon="📊",
    layout="wide"
)

# --- GIAO DIỆN ĐẸP HIỆN ĐẠI ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    h1 { color: #1E3A8A; font-family: 'Segoe UI', sans-serif; font-weight: 700; text-align: center; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 5px solid #1E3A8A; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Hệ Thống Phân Tích Dữ Liệu & Phát Hiện Bất Thường")
st.markdown("<p style='text-align: center; color: #666;'>Tải file dữ liệu lên để tự động kiểm tra và xuất báo cáo thông minh</p>", unsafe_allow_html=True)
st.write("---")

# --- THANH MENU BÊN TRÁI (SIDEBAR) ---
st.sidebar.header("🔑 Xác Thực Hệ Thống")

# 1. Ô BẮT BUỘC NHẬP GEMINI API KEY
api_key = st.sidebar.text_input(
    "Nhập Gemini API Key của bạn:",
    type="password",
    placeholder="AIzaSy...",
    help="Vui lòng nhập API Key để kích hoạt toàn bộ tính năng của hệ thống."
)

st.sidebar.write("---")
st.sidebar.header("📁 Cấu hình & Dữ liệu")

# Kéo slider chọn độ nhạy IQR
iqr_factor = st.sidebar.slider("Độ nhạy phát hiện bất thường (Hệ số IQR)", 1.0, 3.5, 1.5, step=0.1)

# Nút tải file lên
uploaded_file = st.sidebar.file_uploader("Tải lên file dữ liệu của bạn (CSV hoặc Excel)", type=["csv", "xlsx"])


# --- KIỂM TRA ĐIỀU KIỆN API KEY ---
if not api_key:
    # Nếu chưa nhập API Key, khóa toàn bộ giao diện chính và hiển thị cảnh báo màu vàng
    st.warning("⚠️ **HỆ THỐNG ĐANG KHÓA:** Vui lòng nhập **Gemini API Key** ở thanh menu bên trái để mở khóa và sử dụng ứng dụng!")
    st.info("💡 *Mẹo: Bạn có thể lấy API Key miễn phí từ Google AI Studio.*")

else:
    # NẾU ĐÃ NHẬP API KEY -> CHO PHÉP CHẠY ỨNG DỤNG BÌNH THƯỜNG
    st.sidebar.success("✅ Đã ghi nhận API Key!")

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, on_bad_lines='skip', sep=None, engine='python')
            else:
                df = pd.read_excel(uploaded_file)
            df.columns = df.columns.str.strip()
        except Exception as e:
            st.error(f"Không thể đọc file dữ liệu. Lỗi: {e}")
            st.stop()

        if not df.empty:
            if 'Amount' in df.columns:
                df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
            
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
            if not numeric_cols:
                st.error("❌ File tải lên không chứa cột dữ liệu số hợp lệ!")
            else:
                target_col = 'Amount' if 'Amount' in numeric_cols else numeric_cols[0]
                df[target_col] = df[target_col].fillna(0)
                
                # Thuật toán IQR
                q1 = df[target_col].quantile(0.25)
                q3 = df[target_col].quantile(0.75)
                iqr = q3 - q1
                
                lower_bound = q1 - iqr_factor * iqr
                upper_bound = q3 + iqr_factor * iqr
                
                df['Is_Anomaly'] = (df[target_col] < lower_bound) | (df[target_col] > upper_bound)
                anomalies = df[df['Is_Anomaly'] == True].copy()
                
                total_records = len(df)
                anomaly_records = len(anomalies)
                anomaly_rate = (anomaly_records / total_records * 100) if total_records > 0 else 0
                
                tab1, tab2, tab3 = st.tabs(["📊 Tổng Quan Số Liệu", "🚨 Các Dòng Bất Thường", "📋 Xuất Báo Cáo"])
                
                with tab1:
                    st.subheader("📌 Chỉ Số Khóa (KPIs)")
                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                    kpi1.metric("Tổng Số Giao Dịch", f"{total_records:,}")
                    kpi2.metric(f"Tổng Lượng {target_col}", f"${df[target_col].sum():,.2f}")
                    kpi3.metric(f"Mức Trung Bình", f"${df[target_col].mean():,.2f}")
                    kpi4.metric("Số Ca Bất Thường", f"{anomaly_records} ca", delta=f"{anomaly_rate:.2f}% dữ liệu", delta_color="inverse")
                    
                    st.write("---")
                    st.subheader("📋 Toàn bộ bảng dữ liệu gốc:")
                    st.dataframe(df.drop(columns=['Is_Anomaly'], errors='ignore'), use_container_width=True)

                with tab2:
                    st.subheader("🚨 Danh Sách Bản Ghi Nghi Vấn / Bất Thường")
                    if anomaly_records > 0:
                        st.dataframe(anomalies.drop(columns=['Is_Anomaly'], errors='ignore'), use_container_width=True)
                    else:
                        st.success("🎉 Tuyệt vời! Không phát hiện điểm dữ liệu bất thường nào.")

                with tab3:
                    st.subheader("📋 Báo Cáo Tóm Tắt Động")
                    report_text = f"""
                    ### BÁO CÁO PHÂN TÍCH TỰ ĐỘNG
                    * **Ngày lập báo cáo:** {datetime.date.today().strftime('%d/%m/%Y')}
                    * **Tổng số dòng dữ liệu đã quét:** {total_records:,} dòng.
                    * **Số lượng điểm bất thường phát hiện:** {anomaly_records} dòng ({anomaly_rate:.2f}% tổng số mẫu).
                    """
                    st.markdown(report_text)
                    
                    if anomaly_records > 0:
                        try:
                            csv_data = anomalies.drop(columns=['Is_Anomaly'], errors='ignore').to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Tải xuống File CSV Báo Cáo Lỗi",
                                data=csv_data,
                                file_name=f"Bao_cao_bat_thuong_{datetime.date.today()}.csv",
                                mime="text/csv"
                            )
                        except Exception as e:
                            st.error(f"Không thể xuất file: {e}")
    else:
        st.info("👋 Hãy tải file dữ liệu của bạn ở thanh menu bên trái để hệ thống tự động chạy báo cáo nhé.")