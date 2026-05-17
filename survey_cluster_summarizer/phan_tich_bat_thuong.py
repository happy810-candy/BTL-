import streamlit as st
import pandas as pd
import datetime
import io

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

# --- THANH MENU BÊN TRÁI ---
st.sidebar.header("📁 Cấu hình & Dữ liệu")
uploaded_file = st.sidebar.file_uploader("Tải lên file dữ liệu của bạn (CSV hoặc Excel)", type=["csv", "xlsx"])

# Độ nhạy phát hiện bất thường bằng IQR
iqr_factor = st.sidebar.slider("Độ nhạy phát hiện bất thường (Hệ số IQR)", 1.0, 3.5, 1.5, step=0.1,
                               help="Hệ số càng thấp thì càng bắt được nhiều lỗi nhỏ. Mặc định là 1.5.")

if uploaded_file is not None:
    # 1. Đọc file an toàn tuyệt đối
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, on_bad_lines='skip', sep=None, engine='python')
        else:
            df = pd.read_excel(uploaded_file)
        # Làm sạch tên cột
        df.columns = df.columns.str.strip()
    except Exception as e:
        st.error(f"Không thể đọc file dữ liệu. Vui lòng kiểm tra lại định dạng! Lỗi: {e}")
        st.stop()

    if not df.empty:
        # Cố gắng chuyển Amount về dạng số
        if 'Amount' in df.columns:
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

        # Tìm cột số để phân tích
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        if not numeric_cols:
            st.error("❌ File tải lên không chứa cột dữ liệu số (Amount, Số tiền...) hợp lệ!")
        else:
            target_col = 'Amount' if 'Amount' in numeric_cols else numeric_cols[0]

            # Xử lý ô rỗng
            df[target_col] = df[target_col].fillna(0)

            # 2. Thuật toán IQR tính toán trực tiếp bằng Pandas (Không lo lỗi thực thi)
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

            # 3. CHIA GIAO DIỆN THÀNH CÁC TAB
            tab1, tab2, tab3 = st.tabs(["📊 Tổng Quan Số Liệu", "🚨 Các Dòng Bất Thường", "📋 Xuất Báo Cáo"])

            # --- TAB 1: TỔNG QUAN ---
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

            # --- TAB 2: ĐIỂM BẤT THƯỜNG ---
            with tab2:
                st.subheader("🚨 Danh Sách Bản Ghi Nghi Vấn / Bất Thường")
                st.markdown(f"Hệ thống phát hiện **{anomaly_records}** dòng dữ liệu vượt ngoài ngưỡng an toàn toán học.")

                if anomaly_records > 0:
                    # Hiển thị bảng dữ liệu lỗi tách riêng ra
                    st.dataframe(anomalies.drop(columns=['Is_Anomaly'], errors='ignore'), use_container_width=True)
                else:
                    st.success("🎉 Tuyệt vời! Không phát hiện điểm dữ liệu bất thường nào.")

            # --- TAB 3: XUẤT BÁO CÁO ---
            with tab3:
                st.subheader("📋 Báo Cáo Tóm Tắt Động")

                report_text = f"""
                ### BÁO CÁO PHÂN TÍCH TỰ ĐỘNG
                * **Ngày lập báo cáo:** {datetime.date.today().strftime('%d/%m/%Y')}
                * **Tổng số dòng dữ liệu đã quét:** {total_records:,} dòng.
                * **Số lượng điểm bất thường phát hiện:** {anomaly_records} dòng (Chiếm {anomaly_rate:.2f}% tổng số mẫu).
                * **Phương pháp kiểm tra:** Sử dụng Khoảng tứ phân vị (IQR) trên cột `{target_col}` với hệ số nhạy {iqr_factor}.

                #### 🔍 Đánh giá nhanh:
                1. Giá trị lớn nhất xuất hiện trong file: **${df[target_col].max():,.2f}**.
                2. Các dòng tiền / giao dịch có giá trị bất thường nằm riêng biệt tại **Tab số 2**, đề xuất bộ phận chức năng rà soát và đối chiếu lại chứng từ gốc.
                """
                st.markdown(report_text)

                if anomaly_records > 0:
                    try:
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            anomalies.drop(columns=['Is_Anomaly'], errors='ignore').to_excel(writer, index=False, sheet_name='Báo_cáo_bất_thường')
                        excel_data = output.getvalue()

                        st.download_button(
                            label="📥 Tải xuống File Excel Báo Cáo Lỗi",
                            data=excel_data,
                            file_name=f"Bao_cao_bat_thuong_{datetime.date.today()}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    except Exception as e:
                        st.error(f"Không thể xuất file Excel: {e}")
    else:
        st.warning("File của bạn trống không hoặc lỗi cấu trúc dòng.")
else:
    st.info("👋 Chào mừng bạn! Hãy tải file `.csv` hoặc file Excel dữ liệu của bạn ở thanh menu bên trái để hệ thống tự động chạy báo cáo nhé.")