import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import base64
import io

st.set_page_config(
    page_title="📘 Phát Hiện Giao Dịch Bất Thường",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #1f2937;
        }
        .main-title {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(90deg, #2563eb 0%, #7c3aed 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 1rem;
        }
        .subtitle {
            font-size: 1.05rem;
            color: #4b5563;
            margin-bottom: 1.5rem;
        }
        .card {
            background: #ffffff;
            border-radius: 22px;
            padding: 22px;
            box-shadow: 0 20px 70px rgba(15, 23, 42, 0.06);
            border: 1px solid rgba(148, 163, 184, 0.15);
            margin-bottom: 1.25rem;
        }
        .metric {
            border-left: 4px solid #4338ca;
            padding: 16px;
            border-radius: 18px;
            background: #f8fafc;
        }
        .metric h3 {
            margin: 0;
            font-size: 1rem;
            color: #334155;
        }
        .metric h2 {
            margin: 0.5rem 0 0;
            font-size: 2rem;
            color: #111827;
        }
        .download-button button {
            background: linear-gradient(90deg, #6366f1 0%, #9333ea 100%) !important;
            color: white !important;
            border: none !important;
            box-shadow: 0 12px 20px rgba(99, 102, 241, 0.25);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🧾 Hệ Thống Phát Hiện Giao Dịch Bất Thường</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload sổ cái, phân tích dữ liệu giao dịch, đánh giá mức độ rủi ro và xuất báo cáo chính xác.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Cấu hình & hướng dẫn")
    st.markdown(
        """
        1. Upload file Excel/CSV chứa ít nhất một trong các cột: `Date`, `Account`, `Description`, `Amount`, `Balance`.
        2. Nhập API Key nếu bạn muốn ghi chú rằng có thể dùng dịch vụ xử lý dữ liệu lớn.
        3. Chọn ngưỡng bất thường và xem kết quả.
        """
    )

    api_key = st.text_input("API Key hỗ trợ (tùy chọn)", type="password")
    if api_key:
        st.info("API Key đã nhập. Ứng dụng sẽ đánh dấu dữ liệu lớn để bạn kiểm tra cẩn thận.")

    sample_mode = st.checkbox("Hiển thị mẫu dữ liệu", value=False)
    anomaly_threshold = st.slider(
        "Ngưỡng phát hiện bất thường (z-score)",
        min_value=1.5,
        max_value=5.0,
        value=2.5,
        step=0.1,
    )
    extreme_amount = st.number_input(
        "Ngưỡng giá trị lớn bất thường",
        min_value=0.0,
        value=20000000.0,
        step=1000000.0,
        format="%.0f",
    )
    st.markdown("---")
    st.markdown("### 📌 Lưu ý")
    st.markdown(
        """
        - `Date` nên ở định dạng ngày hoặc thời gian.
        - `Amount` có thể là dương/âm. `Account` giúp logic phân tích.
        - API Key chỉ dùng để ghi chú xử lý dữ liệu lớn; phân tích vẫn chạy local an toàn.
        """
    )

    if sample_mode:
        sample_df = pd.DataFrame(
            {
                "Date": ["2026-05-01", "2026-05-01", "2026-05-02", "2026-05-02", "2026-05-03"],
                "Account": ["Tiền mặt", "Ngân hàng", "Ngân hàng", "Tiền mặt", "Quỹ"],
                "Description": [
                    "Thu tiền bán hàng",
                    "Chuyển khoản nội bộ",
                    "Thanh toán hàng hóa",
                    "Rút tiền mặt lớn",
                    "Chi mua thiết bị",
                ],
                "Amount": [15000000, -12000000, -25000000, -80000000, -3000000],
                "Balance": [15000000, 3000000, -22000000, 58000000, 55000000],
            }
        )
        st.dataframe(sample_df)

uploaded_file = st.file_uploader("Upload Sổ Cái Excel/CSV", type=["xlsx", "csv"])


@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    try:
        if uploaded_file.name.lower().endswith(".xlsx"):
            return pd.read_excel(uploaded_file, engine="openpyxl")
        text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        return pd.read_csv(io.StringIO(text), sep=None, engine="python", encoding="utf-8", on_bad_lines="skip")
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, engine="python", encoding="utf-8", on_bad_lines="skip")


def normalize_columns(df):
    rename_map = {
        "ngày": "Date",
        "date": "Date",
        "ngay": "Date",
        "mô tả": "Description",
        "mo ta": "Description",
        "description": "Description",
        "số tiền": "Amount",
        "so tien": "Amount",
        "amount": "Amount",
        "tài khoản": "Account",
        "tai khoan": "Account",
        "account": "Account",
        "số dư": "Balance",
        "so du": "Balance",
        "balance": "Balance",
        "loại": "Type",
        "loai": "Type",
        "type": "Type",
    }
    cols = {c: rename_map.get(c.strip().lower(), c.strip()) for c in df.columns}
    return df.rename(columns=cols)


def clean_data(df):
    df = df.copy()
    if "Amount" in df.columns:
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "Account" in df.columns:
        df["Account"] = df["Account"].astype(str)
    else:
        df["Account"] = "Chưa xác định"
    if "Description" in df.columns:
        df["Description"] = df["Description"].astype(str)
    else:
        df["Description"] = ""
    return df


def map_weekday(date_series):
    mapping = {
        "Monday": "Thứ hai",
        "Tuesday": "Thứ ba",
        "Wednesday": "Thứ tư",
        "Thursday": "Thứ năm",
        "Friday": "Thứ sáu",
        "Saturday": "Thứ bảy",
        "Sunday": "Chủ nhật",
    }
    names = date_series.dt.day_name()
    return names.map(mapping).fillna("Không rõ")


def detect_anomalies(df, anomaly_threshold, extreme_amount):
    df = df.copy()
    if "Amount" not in df.columns:
        raise ValueError("File phải có cột 'Amount' hoặc 'Số tiền'.")

    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df["AbsAmount"] = df["Amount"].abs()
    df["Date"] = pd.to_datetime(df.get("Date", pd.NaT), errors="coerce")
    df["Account"] = df.get("Account", "Chưa xác định").astype(str)
    df["Description"] = df.get("Description", "").astype(str)
    df["Balance"] = pd.to_numeric(df.get("Balance", 0), errors="coerce").fillna(0)

    df["YearMonth"] = df["Date"].dt.to_period("M").astype(str).fillna("Không rõ")
    df["Weekday"] = map_weekday(df["Date"])
    df["Hour"] = df["Date"].dt.hour.fillna(-1).astype(int)
    df["DayMissing"] = df["Date"].isna()

    monthly = df.groupby("YearMonth")["AbsAmount"].agg(["mean", "std"]).reset_index()
    df = df.merge(monthly, on="YearMonth", how="left")
    df["Std"] = df["std"].replace(0, np.nan).fillna(df["mean"].replace(0, np.nan))
    df["ZScoreGlobal"] = ((df["AbsAmount"] - df["mean"]) / df["Std"]).abs().fillna(0)

    account_stats = df.groupby("Account")["AbsAmount"].agg(["mean", "std"]).reset_index()
    account_stats.columns = ["Account", "AccountMean", "AccountStd"]
    df = df.merge(account_stats, on="Account", how="left")
    df["AccountStd"] = df["AccountStd"].replace(0, np.nan).fillna(df["AccountMean"].replace(0, np.nan))
    df["ZScoreAccount"] = ((df["AbsAmount"] - df["AccountMean"]) / df["AccountStd"]).abs().fillna(0)

    df["ZScore"] = df[["ZScoreGlobal", "ZScoreAccount"]].max(axis=1)
    df["HighValue"] = df["AbsAmount"] >= extreme_amount
    df["Weekend"] = df["Weekday"].isin(["Thứ bảy", "Chủ nhật"])
    df["Night"] = df["Hour"].between(0, 5)
    df["NegativeBalance"] = df["Balance"] < 0
    df["DayMissingRisk"] = df["DayMissing"].astype(int) * 0.6

    keywords = [
        "khẩn",
        "gấp",
        "lỗi",
        "hoàn trả",
        "fraud",
        "rút tiền",
        "chuyển nhanh",
        "thanh toán gấp",
        "sai số",
    ]
    df["KeywordRisk"] = df["Description"].str.lower().apply(lambda t: any(k in t for k in keywords))

    df["AnomalyScore"] = (
        df["ZScore"] / max(anomaly_threshold, 0.1)
        + df["HighValue"].astype(int) * 1.6
        + df["Weekend"].astype(int) * 0.5
        + df["Night"].astype(int) * 0.5
        + df["NegativeBalance"].astype(int) * 1.0
        + df["KeywordRisk"].astype(int) * 0.9
        + df["DayMissingRisk"].astype(float)
    )

    df["AnomalyScore"] = df["AnomalyScore"].fillna(0)
    df["RiskLevel"] = pd.cut(
        df["AnomalyScore"],
        bins=[-1, 1.2, 2.5, 4.0, np.inf],
        labels=["Bình thường", "Cảnh báo", "Nghi ngờ", "Nguy cơ cao"],
    )
    df["RiskLevel"] = df["RiskLevel"].cat.add_categories("Không xác định").fillna("Không xác định")
    df["RiskLevel"] = df["RiskLevel"].astype(str)
    return df


def summarize_analysis(df, extreme_amount):
    total = len(df)
    if total == 0:
        return "File không có giao dịch để phân tích."

    anomalies = df[df["RiskLevel"].isin(["Cảnh báo", "Nghi ngờ", "Nguy cơ cao"])]
    critical = df[df["RiskLevel"] == "Nguy cơ cao"]

    summary = [
        f"Tổng số giao dịch: {total}",
        f"Giao dịch bất thường: {len(anomalies)} ({len(anomalies) / total * 100:.1f}%)",
        f"Giao dịch nguy cơ cao: {len(critical)}",
        f"Giá trị trung bình giao dịch: {df['Amount'].mean():,.0f} VNĐ",
        f"Giao dịch vượt ngưỡng {extreme_amount:,.0f}: {df['HighValue'].sum()}",
    ]
    if df["Date"].isna().any():
        summary.append("Một số giao dịch thiếu ngày/thời gian. Vui lòng kiểm tra cột Date.")
    return "\n".join(summary)


def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Anomaly Report")
    return output.getvalue()


def get_download_link(file_bytes, filename, label):
    b64 = base64.b64encode(file_bytes).decode()
    return f"<a href='data:application/octet-stream;base64,{b64}' download='{filename}' class='download-button'>{label}</a>"


if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
    except Exception as exc:
        st.error(f"Không thể đọc dữ liệu: {exc}")
        st.stop()

    df = normalize_columns(df)
    df = clean_data(df)

    if "Amount" not in df.columns or df["Amount"].isna().all():
        st.error("File phải chứa cột `Amount` với dữ liệu số hợp lệ.")
        st.stop()

    if df.empty:
        st.error("File trống hoặc không có dòng dữ liệu.")
        st.stop()

    if len(df) > 5000 and api_key:
        st.warning(
            "Dữ liệu lớn đã được phát hiện. API key đã nhập sẽ giúp bạn ghi nhớ rằng nên kiểm tra kỹ khi dữ liệu quá nhiều."
        )

    st.markdown("### Dữ liệu đầu vào")
    st.dataframe(df.head(10))

    with st.spinner("Đang phân tích dữ liệu giao dịch..."):
        df = detect_anomalies(df, anomaly_threshold, extreme_amount)

    st.success("Hoàn tất phân tích dữ liệu.")

    total_trx = len(df)
    top_risk = df[df["RiskLevel"] == "Nguy cơ cao"].sort_values("AnomalyScore", ascending=False)
    summary_text = summarize_analysis(df, extreme_amount)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Tổng quan",
        "🚨 Giao dịch bất thường",
        "📈 Phân tích",
        "💾 Báo cáo",
    ])

    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Tổng quan nhanh")
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(
            f"<div class='metric'><h3>Tổng giao dịch</h3><h2>{total_trx}</h2></div>",
            unsafe_allow_html=True,
        )
        col2.markdown(
            f"<div class='metric'><h3>Giao dịch bất thường</h3><h2>{len(df[df['RiskLevel'] != 'Bình thường'])}</h2></div>",
            unsafe_allow_html=True,
        )
        col3.markdown(
            f"<div class='metric'><h3>Nguy cơ cao</h3><h2>{len(top_risk)}</h2></div>",
            unsafe_allow_html=True,
        )
        col4.markdown(
            f"<div class='metric'><h3>Trung bình / giao dịch</h3><h2>{df['Amount'].mean():,.0f}</h2></div>",
            unsafe_allow_html=True,
        )

        st.markdown("### Kết quả tóm tắt")
        st.info(summary_text)

        fig1 = px.histogram(
            df,
            x="AnomalyScore",
            nbins=30,
            title="Phân phối điểm bất thường",
            color="RiskLevel",
            color_discrete_map={
                "Bình thường": "#34d399",
                "Cảnh báo": "#facc15",
                "Nghi ngờ": "#fb7185",
                "Nguy cơ cao": "#ef4444",
                "Không xác định": "#94a3b8",
            },
        )
        fig1.update_layout(plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(255,255,255,0)")
        st.plotly_chart(fig1, use_container_width=True)

        account_scores = (
            df.groupby("Account")["AnomalyScore"].mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        fig2 = px.bar(
            account_scores,
            x="Account",
            y="AnomalyScore",
            title="Mức điểm bất thường trung bình theo tài khoản",
            color="AnomalyScore",
            color_continuous_scale="Turbo",
        )
        fig2.update_layout(xaxis_title="Tài khoản", yaxis_title="Điểm bất thường")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Top 20 giao dịch bất thường")
        st.dataframe(
            df.sort_values(["RiskLevel", "AnomalyScore"], ascending=[False, False]).head(20),
            use_container_width=True,
        )

        st.markdown("#### Những giao dịch có giá trị cao và rủi ro")
        risky = df[df["HighValue"]].sort_values("AbsAmount", ascending=False).head(15)
        if not risky.empty:
            st.table(
                risky[["Date", "Account", "Description", "Amount", "AnomalyScore", "RiskLevel"]]
            )
        else:
            st.info("Không có giao dịch giá trị lớn vượt ngưỡng.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Phân tích chi tiết")
        st.markdown("- Điểm `AnomalyScore` được tính từ z-score, giá trị lớn, giao dịch cuối tuần/đêm, số dư âm, và từ khóa cảnh báo.")
        st.markdown("- `Nguy cơ cao` yêu cầu đối chiếu chứng từ và kiểm tra ngay.")
        st.markdown("- `Nghi ngờ` là giao dịch cần rà soát thêm trước khi ghi sổ.")

        fig3 = px.line(
            df.groupby("Date")["AbsAmount"].sum().reset_index(),
            x="Date",
            y="AbsAmount",
            title="Tổng giá trị giao dịch theo ngày",
            markers=True,
        )
        fig3.update_layout(yaxis_title="Tổng giá trị (VNĐ)")
        st.plotly_chart(fig3, use_container_width=True)

        account_detail = (
            df.groupby(["Account", "RiskLevel"])["Amount"].count().reset_index()
        )
        fig4 = px.bar(
            account_detail,
            x="Account",
            y="Amount",
            color="RiskLevel",
            title="Số lượng giao dịch theo tài khoản và mức rủi ro",
        )
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Xuất báo cáo")
        st.markdown("Bạn có thể tải báo cáo chi tiết dưới dạng Excel hoặc CSV để lưu trữ hoặc chia sẻ.")

        excel_bytes = to_excel(df)
        st.markdown(
            get_download_link(
                excel_bytes,
                "bao_cao_giao_dich_bat_thuong.xlsx",
                "📥 Tải báo cáo Excel",
            ),
            unsafe_allow_html=True,
        )

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.markdown(
            get_download_link(csv_bytes, "bao_cao_giao_dich_bat_thuong.csv", "📥 Tải báo cáo CSV"),
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.warning("Vui lòng upload file sổ cái để bắt đầu phân tích giao dịch bất thường.")
    st.markdown("### Định dạng dữ liệu mẫu")
    st.markdown(
        "File nên có các cột: `Date`, `Account`, `Description`, `Amount`, `Balance` (nếu có)."
    )
    st.markdown(
        "Các giá trị `Amount` có thể dương hoặc âm. `Date` nên ở dạng ngày/thời gian."
    )
