import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_and_clean_data(file_path):
    # 1. Đọc dữ liệu từ file CSV
    df = pd.read_csv(file_path)

    # 2. Chuẩn hóa dữ liệu tên cột và định dạng ngày tháng
    df.columns = df.columns.str.strip()
    df["Ngày"] = pd.to_datetime(df["Ngày"], format="%d/%m/%Y")

    # Sắp xếp lại dữ liệu theo thứ tự thời gian tăng dần
    df = df.sort_values(by="Ngày").reset_index(drop=True)

    # 3. Chuyển đổi các cột số từ chuỗi (định dạng VN dùng dấu phẩy) sang float
    cols_to_convert = ["Giá đóng cửa", "Giá mở cửa", "Giá cao nhất", "Giá thấp nhất"]
    for col in cols_to_convert:
        if df[col].dtype == "object":
            df[col] = (
                df[col].astype(str).str.replace(".", "").str.replace(",", ".")
            )
        df[col] = pd.to_numeric(df[col])

    return df


def calculate_indicators(df):
    # Tính đường trung bình động đơn giản (SMA) 10 ngày và 20 ngày để xác định xu hướng
    df["SMA10"] = df["Giá đóng cửa"].rolling(window=10).mean()
    df["SMA20"] = df["Giá đóng cửa"].rolling(window=20).mean()

    # Tính chỉ số sức mạnh tương đối (RSI 14) để biết thị trường có bị quá mua/quá bán không
    delta = df["Giá đóng cửa"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    ma_gain = gain.rolling(window=14).mean()
    ma_loss = loss.rolling(window=14).mean()

    # Tránh chia cho 0
    rs = ma_gain / ma_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)  # Điền giá trị trung hòa nếu không có dữ liệu

    return df


def generate_signals(df):
    """
    Hệ thống tạo tín hiệu giao dịch dựa trên SMA và RSI
    """
    df["Tín_Hiệu"] = "Theo dõi"
    df["Chi_Tiết_Lý_Do"] = ""

    for i in range(1, len(df)):
        close = df.loc[i, "Giá đóng cửa"]
        sma10 = df.loc[i, "SMA10"]
        sma20 = df.loc[i, "SMA20"]
        rsi = df.loc[i, "RSI"]

        # Bỏ qua các hàng đầu tiên chưa đủ dữ liệu tính toán SMA
        if pd.isna(sma20):
            continue

        # KỊCH BẢN MUA (BUY)
        # Giá cắt lên trên các đường trung bình hoặc RSI nằm ở vùng quá bán và bắt đầu hồi phục
        if (close > sma10 and df.loc[i - 1, "Giá đóng cửa"] <= df.loc[i - 1, "SMA10"]) or (rsi < 35):
            df.loc[i, "Tín_Hiệu"] = "MUA (BUY)"
            df.loc[i, "Chi_Tiết_Lý_Do"] = (
                f"Giá ({close:,.2f}) bắt đầu vượt đường xu hướng ngắn hạn hoặc RSI ({rsi:.1f}) vùng quá bán thấp."
            )

        # KỊCH BẢN BÁN (SELL)
        # Giá thủng đường trung bình hoặc RSI nằm ở vùng quá mua (Rủi ro đảo chiều giảm)
        elif (close < sma10 and df.loc[i - 1, "Giá đóng cửa"] >= df.loc[i - 1, "SMA10"]) or (rsi > 75):
            df.loc[i, "Tín_Hiệu"] = "BÁN (SELL)"
            df.loc[i, "Chi_Tiết_Lý_Do"] = (
                f"Giá rơi xuống dưới xu hướng ngắn hạn hoặc RSI ({rsi:.1f}) đi vào vùng quá mua."
            )

        else:
            df.loc[i, "Tín_Hiệu"] = "NẮM GIỮ / THEO DÕI"
            df.loc[i, "Chi_Tiết_Lý_Do"] = "Xu hướng hiện tại chưa có sự đột biến rõ rệt."

    return df


def market_advisor(df):
    """
    Đưa ra nhận định tổng quan tại phiên gần đây nhất
    """
    latest_row = df.iloc[-1]
    prev_row = df.iloc[-2]

    print("==================================================================")
    print(
        f"      BÁO CÁO PHÂN TÍCH THỊ TRƯỜNG VN30 - NGÀY {latest_row['Ngày'].strftime('%d/%m/%Y')}"
    )
    print("==================================================================")
    print(f"-> Giá đóng cửa hiện tại: {latest_row['Giá đóng cửa']:,.2f}")
    print(
        f"-> Biến động so với phiên trước: {latest_row['Giá đóng cửa'] - prev_row['Giá đóng cửa']:+,.2f}"
    )
    print(f"-> Chỉ số RSI (14 phiên): {latest_row['RSI']:.2f}")
    print(f"-> Xu hướng SMA10: {latest_row['SMA10']:,.2f} | SMA20: {latest_row['SMA20']:,.2f}")
    print("------------------------------------------------------------------")
    print(f" KHUYẾN NGHỊ HÀNH ĐỘNG: {latest_row['Tín_Hiệu']}")
    print(f" Lý do kỹ thuật: {latest_row['Chi_Tiết_Lý_Do']}")
    print("------------------------------------------------------------------")

    # GỢI Ý CHIẾN LƯỢC MUA VÀ QUẢN TRỊ RỦI RO
    print("\n💡 GỢI Ý CHIẾN LƯỢC QUYẾT ĐỊNH CHO NGƯỜI MUA:")
    if latest_row["Tín_Hiệu"] == "MUA (BUY)":
        print(
            "   1. Phương thức giải ngân: Chia tiền làm 2-3 phần (ví dụ: 30% - 40% - 30%)."
        )
        print(
            "      Giải ngân phần đầu tiên ngay tại vùng giá này để lấy vị thế."
        )
        print(
            "   2. Điểm gia tăng: Mua thêm khi thị trường có nhịp kiểm chứng (retest) lại hỗ trợ thành công."
        )
        print(
            f"   3. Điểm cắt lỗ (Stop-loss): Tuyệt đối quản trị rủi ro nếu giá thủng vùng ranh giới SMA20 (~{latest_row['SMA20']:,.2f})."
        )
    elif latest_row["Tín_Hiệu"] == "BÁN (SELL)":
        print(
            "   1. Đối với người đang giữ vị thế: Nên chốt lời từng phần để bảo toàn lợi nhuận thu được."
        )
        print(
            "   2. Đối với người muốn mua mới: KIÊN NHẪN ĐỢI. Không mua đuổi (FOMO) khi RSI đang neo cao."
        )
        print(
            "      Hãy đợi thị trường điều chỉnh về các vùng cân bằng thấp hơn trước khi tham gia."
        )
    else:
        print(
            "   1. Thị trường đang đi vào vùng tích lũy hoặc chưa rõ xu hướng bứt phá."
        )
        print(
            "   2. Ưu tiên giữ tiền mặt tỷ lệ an toàn (50/50), hạn chế sử dụng đòn bẩy tài chính (Margin) lúc này."
        )
        print(
            "   3. Tập trung quan sát các cổ phiếu riêng lẻ thuộc rổ VN30 có nền tảng tốt và dòng tiền vào riêng biệt."
        )


def plot_market_chart(df):
    """
    Vẽ đồ thị trực quan giá và các chỉ báo
    """
    plt.figure(figsize=(14, 8))

    # Đồ thị giá và SMA
    plt.subplot(2, 1, 1)
    plt.plot(df["Ngày"], df["Giá đóng cửa"], label="Giá VN30", color="blue", linewidth=2)
    plt.plot(df["Ngày"], df["SMA10"], label="SMA 10 (Ngắn hạn)", color="orange", linestyle="--")
    plt.plot(df["Ngày"], df["SMA20"], label="SMA 20 (Trung hạn)", color="red", linestyle="--")
    plt.title("Phân Tích Xu Hướng Giá VN30 Index")
    plt.ylabel("Điểm chỉ số")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Đồ thị RSI
    plt.subplot(2, 1, 2)
    plt.plot(df["Ngày"], df["RSI"], label="RSI (14)", color="purple")
    plt.axhline(70, color="red", linestyle=":", label="Quá mua (Overbought > 70)")
    plt.axhline(30, color="green", linestyle=":", label="Quá bán (Oversold < 30)")
    plt.ylabel("Giá trị RSI")
    plt.xlabel("Thời gian")
    plt.ylim(10, 90)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ==========================================
# THỰC THI CHƯƠNG TRÌNH
# ==========================================
if __name__ == "__main__":
    # Thay tên file chính xác của bạn vào đây
    file_name = "vn30index.xlsx - LichSuGia.csv"

    # Chạy quy trình phân tích
    df_market = load_and_clean_data(file_name)
    df_market = calculate_indicators(df_market)
    df_market = generate_signals(df_market)

    # Xuất báo cáo khuyến nghị ra màn hình terminal
    market_advisor(df_market)

    # Hiển thị đồ thị trực quan
    plot_market_chart(df_market)