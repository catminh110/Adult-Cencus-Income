import streamlit as st
import pandas as pd
import joblib
import os

# 1. Tải mô hình đã lưu (Dùng đường dẫn tuyệt đối để chống lỗi FileNotFoundError)
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '..', 'src', 'models', 'income_classifier.pkl')
model = joblib.load(model_path)

st.title("Dự đoán Thu nhập Adult Census 💵")
st.markdown("App đáp ứng yêu cầu: Nhập thông tin → Dự đoán + Giải thích (Explain) + Cảnh báo (Ethics/Limitations)")

# --- TỪ ĐIỂN MAP TIẾNG VIỆT -> TIẾNG ANH ---
workclass_map = {'Tư nhân': 'Private', 'Công chức địa phương': 'Local-gov', 'Tự kinh doanh (cá nhân)': 'Self-emp-not-inc', 'Chính phủ liên bang': 'Federal-gov', 'Chính quyền bang': 'State-gov', 'Tự kinh doanh (doanh nghiệp)': 'Self-emp-inc', 'Làm không lương': 'Without-pay'}
edu_map = {'Cử nhân Đại học': 'Bachelors', 'Học một số môn Đại học': 'Some-college', 'Lớp 11': '11th', 'Tốt nghiệp Cấp 3': 'HS-grad', 'Trường Cao đẳng/Nghề': 'Prof-school', 'Cao đẳng học thuật': 'Assoc-acdm', 'Cao đẳng nghề': 'Assoc-voc', 'Lớp 9': '9th', 'Lớp 7-8': '7th-8th', 'Lớp 12': '12th', 'Thạc sĩ': 'Masters', 'Lớp 1-4': '1st-4th', 'Lớp 10': '10th', 'Tiến sĩ': 'Doctorate', 'Lớp 5-6': '5th-6th', 'Mẫu giáo': 'Preschool'}
marital_map = {'Đã kết hôn': 'Married-civ-spouse', 'Đã ly hôn': 'Divorced', 'Độc thân (chưa từng kết hôn)': 'Never-married', 'Ly thân': 'Separated', 'Góa': 'Widowed', 'Kết hôn nhưng vợ/chồng vắng mặt': 'Married-spouse-absent', 'Kết hôn (vợ/chồng trong quân đội)': 'Married-AF-spouse'}
occ_map = {'Hỗ trợ kỹ thuật': 'Tech-support', 'Thợ thủ công/Sửa chữa': 'Craft-repair', 'Dịch vụ khác': 'Other-service', 'Nhân viên Bán hàng/Chứng khoán': 'Sales', 'Quản lý/Giám đốc': 'Exec-managerial', 'Chuyên gia/Bác sĩ/Kỹ sư': 'Prof-specialty', 'Lao động tay chân/Dọn dẹp': 'Handlers-cleaners', 'Vận hành máy móc': 'Machine-op-inspct', 'Hành chính/Văn thư': 'Adm-clerical', 'Nông nghiệp/Đánh bắt': 'Farming-fishing', 'Vận tải/Tài xế': 'Transport-moving', 'Giúp việc nhà': 'Priv-house-serv', 'Bảo vệ/An ninh': 'Protective-serv', 'Lực lượng vũ trang': 'Armed-Forces'}
rel_map = {'Vợ': 'Wife', 'Con cái': 'Own-child', 'Chồng': 'Husband', 'Không ở cùng gia đình': 'Not-in-family', 'Họ hàng khác': 'Other-relative', 'Không kết hôn': 'Unmarried'}
race_map = {'Da trắng': 'White', 'Châu Á - Thái Bình Dương': 'Asian-Pac-Islander', 'Châu Mỹ bản địa': 'Amer-Indian-Eskimo', 'Khác': 'Other', 'Da đen': 'Black'}
sex_map = {'Nữ': 'Female', 'Nam': 'Male'}
country_map = {'Mỹ': 'United-States', 'Campuchia': 'Cambodia', 'Anh': 'England', 'Canada': 'Canada', 'Đức': 'Germany', 'Ấn Độ': 'India', 'Nhật Bản': 'Japan', 'Hàn Quốc': 'South', 'Trung Quốc': 'China', 'Cuba': 'Cuba', 'Việt Nam': 'Vietnam', 'Mexico': 'Mexico', 'Pháp': 'France'}

# 2. Form nhập liệu (Input)
st.header("1. Nhập thông tin (Input)")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Tuổi", 17, 90, 30)
    sel_workclass = st.selectbox("Nơi làm việc", list(workclass_map.keys()))
    sel_edu = st.selectbox("Trình độ Học vấn", list(edu_map.keys()))
    education_num = st.number_input("Số năm đi học (tương đối)", 1, 16, 13)
    sel_marital = st.selectbox("Tình trạng Hôn nhân", list(marital_map.keys()))
    sel_occ = st.selectbox("Nghề nghiệp", list(occ_map.keys()))
    sel_rel = st.selectbox("Mối quan hệ trong nhà", list(rel_map.keys()))

with col2:
    sel_race = st.selectbox("Chủng tộc", list(race_map.keys()))
    sel_sex = st.selectbox("Giới tính", list(sex_map.keys()))
    capital_gain = st.number_input("Thu nhập đầu tư/Chứng khoán ($)", 0, 100000, 0)
    capital_loss = st.number_input("Lỗ đầu tư ($)", 0, 5000, 0)
    hours_per_week = st.number_input("Số giờ làm việc/tuần", 1, 99, 40)
    sel_country = st.selectbox("Quốc gia gốc", list(country_map.keys()))
    fnlwgt = st.number_input("Trọng số dân số (Fnlwgt)", 10000, 1500000, 200000)

# 3. Dự đoán và Giải thích
if st.button("Dự đoán Thu Nhập", type="primary"):
    # Lấy giá trị Tiếng Anh gốc để đưa vào mô hình [cite: 17]
    input_data = pd.DataFrame([[
        age, workclass_map[sel_workclass], fnlwgt, edu_map[sel_edu], education_num, 
        marital_map[sel_marital], occ_map[sel_occ], rel_map[sel_rel], race_map[sel_race], 
        sex_map[sel_sex], capital_gain, capital_loss, hours_per_week, country_map[sel_country]
    ]], columns=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    st.header("2. Kết quả Dự đoán")
    if prediction == 1:
        st.success(f"🎉 Dự đoán: Thu nhập **LỚN HƠN 50,000 USD/năm** (Xác suất: {probability:.2%})")
    else:
        st.error(f"📉 Dự đoán: Thu nhập **NHỎ HƠN HOẶC BẰNG 50,000 USD/năm** (Xác suất: {1 - probability:.2%})")
        
    st.header("3. Giải thích Mô hình (Explainability)")
    st.info("💡 **Giải thích:** Mô hình Logistic Regression này đánh giá rất cao các yếu tố như **Thu nhập đầu tư** và việc **Đã kết hôn** trong việc làm tăng xác suất thu nhập trên 50K. Ngược lại, những nghề như **Giúp việc nhà** sẽ làm giảm mạnh xác suất này.")

# 4. Ethics / Limitations
st.header("4. Cảnh báo Đạo đức & Hạn chế (Ethics & Limitations)")
st.warning("⚠️ **Cảnh báo:** Mô hình này được huấn luyện trên dữ liệu cũ. Phân tích Audit cho thấy mô hình có **thiên kiến (bias)**, dự đoán kém chính xác hơn đối với Nữ giới. **Tuyệt đối không sử dụng** mô hình này để ra quyết định tuyển dụng hay trả lương thực tế[cite: 17, 32].")

# 5. Tích hợp Chatbot vào thanh Sidebar
with st.sidebar:
    st.header("Trợ lý Dự án 🤖")
    st.markdown("Hỏi tôi về: 'mô hình', 'hạn chế', hoặc 'dữ liệu'")

    # Khởi tạo lịch sử chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Tạo một container riêng cho tin nhắn để có thể cuộn (scroll)
    chat_container = st.container(height=400)

    # Hiển thị lịch sử trong container
    for message in st.session_state.messages:
        with chat_container.chat_message(message["role"]):
            st.markdown(message["content"])

    # Ô nhập liệu chat nằm ở dưới cùng của sidebar
    if prompt := st.chat_input("Nhập câu hỏi tại đây..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container.chat_message("user"):
            st.markdown(prompt)

        # Logic trả lời
        prompt_lower = prompt.lower()
        if "giải thích" in prompt_lower or "mô hình" in prompt_lower:
            response = "Mô hình dùng Logistic Regression. Yếu tố tăng thu nhập mạnh nhất là 'Capital Gain' (Đầu tư) và 'Married-civ-spouse' (Kết hôn)."
        elif "hạn chế" in prompt_lower or "thiên kiến" in prompt_lower:
            response = "Mô hình có thiên kiến giới tính (bias), dự đoán Nữ giới kém chính xác hơn Nam giới. Không dùng để quyết định thực tế."
        elif "eda" in prompt_lower or "dữ liệu" in prompt_lower:
            response = "Phân tích cho thấy học vấn cao và làm nhiều giờ/tuần có tỉ lệ đạt thu nhập >50K cao nhất."
        else:
            response = "Tôi chỉ giải đáp về: 'mô hình', 'hạn chế', và 'dữ liệu'."

        with chat_container.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})