import streamlit as st


def setup_page():
    st.set_page_config(page_title="Face Recognition System", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 0rem;
        }

        .camera-box {
            border: 2px dashed #4b4b4b;
            border-radius: 10px;
            height: 500px;
            width: 100%;
            max-width: 640px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: sans-serif;
            color: #888;
        }

        div.stButton > button {
            width: 100%;
            border-radius: 5px;
            height: 3.5em;
        }

        hr {
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state():
    if "mode" not in st.session_state:
        st.session_state.mode = "Lấy hình"
    if "run_cam" not in st.session_state:
        st.session_state.run_cam = False


def render_layout():
    col_left, _, col_right = st.columns([0.3, 0.07, 0.63])

    with col_left:
        st.subheader("Điều Khiển")

        btn_col1, btn_col2 = st.columns(2)
        if btn_col1.button(
            "Thu Thập Dữ Liệu",
            type="primary" if st.session_state.mode == "Lấy hình" else "secondary",
        ):
            st.session_state.mode = "Lấy hình"
            st.session_state.run_cam = False
            st.rerun()

        if btn_col2.button(
            "Nhận Diện Khuôn Mặt",
            type="primary" if st.session_state.mode == "Nhận diện" else "secondary",
        ):
            st.session_state.mode = "Nhận diện"
            st.session_state.run_cam = False
            st.rerun()

        st.divider()

        person_name = "User_A"
        max_img = 200

        if st.session_state.mode == "Lấy hình":
            st.write("**Cấu hình lấy dữ liệu**")
            person_name = st.text_input("Tên định danh (Class):", "User_A")
            max_img = st.slider("Số lượng ảnh cần chụp:", 50, 1000, 200)

            if not st.session_state.run_cam:
                if st.button("MỞ CAMERA CHỤP", use_container_width=True):
                    st.session_state.run_cam = True
                    st.rerun()
            else:
                if st.button("DỪNG VÀ LƯU DỮ LIỆU", use_container_width=True):
                    st.session_state.run_cam = False
                    st.rerun()
        else:
            st.write("**Cấu hình nhận diện**")
            st.caption("Hệ thống tự động huấn luyện lại khi khởi động camera.")
            if not st.session_state.run_cam:
                if st.button("BẮT ĐẦU NHẬN DIỆN", use_container_width=True):
                    st.session_state.run_cam = True
                    st.rerun()
            else:
                if st.button("DỪNG HỆ THỐNG", use_container_width=True):
                    st.session_state.run_cam = False
                    st.rerun()

    with col_right:
        st.subheader("Màn Hình Camera")
        placeholder = st.empty()

    return {
        "placeholder": placeholder,
        "mode": st.session_state.mode,
        "person_name": person_name,
        "max_img": max_img,
    }


def render_idle_camera(placeholder):
    placeholder.markdown(
        '<div class="camera-box">Đang đợi tín hiệu Camera...</div>',
        unsafe_allow_html=True,
    )
