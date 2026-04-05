import streamlit as st
import os
import shutil

def setup_page():
    st.set_page_config(page_title="Face Recognition System", layout="wide")
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; padding-bottom: 0rem; }
        .camera-box {
            border: 2px dashed #4b4b4b; border-radius: 10px;
            height: 500px; width: 100%; max-width: 640px;
            display: flex; align-items: center; justify-content: center;
            font-family: sans-serif; color: #888;
        }
        div.stButton > button { width: 100%; border-radius: 5px; height: 3em; }
        
        .streamlit-expanderHeader {
            background-color: #1e1e1e;
            border-radius: 5px;
            border: 1px solid #4b4b4b;
            margin-bottom: 5px;
        }
        .class-list-bottom-pad { height: 14px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def init_state():
    if "mode" not in st.session_state: st.session_state.mode = "Lấy hình"
    if "run_cam" not in st.session_state: st.session_state.run_cam = False
    if "active_cls" not in st.session_state: st.session_state.active_cls = None

def render_layout():
    col_left, _, col_right = st.columns([0.35, 0.05, 0.6])
    dataset_dir = "dataset"
    if not os.path.exists(dataset_dir): os.makedirs(dataset_dir)

    with col_left:
        with st.container(height=640, border=False):
            st.subheader("Điều Khiển")
            btn_col1, btn_col2 = st.columns(2)
            
            if btn_col1.button("Thu Thập Dữ Liệu", type="primary" if st.session_state.mode == "Lấy hình" else "secondary"):
                st.session_state.mode = "Lấy hình"; st.session_state.run_cam = False; st.rerun()

            if btn_col2.button("Nhận Diện Khuôn Mặt", type="primary" if st.session_state.mode == "Nhận diện" else "secondary"):
                st.session_state.mode = "Nhận diện"; st.session_state.run_cam = False; st.rerun()

            st.divider()

            model_type = "KNN" # Khởi tạo mặc định
            
            if st.session_state.mode == "Lấy hình":
                st.write("**Cấu hình lấy dữ liệu**")
                person_name = st.text_input("Tên định danh (Class):", "User_A")
                max_img = st.slider("Số lượng ảnh cần chụp:", 50, 500, 100)
                if not st.session_state.run_cam:
                    if st.button("MỞ CAMERA CHỤP", use_container_width=True): 
                        st.session_state.run_cam = True
                        st.session_state.active_cls = None
                        st.rerun()
                else:
                    if st.button("DỪNG VÀ LƯU DỮ LIỆU", use_container_width=True):
                        st.session_state.run_cam = False
                        st.rerun()

                st.divider()
                st.write("**Danh sách Class đã có:**")
                classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
                
                if not classes:
                    st.caption("Chưa có dữ liệu.")
                
                for cls in classes:
                    cls_path = os.path.join(dataset_dir, cls)
                    is_active = st.session_state.active_cls == cls
                    toggle_label = f"▼ {cls}" if is_active else f"▶ {cls}"
                    if st.button(toggle_label, key=f"toggle_{cls}", use_container_width=True):
                        if is_active:
                            st.session_state.active_cls = None
                        else:
                            st.session_state.active_cls = cls
                        st.session_state.run_cam = False
                        st.rerun()

                    if is_active:
                        with st.container(border=True):
                            new_name = st.text_input("Đổi tên thành:", cls, key=f"edit_{cls}")
                            if new_name != cls and new_name.strip() != "":
                                os.rename(cls_path, os.path.join(dataset_dir, new_name.strip()))
                                if st.session_state.active_cls == cls:
                                    st.session_state.active_cls = new_name.strip()
                                st.rerun()
                            
                            if st.button(f"Xóa vĩnh viễn", key=f"del_{cls}"):
                                shutil.rmtree(cls_path)
                                if st.session_state.active_cls == cls:
                                    st.session_state.active_cls = None
                                st.rerun()

                st.markdown('<div class="class-list-bottom-pad"></div>', unsafe_allow_html=True)
            else:
                st.session_state.active_cls = None
                st.write("**Cấu hình nhận diện**")
                
                # --- CHỌN MODEL Ở ĐÂY ---
                model_type = st.radio("Chọn Thuật toán Nhận diện:", ["KNN", "SVM"], horizontal=True)
                
                st.caption("Model sẽ chỉ Train lại khi dữ liệu có sự thay đổi.")
                if not st.session_state.run_cam:
                    if st.button("BẮT ĐẦU NHẬN DIỆN", use_container_width=True):
                        st.session_state.run_cam = True
                        st.rerun()
                else:
                    if st.button("DỪNG HỆ THỐNG", use_container_width=True):
                        st.session_state.run_cam = False
                        st.rerun()
        
    with col_right:
        st.subheader("Màn Hình")
        placeholder = st.empty()

    return {
        "placeholder": placeholder,
        "mode": st.session_state.mode,
        "person_name": person_name if st.session_state.mode == "Lấy hình" else "User_A",
        "max_img": max_img if st.session_state.mode == "Lấy hình" else 200,
        "model_type": model_type # Trả về loại model
    }

def render_viewer(placeholder):
    dataset_dir = "dataset"
    view_cls = st.session_state.active_cls
    
    if view_cls and not st.session_state.run_cam:
        cls_path = os.path.join(dataset_dir, view_cls)
        if os.path.exists(cls_path):
            all_imgs = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.png'))]
            with placeholder.container():
                with st.container(height=480):
                    if not all_imgs: st.info("Thư mục trống.")
                    else:
                        cols = st.columns(4)
                        for idx, img in enumerate(all_imgs):
                            with cols[idx % 4]:
                                st.image(os.path.join(cls_path, img), use_container_width=True)
    else:
        placeholder.markdown('<div class="camera-box">Đang đợi tín hiệu Camera hoặc chọn Class để xem...</div>', unsafe_allow_html=True)