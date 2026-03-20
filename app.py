import streamlit as st

from camera import run_data_collection, run_live_recognition
from ui import init_state, render_idle_camera, render_layout, setup_page


def main():
    setup_page()
    init_state()

    ui_data = render_layout()
    placeholder = ui_data["placeholder"]

    if not st.session_state.run_cam:
        render_idle_camera(placeholder)
        return

    if ui_data["mode"] == "Lấy hình":
        run_data_collection(placeholder, ui_data["person_name"], ui_data["max_img"])
        return

    if ui_data["mode"] == "Nhận diện":
        run_live_recognition(placeholder)


if __name__ == "__main__":
    main()