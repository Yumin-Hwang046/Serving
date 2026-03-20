#bash: streamlit run app.py, localhost:8501에서 실행

from pathlib import Path
import numpy as np
import streamlit as st
import onnxruntime as ort
from streamlit_drawable_canvas import st_canvas
from PIL import Image

MODEL_PATH = str(Path(__file__).resolve().parent / "mnist-12.onnx")

st.set_page_config(page_title="MNIST Drawer", layout="wide")

# 기능: 모델 관리
# MNIST ONNX 모델을 로드하고 세션 간 재사용을 위해 캐싱합니다.
@st.cache_resource
def load_model(path: str):
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])

# 기능: 이미지 전처리:
# Grayscale, 28*28 리사이즈, 정규화 [0,1], (배치, 채널, 높이, 너비) 형식
def preprocess(img_rgba: np.ndarray) -> np.ndarray:
    img = Image.fromarray(img_rgba).convert("L")
    img = img.resize((28, 28))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.reshape(1, 1, 28, 28)
    return arr

# 기능: 추론 수행
def infer(session, input_tensor: np.ndarray):
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_tensor})[0]
    probs = softmax(output[0])
    pred = int(np.argmax(probs))
    return pred, probs

def softmax(x):
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / np.sum(exp)

# UI: 이미지 저장소, 저장용 상태
if "gallery" not in st.session_state:
    st.session_state.gallery = []

# 기능: 캐시된 모델 로드
session = load_model(MODEL_PATH)

# ---- 레이아웃: 사이드바(컨트롤) ----
with st.sidebar:
    st.header("컨트롤")
    st.caption("모델 경로")
    st.code(MODEL_PATH, language="text")
    st.divider()
    st.subheader("저장")

# ---- 레이아웃: 메인 헤더 ----
st.title("MNIST 숫자 그리기")
st.caption("왼쪽에서 숫자를 그리고, 전처리 결과와 예측 확률을 확인하세요.")
st.divider()

# ---- 레이아웃: 메인 2컬럼 ----
main = st.container()
with main:
    left, right = st.columns([2, 1])

    with left:
        st.subheader("입력 캔버스")
        canvas = st_canvas(
            fill_color="black",
            stroke_width=16,
            stroke_color="white",
            background_color="black",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key="canvas",
        )

    with right:
        st.subheader("전처리 이미지")
        pre_img = None
        pre_tensor = None
        if canvas.image_data is not None:
            pre_tensor = preprocess(canvas.image_data.astype(np.uint8))
            pre_img = pre_tensor.reshape(28, 28)
            st.image(pre_img, clamp=True, width=140)
        else:
            st.caption("그린 이미지가 아직 없습니다.")

# ---- 레이아웃: 결과 ----
st.subheader("모델 추론 결과")
pred_label = None
pred_probs = None
if pre_tensor is not None:
    pred_label, pred_probs = infer(session, pre_tensor)
    st.bar_chart(pred_probs)
else:
    st.caption("결과는 전처리 이미지가 있을 때 표시됩니다.")

st.divider()

# ---- 사이드바에 저장 버튼/예측 정보 표시 ----
with st.sidebar:
    if st.button("이미지 저장"):
        if pred_label is not None:
            st.session_state.gallery.append(
                {
                    "img": pre_img.copy(),
                    "label": pred_label,
                    "prob": float(pred_probs[pred_label]),
                }
            )
    if pred_label is not None:
        st.write(f"예측 레이블: {pred_label}")
        st.write(f"확률: {pred_probs[pred_label]:.4f}")

# ---- 레이아웃: 이미지 저장소 ----
st.subheader("이미지 저장소")
if st.session_state.gallery:
    cols = st.columns(5)
    for i, item in enumerate(st.session_state.gallery):
        with cols[i % 5]:
            st.image(item["img"], clamp=True, width=80)
            st.caption(f"{item['label']} ({item['prob']:.2f})")
else:
    st.caption("저장된 이미지가 없습니다.")
