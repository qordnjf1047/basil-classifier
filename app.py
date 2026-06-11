import streamlit as st
import torch
import torch.nn.functional as F
import timm
import numpy as np
from PIL import Image
import io
import os
import gdown
from torchvision import transforms
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from rembg import remove

# ── 페이지 설정 ──────────────────────────────────────────
st.set_page_config(
    page_title="바질 질소결핍 진단기",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 바질 질소결핍 진단기")
st.markdown("이미지를 업로드하면 AI가 정상/질소결핍 여부를 판단합니다.")

# ── 모델 로드 ─────────────────────────────────────────────
MODEL_PATH = "best_model_nobg.pth"
GDRIVE_ID  = "1vivDfBxh-kV7G_GIGqfi6l647WDDWlEx"
CLASSES    = ['nitrogen_deficient', 'normal']

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("모델 다운로드 중..."):
            gdown.download(f"https://drive.google.com/uc?id={GDRIVE_ID}", MODEL_PATH, quiet=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# ── 전처리 ───────────────────────────────────────────────
eval_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── 함수 ─────────────────────────────────────────────────
def remove_bg(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    result = remove(buf.getvalue())
    img = Image.open(io.BytesIO(result)).convert("RGBA")
    bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
    bg.paste(img, mask=img.split()[3])
    return bg.convert("RGB")

def get_gradcam(tensor, pred_idx):
    target_layers = [model.conv_head]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    grayscale = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(pred_idx)])
    return grayscale[0]

def predict(pil_img):
    nobg = remove_bg(pil_img)
    tensor = eval_transform(nobg).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]
    pred_idx = probs.argmax().item()
    conf = probs.max().item() * 100

    # GradCAM
    img_np = np.array(nobg.resize((300, 300))) / 255.0
    cam_map = get_gradcam(tensor, pred_idx)
    cam_overlay = show_cam_on_image(img_np.astype(np.float32), cam_map, use_rgb=True)

    return CLASSES[pred_idx], conf, probs, nobg, cam_overlay

# ── UI ───────────────────────────────────────────────────
uploaded = st.file_uploader("바질 이미지 업로드", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="업로드된 이미지", use_column_width=True)

    with st.spinner("AI 분석 중..."):
        pred, conf, probs, nobg, cam = predict(img)

    # 결과 표시
    color = "green" if pred == "normal" else "red"
    label = "✅ 정상" if pred == "normal" else "⚠️ 질소결핍"
    st.markdown(f"## <span style='color:{color}'>{label} ({conf:.1f}%)</span>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image(nobg, caption="배경 제거 이미지")
    with col2:
        st.image(cam, caption="GradCAM (AI 주목 영역)")

    st.markdown("### 클래스별 확률")
    for i, cls in enumerate(CLASSES):
        label_kr = "정상" if cls == "normal" else "질소결핍"
        st.progress(int(probs[i].item() * 100), text=f"{label_kr}: {probs[i].item()*100:.1f}%")
