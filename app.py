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
from torchcam.methods import GradCAMpp
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
from rembg import remove

st.set_page_config(page_title="바질 질소결핍 진단기", page_icon="🌿", layout="centered")
st.title("🌿 바질 질소결핍 진단기")
st.markdown("이미지를 업로드하면 AI가 정상/질소결핍 여부를 판단합니다.")

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

eval_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def remove_bg(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    result = remove(buf.getvalue())
    img = Image.open(io.BytesIO(result)).convert("RGBA")
    bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
    bg.paste(img, mask=img.split()[3])
    return bg.convert("RGB")

def get_gradcam(nobg_img, tensor, pred_idx):
    cam_extractor = GradCAMpp(model, target_layer=model.conv_head)
    out = model(tensor)
    activation_map = cam_extractor(pred_idx, out)
    cam_extractor.remove_hooks()
    result = overlay_mask(
        to_pil_image(eval_transform(nobg_img)),
        to_pil_image(activation_map[0].squeeze(0), mode='F'),
        alpha=0.5
    )
    return np.array(result)

def predict(pil_img):
    nobg = remove_bg(pil_img)
    tensor = eval_transform(nobg).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]
    pred_idx = probs.argmax().item()
    conf = probs.max().item() * 100
    tensor_grad = eval_transform(nobg).unsqueeze(0).to(device)
    tensor_grad.requires_grad_(True)
    cam_overlay = get_gradcam(nobg, tensor_grad, pred_idx)
    return CLASSES[pred_idx], conf, probs, nobg, cam_overlay

uploaded = st.file_uploader("바질 이미지 업로드", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="업로드된 이미지", use_column_width=True)

    with st.spinner("AI 분석 중..."):
        pred, conf, probs, nobg, cam = predict(img)

    color = "green" if pred == "normal" else "red"
    label = "✅ 정상" if pred == "normal" else "⚠️ 질소결핍"
    st.markdown(f"## <span style='color:{color}'>{label} ({conf:.1f}%)</span>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image(nobg, caption="배경 제거 이미지")
    with col2:
        st.image(cam, caption="GradCAM++ (AI 주목 영역)")

    st.markdown("### 클래스별 확률")
    for i, cls in enumerate(CLASSES):
        label_kr = "정상" if cls == "normal" else "질소결핍"
        st.progress(int(probs[i].item() * 100), text=f"{label_kr}: {probs[i].item()*100:.1f}%")
