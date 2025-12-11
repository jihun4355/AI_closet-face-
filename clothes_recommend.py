import pandas as pd
import os
import cv2
from PIL import Image
import numpy as np

def recommend_clothes(emotion_input:str):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'clothes_data.csv')
    img_dir = os.path.join(script_dir, 'clothes_style')

    # === CSV 불러오기 ===
    df = pd.read_csv(csv_path)

    # === 감정 입력 받기 ===
    # emotion_input = input("감정을 입력하세요 (happy, sad, angry, default 중 하나): ").strip().lower()
    # valid_emotions = ["happy", "sad", "angry", "default"]
    # if emotion_input not in valid_emotions:
    #     print("잘못된 감정 입력입니다.")
    #     exit()

    # === 상의/하의 스타일 정의 ===
    top_styles = ['top', 'shirt']
    bottom_styles = ['shorts', 'pants']

    # === 조건에 맞는 옷 필터링 ===
    available = df[
        (df['emotion'].str.lower() == emotion_input) &
        (df['in_stock'].astype(str).str.upper() == "TRUE")
    ]

    # === 한글 경로 이미지 열기 함수 ===
    def load_image_unicode(path):
        try:
            with Image.open(path) as pil_img:
                return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"이미지 열기 실패: {e}")
            return None

    # === 상의 추천 ===
    top_candidates = available[available['style'].str.lower().isin(top_styles)]
    if not top_candidates.empty:
        top = top_candidates.sample(1).iloc[0]
        top_img_path = os.path.join(img_dir, top['filename'])

        print("\n상의 추천:")
        print(f"- 파일명: {top['filename']}")
        print(f"- 설명: {top['description']}")
        print(f"- 경로: {top_img_path}")

        img = load_image_unicode(top_img_path)
        if img is None:
            print("상의 이미지 파일을 열 수 없습니다.")
        else:
            img = cv2.resize(img, (300, 300))
            cv2.imshow("상의 추천", img)
    else:
        print("추천 가능한 상의가 없습니다.")

    # === 하의 추천 ===
    bottom_candidates = available[available['style'].str.lower().isin(bottom_styles)]
    if not bottom_candidates.empty:
        bottom = bottom_candidates.sample(1).iloc[0]
        bottom_img_path = os.path.join(img_dir, bottom['filename'])

        print("\n하의 추천:")
        print(f"- 파일명: {bottom['filename']}")
        print(f"- 설명: {bottom['description']}")
        print(f"- 경로: {bottom_img_path}")

        img = load_image_unicode(bottom_img_path)
        if img is None:
            print("❌ 하의 이미지 파일을 열 수 없습니다.")
        else:
            img = cv2.resize(img, (300, 300))
            cv2.imshow("하의 추천", img)
    else:
        print("추천 가능한 하의가 없습니다.")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
