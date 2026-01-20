# Z-Image Pro : Studio

fal.ai Z-Image Turbo를 테스트를 위한 이미지 생성 Streamlit 앱입니다.

## 주요 기능
- 프롬프트 기반 이미지 생성 (fal.ai Model API)
- 장수/시간 기준 배치 작업
- 생성 결과 요약 로그
- 이미지 다운로드 버튼 제공

## 설치
```bash
python -m venv fal-env
source fal-env/bin/activate
pip install -r requirements.txt
```

## 실행
```bash
export FAL_KEY="your-fal-key"
streamlit run z_image_web_pro.py
```

## 주의사항
- `FAL_KEY`는 환경변수로만 설정하세요.
- 결과 이미지와 로그 파일은 저장하지 않습니다.
