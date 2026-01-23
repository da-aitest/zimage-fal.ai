import json
import os
import time
from datetime import datetime

import fal_client
import requests
import streamlit as st

PRICE_PER_MP = 0.005
LOG_DIR = "logs"

ASPECT_RATIOS = {
    "Default": (1024, 1024),
    "Custom": None,
    "Square HD": (1024, 1024),
    "Square": (512, 512),
    "Portrait 3:4": (768, 1024),
    "Portrait 9:16": (576, 1024),
    "Landscape 4:3": (1024, 768),
    "Landscape 16:9": (1024, 576),
}


def calculate_cost(width: int, height: int) -> tuple[float, float]:
    """해상도에 따른 실시간 비용 계산"""
    mp = (width * height) / 1_000_000
    cost = mp * PRICE_PER_MP
    return cost, mp


def fetch_image_bytes(image_url: str) -> bytes | None:
    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception:
        return None


def parse_index_filter(text: str) -> set[int]:
    """예: '1,3,5-7' -> {1,3,5,6,7}"""
    if not text:
        return set()
    indices: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            if start_str.strip().isdigit() and end_str.strip().isdigit():
                start = int(start_str)
                end = int(end_str)
                if start > end:
                    start, end = end, start
                indices.update(range(start, end + 1))
        elif part.isdigit():
            indices.add(int(part))
    return indices


def generate_image_fal(
    prompt: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
) -> tuple[str, float]:
    """fal.ai API 호출"""
    arguments = {
        "prompt": prompt,
        "seed": seed,
        "num_inference_steps": steps,
        "guidance_scale": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "image_size": {"width": width, "height": height},
        "num_images": 1,
        "enable_safety_checker": False,
    }

    endpoint = "fal-ai/z-image/turbo"

    req_start = time.time()
    handler = fal_client.submit(endpoint, arguments=arguments)
    result = handler.get()
    req_end = time.time()

    inference_time = req_end - req_start
    image_url = result["images"][0]["url"]

    return image_url, inference_time


def ensure_log_dir() -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    return LOG_DIR


def write_log_file(
    log_lines: list[str],
    settings_payload: dict,
    results_payload: dict,
) -> str:
    ensure_log_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(LOG_DIR, f"zimage_log_{timestamp}.json")
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "timestamp": timestamp,
                "settings": settings_payload,
                "results": results_payload,
                "log_lines": log_lines,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    return file_path


def main() -> None:
    st.set_page_config(page_title="Z-Image Pro", page_icon="🚀", layout="wide")

    if "running" not in st.session_state:
        st.session_state.running = False
    if "log_lines" not in st.session_state:
        st.session_state.log_lines = []
    if "images_history" not in st.session_state:
        st.session_state.images_history = []
    if "generated_count" not in st.session_state:
        st.session_state.generated_count = 0
    if "total_cost" not in st.session_state:
        st.session_state.total_cost = 0.0
    if "speed_label" not in st.session_state:
        st.session_state.speed_label = "평균 속도"
    if "speed_value" not in st.session_state:
        st.session_state.speed_value = "0.00s/장"
    if "render_seq" not in st.session_state:
        st.session_state.render_seq = 0
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "current_task" not in st.session_state:
        st.session_state.current_task = 0
    if "total_tasks" not in st.session_state:
        st.session_state.total_tasks = 0
    if "mode_label" not in st.session_state:
        st.session_state.mode_label = ""
    if "elapsed_seconds" not in st.session_state:
        st.session_state.elapsed_seconds = 0.0
    if "total_seconds" not in st.session_state:
        st.session_state.total_seconds = 0.0

    if "prompt" not in st.session_state:
        st.session_state.prompt = (
            "Breathtaking Aurora Borealis (Northern Lights) swirling over a frozen arctic "
            "landscape, starry night sky, reflection on ice, snow-covered trees in "
            "silhouette, deep dark night atmosphere, no artificial lights, lit only by the "
            "vivid green and purple glow of the aurora, cinematic composition, highly "
            "saturated colors, long exposure photography style, high contrast, 8k, "
            "extremely detailed, photorealistic, masterpiece, wide angle shot."
        )
    if "ratio_name" not in st.session_state:
        st.session_state.ratio_name = "Default"
    if "custom_width" not in st.session_state:
        st.session_state.custom_width = 1024
    if "custom_height" not in st.session_state:
        st.session_state.custom_height = 768
    if "steps" not in st.session_state:
        st.session_state.steps = 9
    if "seed_input" not in st.session_state:
        st.session_state.seed_input = -1
    if "mode" not in st.session_state:
        st.session_state.mode = "장수 기준"
    if "target_count" not in st.session_state:
        st.session_state.target_count = 1
    if "target_seconds" not in st.session_state:
        st.session_state.target_seconds = 60

    st.title("🚀 Z-Image Pro : Studio")
    st.markdown("fal.ai API를 활용한 테스트용 화면입니다.")

    with st.sidebar:
        st.header("🔑 인증 & 설정")
        st.caption("API Key는 백엔드(환경변수)에서만 설정됩니다.")
        st.divider()

        st.subheader("🎨 이미지 설정")
        prompt = st.text_area(
            "프롬프트 (Prompt)",
            height=120,
            key="prompt",
        )

        ratio_name = st.selectbox(
            "화면비 (Aspect Ratio)",
            list(ASPECT_RATIOS.keys()),
            key="ratio_name",
        )
        if ratio_name == "Custom":
            width = st.number_input(
                "Width",
                min_value=64,
                max_value=2048,
                step=64,
                key="custom_width",
            )
            height = st.number_input(
                "Height",
                min_value=64,
                max_value=2048,
                step=64,
                key="custom_height",
            )
        else:
            width, height = ASPECT_RATIOS[ratio_name]

        col1, col2 = st.columns(2)
        with col1:
            steps = st.number_input("Steps", 1, 50, key="steps")
        with col2:
            seed_input = st.number_input("Seed (-1=Random)", step=1, key="seed_input")

        st.divider()

        st.subheader("⚙️ 배치 작업 설정")
        mode = st.radio("작업 모드", ["장수 기준", "시간 기준"], key="mode")

        if mode == "장수 기준":
            target_val = st.number_input("목표 장수", 1, 1000, key="target_count")
            is_time_mode = False
        else:
            target_val = st.number_input("목표 시간 (초)", 10, 3600, key="target_seconds")
            is_time_mode = True

        cost_unit, mp_unit = calculate_cost(int(width), int(height))
        st.session_state.current_width = int(width)
        st.session_state.current_height = int(height)
        st.info(
            "\n".join(
                [
                    "💰 **예상 단가**",
                    "",
                    f"규격: {int(width)}x{int(height)} ({mp_unit:.2f} MP)",
                    f"1장당: **${cost_unit:.5f}**",
                ]
            )
        )

        # 설정 가져오기/내보내기 UI는 현재 비활성화

        start_btn = st.button("🔥 작업 시작", type="primary", use_container_width=True)

    summary_area = st.empty()
    gallery_area = st.empty()
    log_area = st.empty()

    def render_outputs() -> None:
        st.session_state.render_seq += 1
        render_seq = st.session_state.render_seq
        with summary_area.container():
            if st.session_state.is_running:
                progress_text = (
                    f"작업 중 · {st.session_state.mode_label} · "
                    f"{st.session_state.current_task}번째 진행"
                )
                if st.session_state.mode_label.startswith("시간 기준"):
                    elapsed = st.session_state.elapsed_seconds
                    remaining = max(st.session_state.total_seconds - elapsed, 0)
                    progress_text += f" · 경과 {elapsed:.0f}s / 남음 {remaining:.0f}s"
                st.info(progress_text)
            elif st.session_state.total_tasks:
                st.info(
                    f"작업 완료 · {st.session_state.mode_label} · "
                    f"{st.session_state.total_tasks}건 처리"
                )

            if st.session_state.generated_count:
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("총 생성", f"{st.session_state.generated_count}장")
                with metric_col2:
                    st.metric("총 비용", f"${st.session_state.total_cost:.4f}")
                with metric_col3:
                    st.metric(st.session_state.speed_label, st.session_state.speed_value)

        with log_area.container():
            st.subheader("🧾 로그")
            if st.session_state.log_lines and not st.session_state.is_running:
                st.code("\n".join(st.session_state.log_lines[-400:]))
            elif st.session_state.is_running:
                st.caption("작업 진행 중입니다. 완료 후 요약 로그가 표시됩니다.")
            else:
                st.caption("작업 완료 후 요약 로그가 표시됩니다.")

        if st.session_state.images_history and not st.session_state.is_running:
            with gallery_area.container():
                with st.expander("📸 이미지 보기", expanded=False):
                    st.caption("검색 예시: 1,3,5-7")
                    filter_text = st.text_input(
                        "이미지 번호 필터",
                        placeholder="예: 1,3,5-7",
                        key="image_filter_text",
                    )
                    selected_indices = parse_index_filter(filter_text)
                    images_history = sorted(
                        st.session_state.images_history,
                        key=lambda item: item.get("index", 0),
                    )
                    if selected_indices:
                        images_history = [
                            item
                            for item in images_history
                            if item.get("index") in selected_indices
                        ]
                    if len(images_history) == 1:
                        item = images_history[0]
                        image_index = item.get("index", 1)
                        st.image(
                            item["url"],
                            caption=f"{image_index}번 · ⏱️{item['time']:.2f}s",
                            width=900,
                        )
                        file_bytes = fetch_image_bytes(item["url"])
                        if file_bytes:
                            st.download_button(
                                label=f"💾 {item['filename']}",
                                data=file_bytes,
                                file_name=item["filename"],
                                mime="image/png",
                                key=f"download_single_{render_seq}_{item['filename']}",
                            )
                    elif len(images_history) == 2:
                        cols = st.columns(2)
                        for col_idx, item in enumerate(images_history):
                            with cols[col_idx]:
                                image_index = item.get("index", col_idx + 1)
                                st.image(
                                    item["url"],
                                    caption=f"{image_index}번 · ⏱️{item['time']:.2f}s",
                                    width=520,
                                )
                                file_bytes = fetch_image_bytes(item["url"])
                                if file_bytes:
                                    st.download_button(
                                        label=f"💾 {item['filename']}",
                                        data=file_bytes,
                                        file_name=item["filename"],
                                        mime="image/png",
                                        key=f"download_{render_seq}_2_{col_idx}",
                                    )
                    else:
                        for row_start in range(0, len(images_history), 3):
                            row_items = images_history[row_start : row_start + 3]
                            cols = st.columns(3)
                            for col_idx, item in enumerate(row_items):
                                with cols[col_idx]:
                                    image_index = item.get("index", col_idx + 1)
                                    st.image(
                                        item["url"],
                                        caption=f"{image_index}번 · ⏱️{item['time']:.2f}s",
                                        width=300,
                                    )
                                    file_bytes = fetch_image_bytes(item["url"])
                                    if file_bytes:
                                        st.download_button(
                                            label=f"💾 {item['filename']}",
                                            data=file_bytes,
                                            file_name=item["filename"],
                                            mime="image/png",
                                            key=f"download_{render_seq}_{row_start}_{col_idx}",
                                        )

    if start_btn:
        if not os.environ.get("FAL_KEY"):
            st.error("⚠️ 서버 환경변수에 FAL_KEY가 설정되어 있어야 합니다!")
            st.stop()

        st.session_state.log_lines = []
        st.session_state.images_history = []
        st.session_state.generated_count = 0
        st.session_state.total_cost = 0.0
        st.session_state.is_running = True
        st.session_state.current_task = 0
        st.session_state.total_tasks = target_val
        st.session_state.mode_label = "장수 기준" if not is_time_mode else f"시간 기준 {target_val}초"
        st.session_state.elapsed_seconds = 0.0
        st.session_state.total_seconds = float(target_val) if is_time_mode else 0.0

        start_time = time.time()
        generated_count = 0
        total_cost = 0.0
        images_history = []
        results = []

        try:
            while True:
                elapsed = time.time() - start_time
                st.session_state.elapsed_seconds = elapsed

                if is_time_mode:
                    if elapsed >= target_val:
                        break
                else:
                    if generated_count >= target_val:
                        break

                current_seed = (
                    seed_input if seed_input != -1 else int(time.time() * 1000) % 2**31
                )

                image_start_time = time.time()

                try:
                    img_url, infer_time = generate_image_fal(
                        prompt, width, height, steps, current_seed
                    )

                    filename = (
                        f"{datetime.now().strftime('%H%M%S')}_"
                        f"{generated_count + 1:03d}_{current_seed}.png"
                    )

                    generated_count += 1
                    total_cost += cost_unit
                    st.session_state.generated_count = generated_count
                    st.session_state.total_cost = total_cost
                    st.session_state.current_task = generated_count

                    images_history.append(
                        {
                            "url": img_url,
                            "time": infer_time,
                            "filename": filename,
                            "index": generated_count,
                        }
                    )
                    st.session_state.images_history = images_history

                    results.append(
                        {
                            "time": infer_time,
                            "success": True,
                            "filename": filename,
                        }
                    )

                    if generated_count == 1:
                        st.session_state.speed_label = "생성 속도"
                        st.session_state.speed_value = f"{infer_time:.2f}s"
                    else:
                        st.session_state.speed_label = "평균 속도"
                        st.session_state.speed_value = f"{elapsed / generated_count:.2f}s/장"

                    render_outputs()

                except Exception as exc:
                    fail_time = time.time() - image_start_time
                    results.append(
                        {
                            "time": fail_time,
                            "success": False,
                            "filename": "-",
                        }
                    )
                    time.sleep(1)

        except KeyboardInterrupt:
            st.warning("작업이 사용자에 의해 중단되었습니다.")

        end_time = time.time()
        total_elapsed = end_time - start_time
        successful = [r for r in results if r["success"]]
        if successful:
            times = [r["time"] for r in successful]
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            std_dev = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
            images_per_hour = 3600 / avg_time if avg_time > 0 else 0
        else:
            avg_time = min_time = max_time = std_dev = images_per_hour = 0.0

        megapixels = (int(width) * int(height)) / 1_000_000
        cost_per_image = megapixels * PRICE_PER_MP
        total_test_cost = len(successful) * cost_per_image
        cost_per_hour = images_per_hour * cost_per_image

        start_time_str = datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")
        end_time_str = datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")
        cold_boots = sum(1 for r in results if r["time"] >= 2.0)
        settings_payload = {
            "prompt": prompt,
            "ratio_name": ratio_name,
            "width": int(width),
            "height": int(height),
            "steps": steps,
            "seed_input": seed_input,
            "mode": "장수 기준" if not is_time_mode else "시간 기준",
            "target_value": int(target_val),
        }

        log_lines = [
            f"\n{'=' * 60}",
            "📊 테스트 결과 요약",
            f"{'=' * 60}",
            f"시작 시간: {start_time_str}",
            f"완료 시간: {end_time_str}",
            f"총 소요 시간: {total_elapsed:.2f}초",
            f"cold boots: {cold_boots} (>=2.0초)",
            "\n🎯 성공률:",
            f"  {len(successful)}/{len(results)} ({(len(successful) / len(results) * 100):.1f}%)"
            if results
            else "  0/0 (0.0%)",
            "\n⏱️  성능 지표:",
            f"  평균 생성시간: {avg_time:.2f}초",
            f"  최소 시간: {min_time:.2f}초",
            f"  최대 시간: {max_time:.2f}초",
            f"  표준편차: {std_dev:.2f}초",
            f"  시간당 생성 가능: {images_per_hour:.0f}장",
            "\n💰 비용 분석:",
            f"  해상도: {int(width)}x{int(height)} ({megapixels:.2f}MP)",
            f"  이미지당 비용: ${cost_per_image:.4f}",
            f"  이번 테스트 총 비용: ${total_test_cost:.4f} ({len(successful)}장)",
            f"  시간당 예상 비용: ${cost_per_hour:.2f}",
            "\n✅ 결과 요약:",
            f"  성공: {len(successful)}장",
            f"  실패: {len(results) - len(successful)}장",
        ]

        results_payload = {
            "total": len(results),
            "success": len(successful),
            "failed": len(results) - len(successful),
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "std_dev": std_dev,
            "images_per_hour": images_per_hour,
            "megapixels": megapixels,
            "cost_per_image": cost_per_image,
            "total_test_cost": total_test_cost,
            "cost_per_hour": cost_per_hour,
            "start_time": start_time_str,
            "end_time": end_time_str,
            "total_elapsed": total_elapsed,
            "cold_boots": cold_boots,
        }
        log_file_path = write_log_file(log_lines, settings_payload, results_payload)
        st.session_state.log_lines = log_lines
        st.session_state.log_file_path = log_file_path
        st.session_state.is_running = False
        st.session_state.total_tasks = len(results)

        render_outputs()

        with st.container():
            st.subheader("📝 프롬프트")
            st.info(prompt)

    if not start_btn:
        render_outputs()

    if not st.session_state.is_running and getattr(st.session_state, "log_file_path", ""):
        with st.container():
            st.subheader("📁 로그 파일 위치")
            st.info(os.path.abspath(st.session_state.log_file_path))


if __name__ == "__main__":
    main()
