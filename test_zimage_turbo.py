import argparse
import json
import os
import time
from datetime import datetime

import fal_client


class ZImageTurboTest:
    """Z-Image Turbo 성능 및 비용 테스트"""

    def __init__(self) -> None:
        self.model_name = "fal-ai/z-image/turbo"
        self.results = []

    def run_test(
        self,
        num_tests: int = 10,
        resolution: str = "landscape_16_9",
        num_inference_steps: int = 8,
        guidance_scale: float | None = None,
        sampler: str | None = None,
        scheduler: str | None = None,
        denoise: float | None = None,
        seed: int | None = None,
    ):
        """
        성능 측정

        resolution 옵션:
        - square_hd: 1024x1024 (1MP)
        - square: 512x512 (0.25MP)
        - portrait_16_9: 576x1024 (~0.59MP)
        - landscape_16_9: 1024x576 (~0.59MP)
        - landscape_4_3: 1024x768 (~0.79MP)
        """
        print(f"\n{'=' * 60}")
        print("Z-Image Turbo 성능 테스트")
        print(f"{'=' * 60}")
        print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"모델: {self.model_name}")
        print(f"해상도: {resolution}")
        print(f"테스트 횟수: {num_tests}회")
        print(f"스텝 수: {num_inference_steps}")
        if guidance_scale is not None:
            print(f"CFG: {guidance_scale}")
        if sampler:
            print(f"샘플러: {sampler}")
        if scheduler:
            print(f"스케줄러: {scheduler}")
        if denoise is not None:
            print(f"노이즈 제거량: {denoise}")
        if seed is not None:
            print(f"시드: {seed}")
        print(f"{'=' * 60}\n")

        for i in range(num_tests):
            print(f"[{i + 1}/{num_tests}] 이미지 생성 중...", end=" ")
            start_time = time.time()

            try:
                arguments = {
                    "prompt": (
                        "cute anime style girl with massive fluffy fennec ears and a big fluffy tail "
                        "blonde messy long hair blue eyes wearing a maid outfit with a long black "
                        'gold leaf pattern dress and a white apron, it is a postcard held by a hand '
                        'in front of a beautiful realistic city at sunset and there is cursive writing '
                        'that says "ZImage, Now in ComfyUI"'
                    ),
                    "image_size": resolution,
                    "num_inference_steps": num_inference_steps,
                    "num_images": 1,
                    "enable_safety_checker": True,
                }
                if guidance_scale is not None:
                    arguments["guidance_scale"] = guidance_scale
                if sampler:
                    arguments["sampler"] = sampler
                if scheduler:
                    arguments["scheduler"] = scheduler
                if denoise is not None:
                    arguments["denoise"] = denoise
                if seed is not None:
                    arguments["seed"] = seed

                result = fal_client.subscribe(self.model_name, arguments=arguments)
                output = result
                elapsed = time.time() - start_time

                self.results.append(
                    {
                        "test_num": i + 1,
                        "time": elapsed,
                        "success": True,
                        "image_url": output["images"][0]["url"] if output.get("images") else None,
                    }
                )
                print(f"✓ {elapsed:.2f}초")
            except Exception as exc:
                elapsed = time.time() - start_time
                self.results.append(
                    {
                        "test_num": i + 1,
                        "time": elapsed,
                        "success": False,
                        "error": str(exc),
                    }
                )
                print(f"✗ 실패: {exc}")

            time.sleep(0.5)

        return self.print_summary(resolution)

    def print_summary(self, resolution: str):
        """결과 요약 출력"""
        successful = [r for r in self.results if r["success"]]

        if not successful:
            print("\n❌ 모든 테스트 실패")
            return None

        times = [r["time"] for r in successful]
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        std_dev = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
        images_per_hour = 3600 / avg_time if avg_time > 0 else 0

        resolution_mp = {
            "square_hd": 1.0,
            "square": 0.25,
            "portrait_16_9": 0.589824,
            "landscape_16_9": 0.589824,
            "landscape_4_3": 0.786432,
        }

        megapixels = resolution_mp.get(resolution, 0.589824)
        cost_per_image = megapixels * 0.005
        cost_per_hour = images_per_hour * cost_per_image
        total_test_cost = len(successful) * cost_per_image

        print(f"\n{'=' * 60}")
        print("📊 테스트 결과 요약")
        print(f"{'=' * 60}")
        print(f"완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n🎯 성공률:")
        print(f"  {len(successful)}/{len(self.results)} ({len(successful) / len(self.results) * 100:.1f}%)")
        print("\n⏱️  성능 지표:")
        print(f"  평균 생성시간: {avg_time:.2f}초")
        print(f"  최소 시간: {min_time:.2f}초")
        print(f"  최대 시간: {max_time:.2f}초")
        print(f"  표준편차: {std_dev:.2f}초")
        print(f"  시간당 생성 가능: {images_per_hour:.0f}장")
        print("\n💰 비용 분석:")
        print(f"  해상도: {resolution} ({megapixels:.2f}MP)")
        print(f"  이미지당 비용: ${cost_per_image:.4f}")
        print(f"  이번 테스트 총 비용: ${total_test_cost:.4f} ({len(successful)}장)")
        print(f"  시간당 예상 비용: ${cost_per_hour:.2f}")
        print(f"{'=' * 60}\n")

        print("📋 상세 결과:")
        print(f"{'테스트':<8} {'시간(초)':<10} {'상태':<8}")
        print("-" * 60)
        for result in self.results:
            status = "✓ 성공" if result["success"] else "✗ 실패"
            print(f"{result['test_num']:<8} {result['time']:<10.2f} {status:<8}")
        print()

        self.save_results(
            {
                "model": self.model_name,
                "resolution": resolution,
                "megapixels": megapixels,
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "std_dev": std_dev,
                "images_per_hour": images_per_hour,
                "cost_per_image": cost_per_image,
                "cost_per_hour": cost_per_hour,
                "total_test_cost": total_test_cost,
                "success_rate": len(successful) / len(self.results) * 100,
            }
        )

        return {
            "avg_time": avg_time,
            "images_per_hour": images_per_hour,
            "cost_per_image": cost_per_image,
        }

    def save_results(self, summary):
        """결과를 JSON 파일로 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"zimage_turbo_test_{timestamp}.json"

        data = {
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "detailed_results": self.results,
        }

        with open(filename, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)

        print(f"💾 결과 저장: {filename}\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Z-Image Turbo 테스트")
    parser.add_argument("--num-tests", type=int, default=10, help="테스트 횟수")
    parser.add_argument(
        "--resolution",
        default="square_hd",
        help="해상도 옵션 (square, square_hd, portrait_16_9, landscape_16_9, landscape_4_3)",
    )
    parser.add_argument("--steps", type=int, default=9, help="스텝 수")
    parser.add_argument("--cfg", type=float, default=1.0, help="CFG 스케일")
    parser.add_argument("--sampler", default="euler", help="샘플러 이름")
    parser.add_argument("--scheduler", default="simple", help="스케줄러 이름")
    parser.add_argument("--denoise", type=float, default=1.0, help="노이즈 제거량")
    parser.add_argument(
        "--seed",
        default="random",
        help="시드 값 (숫자 또는 'random')",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="준비 확인 프롬프트를 건너뜁니다",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Z-Image Turbo 테스트 프로그램")
    print("=" * 60)
    print("\n📌 테스트 전 체크리스트:")
    print("  1. ✅ fal.ai 회원가입 완료")
    print("  2. ✅ 빌링 정보 등록 완료")
    print("  3. ✅ API Key 발급 완료")
    print("  4. ✅ FAL_KEY 환경변수 설정 완료")
    print("  5. ⚠️  https://fal.ai/dashboard/billing 에서 현재 크레딧 기록")
    print()

    if not os.environ.get("FAL_KEY"):
        print("❌ 오류: FAL_KEY 환경변수가 설정되지 않았습니다!")
        print("   export FAL_KEY='your-api-key' 를 실행하세요.")
        raise SystemExit(1)

    args = _parse_args()
    if not args.yes:
        input("✅ 모든 준비가 완료되었으면 Enter를 누르세요...")

    seed_value = None
    if isinstance(args.seed, str) and args.seed.lower() != "random":
        try:
            seed_value = int(args.seed)
        except ValueError:
            print("❌ 오류: --seed는 숫자 또는 'random' 이어야 합니다.")
            raise SystemExit(1)

    tester = ZImageTurboTest()
    tester.run_test(
        num_tests=args.num_tests,
        resolution=args.resolution,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        sampler=args.sampler,
        scheduler=args.scheduler,
        denoise=args.denoise,
        seed=seed_value,
    )

    print("=" * 60)
    print("✅ 테스트 완료!")
    print("=" * 60)
    print("\n📊 다음 단계:")
    print("  1. https://fal.ai/dashboard/billing 접속")
    print("  2. 실제 사용 비용 확인")
    print("  3. 저장된 JSON 파일로 상세 분석")
    print()
