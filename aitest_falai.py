import argparse
import json
import time
from datetime import datetime

import fal
import fal_client
from fal.toolkit.image import Image
from pydantic import BaseModel, Field


class HelloWorldApp(fal.App):
    @fal.endpoint("/hello")
    def run(self) -> dict:
        return {"message": "Hello, World!"}


class ImageRequest(BaseModel):
    prompt: str = Field(description="Image generation prompt")
    num_inference_steps: int = Field(default=28, description="Number of inference steps")
    width: int = Field(default=1024, description="Image width")
    height: int = Field(default=1024, description="Image height")


class ImageResponse(BaseModel):
    image: Image


class StableDiffusionApp(fal.App):
    # GPU 선택 (예: "GPU-H100", "GPU-A100", "GPU-L4")
    machine_type = "GPU-H100"

    requirements = [
        "diffusers==0.30.3",
        "torch==2.6.0",
        "transformers==4.47.1",
        "accelerate",
    ]

    async def setup(self) -> None:
        from diffusers import StableDiffusionPipeline
        import torch

        print("Loading model...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype=torch.float16,
        ).to("cuda")
        print("Model loaded successfully!")

    @fal.endpoint("/")
    async def generate(self, request: ImageRequest) -> ImageResponse:
        print(f"Generating image with prompt: {request.prompt}")
        result = self.pipe(
            prompt=request.prompt,
            num_inference_steps=request.num_inference_steps,
            width=request.width,
            height=request.height,
        )
        image = result.images[0]
        return ImageResponse(image=Image.from_pil(image))


class GPUPerformanceTest:
    def __init__(self, app_url: str, gpu_name: str):
        self.app_url = app_url
        self.gpu_name = gpu_name
        self.results = []

    def run_test(self, num_tests: int = 10, prompt: str = "a cat wearing a hat") -> None:
        print(f"\n{'=' * 50}")
        print(f"테스트 시작: {self.gpu_name}")
        print(f"시간: {datetime.now()}")
        print(f"{'=' * 50}\n")

        for i in range(num_tests):
            print(f"테스트 {i + 1}/{num_tests}...")
            start_time = time.time()

            try:
                result = fal_client.submit(
                    self.app_url,
                    arguments={
                        "prompt": f"{prompt} {i}",
                        "num_inference_steps": 28,
                    },
                )
                output = result.get()
                elapsed_time = time.time() - start_time

                self.results.append(
                    {
                        "test_num": i + 1,
                        "time": elapsed_time,
                        "success": True,
                        "image_url": output.get("image", {}).get("url", ""),
                    }
                )
                print(f"  ✓ 완료: {elapsed_time:.2f}초")
            except Exception as exc:
                elapsed_time = time.time() - start_time
                self.results.append(
                    {
                        "test_num": i + 1,
                        "time": elapsed_time,
                        "success": False,
                        "error": str(exc),
                    }
                )
                print(f"  ✗ 실패: {exc}")

        self.print_summary()
        self.save_results()

    def print_summary(self) -> None:
        successful_results = [r for r in self.results if r["success"]]

        if not successful_results:
            print("\n⚠️ 모든 테스트 실패")
            return

        times = [r["time"] for r in successful_results]
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        images_per_hour = 3600 / avg_time if avg_time > 0 else 0

        print(f"\n{'=' * 50}")
        print(f"📊 테스트 결과 요약: {self.gpu_name}")
        print(f"{'=' * 50}")
        print(f"성공: {len(successful_results)}/{len(self.results)}")
        print(f"평균 생성시간: {avg_time:.2f}초")
        print(f"최소 생성시간: {min_time:.2f}초")
        print(f"최대 생성시간: {max_time:.2f}초")
        print(f"시간당 생성량: {images_per_hour:.0f}장")
        print(f"{'=' * 50}\n")

    def save_results(self) -> None:
        filename = f"test_results_{self.gpu_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "gpu": self.gpu_name,
                    "test_count": len(self.results),
                    "results": self.results,
                },
                file,
                indent=2,
                ensure_ascii=False,
            )
        print(f"결과 저장: {filename}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="fal.ai 테스트 유틸리티")
    parser.add_argument(
        "--performance",
        action="store_true",
        help="GPU 성능 테스트 실행",
    )
    parser.add_argument(
        "--app-url",
        default="username/h100-app",
        help="배포된 앱 URL (예: username/h100-app)",
    )
    parser.add_argument(
        "--gpu-name",
        default="H100",
        help="GPU 이름 (예: H100)",
    )
    parser.add_argument(
        "--num-tests",
        type=int,
        default=10,
        help="테스트 횟수",
    )
    parser.add_argument(
        "--prompt",
        default="a cat wearing a hat",
        help="기본 프롬프트",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.performance:
        print("⚠️ 테스트 전에 https://fal.ai/dashboard/billing 에서 크레딧을 기록하세요!")
        input("준비되면 Enter를 누르세요...")
        tester = GPUPerformanceTest(args.app_url, args.gpu_name)
        tester.run_test(num_tests=args.num_tests, prompt=args.prompt)
