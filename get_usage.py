import argparse
import json
import os
from datetime import datetime, timedelta

import requests


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="fal Usage API 조회")
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="최근 N일 조회 (기본: 1일)",
    )
    parser.add_argument(
        "--timeframe",
        default="hour",
        help="집계 단위 (minute, hour, day 등)",
    )
    parser.add_argument(
        "--endpoint-id",
        default="",
        help="특정 엔드포인트 필터 (예: fal-ai/z-image/turbo)",
    )
    parser.add_argument(
        "--timezone",
        default="UTC",
        help="시간대 (기본: UTC)",
    )
    return parser.parse_args()


def main() -> None:
    api_key = os.environ.get("FAL_USAGE_KEY") or os.environ.get("FAL_KEY")
    if not api_key:
        print("❌ FAL_USAGE_KEY 또는 FAL_KEY 환경변수가 설정되지 않았습니다!")
        raise SystemExit(1)

    headers = {
        "Authorization": f"Key {api_key}",
        "Content-Type": "application/json",
    }

    args = _parse_args()
    end_time = datetime.now()
    start_time = end_time - timedelta(days=args.days)

    params = {
        "start": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "expand": "time_series,summary",
        "timeframe": args.timeframe,
        "timezone": args.timezone,
    }
    if args.endpoint_id:
        params["endpoint_id"] = args.endpoint_id

    print(f"📊 사용량 조회 중... ({start_time:%Y-%m-%d} ~ {end_time:%Y-%m-%d})")
    print("=" * 60)

    fal_host = os.environ.get("FAL_HOST", "api.fal.ai").strip()
    base_url = f"https://{fal_host}"
    response = requests.get(
        f"{base_url}/v1/models/usage",
        headers=headers,
        params=params,
        timeout=30,
    )

    if response.status_code != 200:
        print(f"❌ API 호출 실패: {response.status_code}")
        print(f"응답: {response.text}")
        raise SystemExit(1)

    try:
        data = response.json()
    except ValueError:
        print("❌ JSON 파싱 실패 (응답이 JSON이 아닙니다).")
        print(f"응답: {response.text}")
        raise SystemExit(1)

    with open("usage_raw.json", "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    print("💾 전체 응답 저장: usage_raw.json\n")

    if "summary" in data:
        summary = data["summary"]
        print("📈 오늘 사용량 요약:")
        if isinstance(summary, list):
            total_cost = sum(item.get("cost", 0) for item in summary)
            total_units = sum(item.get("quantity", 0) for item in summary)
            print(f"  총 비용: ${total_cost:.4f}")
            print(f"  총 단위: {total_units}")
        else:
            print(f"  총 비용: ${summary.get('total_cost', 0):.4f}")
            print(f"  총 요청: {summary.get('total_requests', 0)}회")
            print(f"  총 단위: {summary.get('total_units', 0)}")
        print()

    if "time_series" in data:
        print("⏰ 시간별 상세 내역:")
        print("-" * 60)

        total_units = 0
        total_cost = 0

        for entry in data["time_series"]:
            timestamp = entry["bucket"]
            results = entry.get("results", [])

            if not results:
                continue

            print(f"\n🕐 {timestamp}")

            for result in results:
                endpoint = result.get("endpoint_id", "unknown")
                quantity = result.get("quantity", 0)
                cost = result.get("cost", 0)
                unit = result.get("unit", "unknown")
                unit_price = result.get("unit_price", 0)

                total_units += quantity
                total_cost += cost

                print(f"  모델: {endpoint}")
                print(f"  수량: {quantity} {unit}")
                print(f"  단가: ${unit_price:.4f}/{unit}")
                print(f"  비용: ${cost:.4f}")

        print("\n" + "=" * 60)
        print("📊 총계:")
        print(f"  총 단위: {total_units}")
        print(f"  총 비용: ${total_cost:.4f}")
        if total_units > 0:
            print(f"  단위당 평균: ${total_cost / total_units:.4f}")
        print("=" * 60)
    else:
        print("⚠️  시간별 데이터가 없습니다.")

    print("\n💡 더 상세한 정보는 usage_raw.json 파일을 확인하세요.")


if __name__ == "__main__":
    main()
