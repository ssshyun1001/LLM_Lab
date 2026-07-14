"""
Day 1-1: 환경설정 + 첫 API 호출

이 코드에서 배우는 것:
- .env 파일에서 API 키를 안전하게 불러오는 법
- openai 클라이언트 초기화
- 가장 단순한 chat.completions.create() 호출

사전 준비:
1. pip install -r ../requirements.txt
2. ..env 을 복사해서 .env 파일 만들고 실제 API 키 입력
   (OpenAI 대시보드 > API keys 에서 발급: https://platform.openai.com/api-keys)
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. .env 파일에서 환경변수 불러오기
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY가 설정되지 않았습니다. "
        ".env 파일을 만들고 API 키를 넣었는지 확인하세요."
    )

# 2. 클라이언트 초기화
client = OpenAI(api_key=api_key)


# 3. 가장 단순한 호출: 메시지 하나 보내고 응답 받기
def first_call():
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # 저렴하고 빠른 모델로 실습 (비용 절약)
        messages=[
            {"role": "user", "content": "너는 어떤 모델이고, 지금 무슨 일을 할 수 있어? 3줄로 요약해줘."}
        ],
    )
    return response


if __name__ == "__main__":
    print("=== OpenAI API 첫 호출 테스트 ===\n")

    response = first_call()

    # 응답에서 실제 텍스트만 꺼내기
    message = response.choices[0].message.content
    print("모델 응답:")
    print(message)

    # 참고: 응답 객체 안에는 사용된 토큰 수 등 메타 정보도 들어있음
    print("\n--- 사용된 토큰 정보 ---")
    print(f"입력 토큰: {response.usage.prompt_tokens}")
    print(f"출력 토큰: {response.usage.completion_tokens}")
    print(f"총 토큰: {response.usage.total_tokens}")