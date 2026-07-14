"""
Day 1-2: Chat Completion 기본기

이 코드에서 배우는 것:
- system / user / assistant role의 역할 차이
- temperature 파라미터가 응답에 미치는 영향
- messages 리스트에 대화 기록을 누적해서 멀티턴 대화 구현하기

실행하면 두 가지를 확인할 수 있습니다:
  1) temperature 비교 실험
  2) 터미널에서 직접 대화할 수 있는 멀티턴 챗봇
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"


# ------------------------------------------------------------------
# 실험 1: temperature 비교
# temperature는 0~2 사이 값. 낮을수록 일관되고 예측 가능한 답변,
# 높을수록 창의적이고 다양한 답변이 나옴.
# ------------------------------------------------------------------
def compare_temperature():
    prompt = "회사 이름을 하나 지어줘. AI 스타트업이고, 이름만 짧게 대답해."

    print("=== Temperature 비교 실험 ===")
    print(f"질문: {prompt}\n")

    for temp in [0.0, 1.0, 1.8]:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=temp,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.choices[0].message.content.strip()
        print(f"[temperature={temp}] -> {answer}")

    print()


# ------------------------------------------------------------------
# 실험 2: system 메시지로 캐릭터/역할 부여하기
# system 메시지는 대화 전체의 "규칙"을 정하는 역할
# ------------------------------------------------------------------
def system_role_demo():
    print("=== System 메시지 효과 비교 ===")
    question = "파이썬이 뭐야? 2문장 이내로 답해."

    # system 메시지 없이
    plain = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
    )

    # system 메시지로 캐릭터 부여
    with_persona = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "너는 초등학생에게 설명하는 친절한 선생님이야. 쉬운 비유를 꼭 써서 2문장 이내로 답해.",
            },
            {"role": "user", "content": question},
        ],
    )

    print("[system 메시지 없음]")
    print(plain.choices[0].message.content.strip())
    print("\n[system 메시지로 '초등학생용 선생님' 페르소나 부여]")
    print(with_persona.choices[0].message.content.strip())
    print()


# ------------------------------------------------------------------
# 실험 3: 멀티턴 대화 (핵심)
# 대화 기록을 messages 리스트에 계속 누적해야 모델이 "기억"함
# (실제로는 매 요청마다 전체 히스토리를 다시 보내는 것 - 모델 자체엔 기억이 없음)
# ------------------------------------------------------------------
def multi_turn_chat():
    print("=== 멀티턴 대화 시작 (종료하려면 'exit' 입력) ===\n")

    # 대화 기록을 저장할 리스트. system 메시지로 챗봇 성격을 정의.
    messages = [
        {"role": "system", "content": "너는 친절하고 간결하게 답하는 AI 어시스턴트야."}
    ]

    while True:
        user_input = input("나: ")
        if user_input.strip().lower() == "exit":
            print("대화를 종료합니다.")
            break

        # 1. 사용자 메시지를 기록에 추가
        messages.append({"role": "user", "content": user_input})

        # 2. 전체 기록을 API에 전달 (이전 대화 전부 포함)
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )

        assistant_reply = response.choices[0].message.content

        # 3. 모델의 응답도 기록에 추가해야 다음 턴에서 "기억"함
        messages.append({"role": "assistant", "content": assistant_reply})

        print(f"AI: {assistant_reply}\n")


if __name__ == "__main__":
    compare_temperature()
    system_role_demo()
    multi_turn_chat()