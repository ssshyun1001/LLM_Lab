"""
Day 2: Prompt Engineering - zero-shot / few-shot / Chain-of-Thought(CoT) 비교

이 코드에서 배우는 것:
- zero-shot: 예시 없이 그냥 질문만 던지는 방식
- few-shot: 질문 전에 "예시 몇 개"를 미리 보여주고 패턴을 따라하게 하는 방식
- CoT(Chain-of-Thought): "단계별로 생각해봐"라고 유도해서 추론 과정을 거치게 하는 방식

핵심 아이디어:
  세 방식 모두 "같은 모델"을 쓰지만, messages에 무엇을 어떻게 담아 보내느냐에 따라
  답변의 정확도와 품질이 크게 달라진다는 걸 직접 눈으로 확인하는 게 목표입니다.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"

# 실습용 문제: 모델이 종종 헷갈리는 간단한 산수/논리 문제를 씀
# (일부러 "직관적으로는 틀리기 쉬운" 문제를 골랐음 - 전략 차이가 잘 드러나도록)
PROBLEM = (
    "한 상자에 사과가 23개 있었다. 그중 12개를 먹고, "
    "다시 15개를 새로 사서 넣었다. 그 후 상자에 있던 사과의 절반을 친구에게 줬다. "
    "지금 상자에 남은 사과는 몇 개야? 최종 숫자만 답해."
)


# ------------------------------------------------------------------
# 방법 1: Zero-shot
# 예시(shot)를 하나도 안 주고, 질문만 그대로 던지는 가장 단순한 방식.
# 장점: 빠르고 프롬프트가 짧음 (토큰 절약)
# 단점: 모델이 형식을 잘못 이해하거나, 복잡한 문제에서 실수할 확률이 더 높음
# ------------------------------------------------------------------
def zero_shot():
    # messages에 system 없이 user 질문 하나만 담아서 보냄
    # -> 모델은 아무 힌트 없이 "알아서" 답변 형식을 정함
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,  # 비교 실험이므로 무작위성을 없애서 결과를 일관되게 만듦
        messages=[
            {"role": "user", "content": PROBLEM}
        ],
    )
    return response.choices[0].message.content.strip()


# ------------------------------------------------------------------
# 방법 2: Few-shot
# 실제 질문 전에 "이런 식으로 풀면 돼"라는 예시(shot)를 몇 개 먼저 보여줌.
# 모델은 예시의 패턴(문제 -> 풀이 -> 답 형식)을 그대로 따라하려는 경향이 있음.
# 장점: 원하는 출력 형식을 강하게 유도할 수 있음
# 단점: 프롬프트가 길어짐 (토큰 비용 증가)
# ------------------------------------------------------------------
def few_shot():
    # 예시 1, 2를 user/assistant 역할로 미리 "가짜 대화"처럼 넣어줌
    # -> 모델 입장에서는 "아, 이전에 이렇게 풀어줬었지" 하고 패턴을 학습(모방)함
    messages = [
        {
            "role": "user",
            "content": "바구니에 사탕이 10개 있다. 4개를 먹고 6개를 더 샀다. 지금 몇 개?",
        },
        {
            "role": "assistant",
            "content": "10 - 4 = 6, 6 + 6 = 12. 답: 12",
        },
        {
            "role": "user",
            "content": "책이 30권 있다. 10권을 팔고, 5권을 기부받았다. 그 후 절반을 서점에 반납했다. 지금 몇 권?",
        },
        {
            "role": "assistant",
            "content": "30 - 10 = 20, 20 + 5 = 25, 25 / 2 = 12.5 -> 반납 후 남은 건 12.5권이지만 책은 정수이므로 12권 또는 13권. 여기선 12로 계산. 답: 12",
        },
        # 위 두 개의 예시(few-shot examples)를 보여준 뒤, 실제로 풀어야 할 문제를 마지막에 넣음
        {"role": "user", "content": PROBLEM},
    ]

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=messages,
    )
    return response.choices[0].message.content.strip()


# ------------------------------------------------------------------
# 방법 3: Chain-of-Thought (CoT)
# "단계별로 차근차근 생각해봐"라는 지시를 추가해서,
# 모델이 최종 답만 뱉지 않고 중간 추론 과정을 거치도록 유도함.
# 사람이 암산보다 손으로 풀 때 실수가 줄어드는 것과 비슷한 원리.
# 장점: 복잡한 논리/수학 문제에서 정확도가 눈에 띄게 올라가는 경우가 많음
# 단점: 답변이 길어짐 (토큰 비용 증가), 항상 정확도가 오르는 건 아님
# ------------------------------------------------------------------
def chain_of_thought():
    # 핵심은 프롬프트 마지막에 붙인 한 문장:
    # "단계별로 하나씩 계산 과정을 보여준 다음, 마지막 줄에 최종 답을 알려줘."
    # 이 지시 하나가 모델의 "사고 과정"을 겉으로 드러나게 만듦
    cot_prompt = PROBLEM + "\n\n단계별로 하나씩 계산 과정을 보여준 다음, 마지막 줄에 '답: 숫자' 형식으로 최종 답을 알려줘."

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "user", "content": cot_prompt}
        ],
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    print("=== 문제 ===")
    print(PROBLEM)
    print("\n(정답 계산: (23-12+15)/2 = 26/2 = 13)\n")

    print("=" * 60)
    print("[1] Zero-shot 결과")
    print("=" * 60)
    print(zero_shot())

    print("\n" + "=" * 60)
    print("[2] Few-shot 결과")
    print("=" * 60)
    print(few_shot())

    print("\n" + "=" * 60)
    print("[3] Chain-of-Thought 결과")
    print("=" * 60)
    print(chain_of_thought())
