"""
Day 2 : Self-consistency

이 코드에서 배우는 것:
- 같은 질문을 "여러 번" 물어봐서 나온 답변들 중 다수결로 최종 답을 정하는 기법
- CoT(Chain-of-Thought)와 함께 쓰면 특히 효과적인 이유

핵심 아이디어:
  temperature를 0보다 높게 주면, 같은 질문이라도 모델이 매번 조금씩 다르게
  "생각의 경로"를 탐색함. 그 경로 중 일부는 실수를 하고, 일부는 맞음.
  이걸 여러 번(N번) 반복해서 제일 많이 나온 답을 고르면,
  한 번만 물어봤을 때보다 정답률이 올라가는 경우가 많음.

  비유: 사람도 어려운 문제를 한 번 풀고 끝내는 것보다,
        같은 문제를 다른 방식으로 여러 번 풀어보고 "가장 자주 나온 답"을
        고르면 실수를 줄일 수 있는 것과 비슷함.

주의:
- API 요청을 N번 보내는 방식이라 비용이 N배로 늘어남 (예: 5번 반복 = 토큰 비용 5배)
- 그래서 실무에서는 "정확도가 특히 중요한 소수의 질문"에만 선택적으로 씀
"""

import os
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"

# Day 2 첫 실습(01_zero_few_cot.py)과 같은 문제를 재사용
# -> CoT 단독으로 풀 때 vs self-consistency로 풀 때의 차이를 비교하기 위함
PROBLEM = (
    "한 상자에 사과가 23개 있었다. 그중 12개를 먹고, "
    "다시 15개를 새로 사서 넣었다. 그 후 상자에 있던 사과의 절반을 친구에게 줬다. "
    "지금 상자에 남은 사과는 몇 개야?"
)

# CoT 지시를 붙여서 모델이 풀이 과정을 거치도록 유도 (self-consistency는 보통 CoT와 세트로 씀)
COT_PROMPT = (
        PROBLEM
        + "\n\n단계별로 하나씩 계산 과정을 보여준 다음, 마지막 줄에 "
          "'답: 숫자' 형식으로만 최종 숫자를 알려줘."
)

N_SAMPLES = 5  # 같은 질문을 몇 번 반복해서 물어볼지 (많을수록 정확하지만 비용도 그만큼 증가)


# ------------------------------------------------------------------
# 하나의 답변에서 "답: 13" 같은 패턴을 찾아 숫자만 뽑아내는 함수
# 모델 출력이 매번 형식이 조금씩 다를 수 있어서, 최대한 안전하게 파싱함
# ------------------------------------------------------------------
def extract_final_answer(text: str) -> str:
    # "답:" 이후에 오는 부분만 잘라서 가져옴
    if "답:" in text:
        tail = text.split("답:")[-1].strip()
    else:
        tail = text.strip()

    # 숫자만 남기고 나머지(공백, 마침표, 단위 등)는 제거
    digits = "".join(ch for ch in tail if ch.isdigit())
    return digits if digits else "파싱 실패"


# ------------------------------------------------------------------
# 같은 질문을 N번 반복 호출 -> 각 답변에서 최종 숫자 추출 -> 다수결로 결정
# ------------------------------------------------------------------
def self_consistency(n_samples: int = N_SAMPLES):
    answers = []  # 매 시도마다 뽑아낸 최종 숫자를 저장
    raw_responses = []  # 원본 답변 전체도 참고용으로 저장

    for i in range(n_samples):
        response = client.chat.completions.create(
            model=MODEL,
            # temperature를 0보다 높게 줘야 매번 다른 "생각의 경로"가 나옴
            # 0으로 두면 매번 거의 똑같은 답만 나와서 다수결의 의미가 없어짐
            temperature=0.7,
            messages=[{"role": "user", "content": COT_PROMPT}],
        )

        text = response.choices[0].message.content.strip()
        raw_responses.append(text)

        final = extract_final_answer(text)
        answers.append(final)

        print(f"[시도 {i + 1}/{n_samples}] 추출된 답: {final}")

    # Counter로 어떤 답이 몇 번 나왔는지 집계
    vote_counts = Counter(answers)
    most_common_answer, count = vote_counts.most_common(1)[0]

    return {
        "all_answers": answers,
        "vote_counts": vote_counts,
        "final_answer": most_common_answer,
        "confidence": f"{count}/{n_samples}",
        "raw_responses": raw_responses,
    }


if __name__ == "__main__":
    print("=== 문제 ===")
    print(PROBLEM)
    print("(정답 계산: (23 - 12 + 15) / 2 = 26 / 2 = 13)\n")

    print(f"=== Self-consistency 실행 (총 {N_SAMPLES}번 시도) ===\n")
    result = self_consistency()

    print("\n" + "=" * 50)
    print("전체 답변 분포:", dict(result["vote_counts"]))
    print(f"다수결 최종 답: {result['final_answer']}  (신뢰도: {result['confidence']})")
    print("=" * 50)
