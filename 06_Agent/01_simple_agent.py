"""
Day 7 (마지막): function calling 기반 간단 에이전트

이 코드에서 배우는 것:
- function calling(tool use)이 실제로 어떻게 동작하는지: 모델이 "도구를 써야겠다"고
  판단하면, 우리 코드에게 "이 함수를 이 인자로 실행해줘"라고 요청하는 방식
- 에이전트의 핵심 루프: 질문 -> (필요하면) 도구 호출 -> 결과를 다시 모델에게 전달
  -> (필요하면 또) 도구 호출 -> ... -> 최종 답변
- 04_RAG에서 만든 "문서 검색" 기능을 에이전트가 쓸 수 있는 "도구" 중 하나로 재사용

핵심 아이디어:
  지금까지는 우리가 직접 "이 질문엔 RAG를 쓰자" "이 질문엔 그냥 답하자"를 코드로 정해줬음.
  에이전트는 그 판단 자체를 모델에게 맡김. 우리는 "계산기"와 "문서 검색" 두 가지 도구를
  모델에게 소개(스키마로 설명)만 해주고, 실제로 어떤 도구를 언제 쓸지는 모델이 질문을 보고
  스스로 결정함.

  이게 가능한 이유: 모델은 텍스트를 생성하는 대신, "나 이 함수를 이 인자로 불러줘"라는
  구조화된 요청(tool_calls)을 응답으로 대신 내놓을 수 있음. 우리는 그 요청을 받아서
  실제 파이썬 함수를 실행하고, 결과를 다시 모델에게 돌려주는 역할만 하면 됨.
"""

import os
import sys
import json
from importlib import import_module
from dotenv import load_dotenv
from openai import OpenAI

# 04_RAG/02_vector_store.py의 build_vector_store(), search()를 재사용
# -> "문서 검색"을 에이전트의 도구 중 하나로 그대로 가져다 씀
RAG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "04_RAG")
sys.path.append(os.path.abspath(RAG_DIR))
vector_store = import_module("02_vector_store")

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"


# ------------------------------------------------------------------
# 도구 1: 계산기
# 모델은 산수 계산을 종종 틀림 (특히 큰 수 곱셈, 복잡한 계산).
# "직접 계산하지 말고 이 함수를 호출해서 계산하라"고 시키면 정확도가 올라감.
#
# 주의: eval()은 실제 서비스에서는 위험할 수 있음(임의 코드 실행 가능성).
# 여기서는 학습 목적으로 최소한의 안전장치(허용 문자 검사)만 넣었음.
# 실무에서는 ast.literal_eval 기반 파서나 전용 수식 계산 라이브러리를 쓰는 게 안전함.
# ------------------------------------------------------------------
def calculator(expression: str) -> str:
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_chars:
        return "오류: 숫자와 +,-,*,/,(,) 외의 문자는 계산할 수 없습니다."

    try:
        result = eval(expression)  # 허용 문자 검사를 통과한 순수 산술식만 실행됨
        return str(result)
    except Exception as e:
        return f"계산 오류: {e}"


# ------------------------------------------------------------------
# 도구 2: 문서 검색 (04_RAG의 search 기능 재사용)
# 에이전트가 "이 질문은 문서를 찾아봐야 답할 수 있겠다"고 판단하면 이 도구를 호출함.
# ------------------------------------------------------------------
_collection = None  # 벡터DB는 한 번만 로드해서 재사용 (매번 다시 만들지 않도록 캐싱)


def search_knowledge_base(query: str) -> str:
    global _collection
    if _collection is None:
        _collection = vector_store.build_vector_store()

    results = vector_store.search(_collection, query, top_k=2)
    # 검색된 chunk들을 하나의 문자열로 합쳐서 모델에게 돌려줌
    combined = "\n\n".join(chunk for chunk, _distance in results)
    return combined


# ------------------------------------------------------------------
# 실제로 실행 가능한 함수들을 이름으로 찾을 수 있게 딕셔너리로 매핑
# 모델이 "calculator라는 함수를 불러줘"라고 요청하면, 이 딕셔너리에서 찾아 실행함
# ------------------------------------------------------------------
AVAILABLE_TOOLS = {
    "calculator": calculator,
    "search_knowledge_base": search_knowledge_base,
}

# ------------------------------------------------------------------
# 도구 스키마(설명서) 정의
# 모델은 이 스키마를 읽고 "지금 질문엔 어떤 도구를, 어떤 인자로 써야 하는지" 판단함.
# name/description/parameters를 최대한 명확하게 적어줘야 모델이 정확히 판단함.
# ------------------------------------------------------------------
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "사칙연산(+,-,*,/) 수식을 정확하게 계산한다. 숫자 계산이 필요할 때 사용.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "계산할 수식. 예: '23 - 12 + 15'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "AI, 머신러닝, LLM, RAG, 에이전트 등에 대해 저장된 문서에서 "
                "관련 내용을 검색한다. 일반 상식이 아닌, 특정 개념 설명이 필요할 때 사용."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색할 질문이나 키워드",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


# ------------------------------------------------------------------
# 에이전트 핵심 루프
#
# 1. 사용자 질문을 messages에 넣고 모델 호출 (tools=TOOLS_SCHEMA와 함께)
# 2. 모델 응답에 tool_calls가 있으면:
#      - 요청된 함수를 실제로 실행
#      - 실행 결과를 role="tool"로 messages에 추가
#      - 다시 모델을 호출 (모델이 도구 결과를 보고 다음 행동을 결정하게 함)
# 3. tool_calls가 없으면(=모델이 최종 답변을 냈으면) 반복을 멈추고 답변 반환
#
# 이 "호출 -> 도구 실행 -> 결과 반영 -> 재호출"의 반복이 에이전트의 본질임.
# ------------------------------------------------------------------
def run_agent(user_question: str, max_iterations: int = 5) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "너는 계산기와 문서 검색 도구를 쓸 수 있는 어시스턴트야. "
                "필요할 때만 도구를 사용하고, 도구 없이 답할 수 있는 질문은 바로 답해."
            ),
        },
        {"role": "user", "content": user_question},
    ]

    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",  # 모델이 도구를 쓸지 말지 스스로 판단하게 함
        )

        message = response.choices[0].message

        # 모델이 도구 호출을 요청하지 않았다면 -> 최종 답변으로 판단하고 종료
        if not message.tool_calls:
            return message.content

        # 모델의 응답(도구 호출 요청 포함)을 대화 기록에 추가
        # 이걸 넣어줘야 다음 호출에서 모델이 "내가 아까 무슨 도구를 불렀는지" 기억함
        messages.append(message)

        # 모델이 여러 도구를 동시에 요청했을 수도 있으므로 하나씩 순회하며 실행
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)  # 문자열 -> dict 변환

            print(f"  [도구 호출] {function_name}({function_args})")

            # AVAILABLE_TOOLS 딕셔너리에서 실제 파이썬 함수를 찾아 실행
            function_to_call = AVAILABLE_TOOLS[function_name]
            function_result = function_to_call(**function_args)

            print(f"  [도구 결과] {function_result[:100]}{'...' if len(function_result) > 100 else ''}")

            # 도구 실행 결과를 role="tool"로 messages에 추가
            # tool_call_id로 "어떤 요청에 대한 결과인지" 매칭시킴
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": function_result,
                }
            )

        # 도구 결과까지 반영된 messages로 다시 루프를 돌아 모델을 재호출함

    return "최대 반복 횟수를 초과했습니다. (도구 호출이 계속 반복되는 문제가 있을 수 있음)"


if __name__ == "__main__":
    # 세 가지 성격이 다른 질문으로 테스트:
    #   1) 계산기만 필요한 질문
    #   2) 문서 검색만 필요한 질문
    #   3) 아무 도구도 필요 없는 일반 질문 (모델이 도구를 안 쓰는 것도 확인)
    questions = [
        "각 상자에 사과가 128개씩 들어있는 상자가 7개 있어. 그중 250개를 팔았어. 몇 개 남았어?",
        "RAG 파이프라인은 어떤 단계들로 구성돼있어? 저장된 문서를 찾아서 알려줘.",
        "너는 어떤 언어 모델이야? 도구 없이 그냥 답해줘.",
    ]

    for question in questions:
        print("=" * 70)
        print(f"질문: {question}")
        print("=" * 70)

        answer = run_agent(question)

        print(f"\n[최종 답변]\n{answer}\n")

    print("생각해볼 것:")
    print("- 1번 질문에서 모델이 계산기 도구를 실제로 호출했나, 아니면 직접 암산했나?")
    print("- 2번 질문에서 search_knowledge_base가 호출되고, 그 결과가 답변에 반영됐나?")
    print("- 3번 질문에서는 도구를 전혀 안 쓰고 바로 답했나? (불필요한 도구 호출이 없어야 좋음)")
    print("- tool_choice='auto' 대신 'required'로 바꾸면 어떻게 될지 예상해보기")
