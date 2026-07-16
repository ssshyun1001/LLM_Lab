"""
Day 5: RAG 파이프라인 완성 - 검색된 chunk + LLM으로 답변 생성

이 코드에서 배우는 것:
- Day 4에서 만든 "검색(Retrieval)" 기능에 "생성(Generation)"을 연결해서
  진짜 RAG(Retrieval-Augmented Generation) 파이프라인을 완성함
- RAG를 썼을 때와 안 썼을 때 답변이 어떻게 달라지는지 직접 비교

핵심 아이디어:
  지금까지 배운 걸 순서대로 이으면 RAG가 됨:
    1) 사용자 질문이 들어옴
    2) (Day 3-4) 질문을 임베딩 -> 벡터DB에서 관련 chunk 검색
    3) (NEW) 검색된 chunk들을 "참고 자료"로 삼아 system/user 프롬프트에 끼워 넣음
    4) (Day 1-2) 그 프롬프트를 chat completion에 보내서 최종 답변 생성

  즉 RAG는 새로운 기술이 아니라, 지금까지 배운 임베딩 검색 + chat completion을
  "검색 결과를 프롬프트에 넣어준다"는 방식으로 이어붙인 것뿐임.
"""

import os
import sys
from importlib import import_module
from dotenv import load_dotenv
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
vector_store = import_module("02_vector_store")  # build_vector_store(), search(), get_embedding() 재사용

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHAT_MODEL = "gpt-4o-mini"


# ------------------------------------------------------------------
# 검색된 chunk들을 하나의 "참고 자료" 텍스트로 합치는 함수
# LLM에게 "이 내용을 참고해서 답해줘"라고 전달하기 위한 형태로 가공함
# ------------------------------------------------------------------
def build_context(retrieved_chunks: list[tuple[str, float]]) -> str:
    # retrieved_chunks는 [(chunk_text, distance), ...] 형태
    # 번호를 붙여서 어떤 chunk가 몇 번째로 관련 있는지 구분되게 함
    context_parts = []
    for i, (chunk, _distance) in enumerate(retrieved_chunks, start=1):
        context_parts.append(f"[참고자료 {i}]\n{chunk}")

    return "\n\n".join(context_parts)


# ------------------------------------------------------------------
# RAG 방식으로 답변 생성: 검색 -> 프롬프트 구성 -> LLM 호출
# ------------------------------------------------------------------
def answer_with_rag(collection, question: str, top_k: int = 3) -> dict:
    # 1. Day 4에서 만든 search() 함수로 질문과 관련된 chunk 검색
    retrieved = vector_store.search(collection, question, top_k=top_k)

    # 2. 검색된 chunk들을 하나의 참고 자료 텍스트로 합침
    context = build_context(retrieved)

    # 3. system 메시지로 "주어진 참고자료만 근거로 답하라"는 규칙을 명시
    #    이 지시가 없으면 모델이 참고자료를 무시하고 자기 지식으로 답할 수 있음
    system_prompt = (
        "너는 주어진 참고자료를 바탕으로 질문에 답하는 어시스턴트야. "
        "참고자료에 없는 내용은 답하지 말고, 참고자료에서 근거를 찾을 수 없으면 "
        "'주어진 자료에서 찾을 수 없습니다'라고 답해. 답변은 2문장 이내로 간결하게."
    )

    user_prompt = f"참고자료:\n{context}\n\n질문: {question}"

    # 4. 지금까지 배운 chat completion 호출 (Day 1-2와 동일한 구조)
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return {
        "question": question,
        "retrieved_chunks": retrieved,
        "context": context,
        "answer": response.choices[0].message.content.strip(),
    }


# ------------------------------------------------------------------
# 비교용: RAG 없이 그냥 LLM에게 바로 물어봤을 때
# (참고자료 없이 모델이 원래 알고 있던 지식만으로 답함)
# ------------------------------------------------------------------
def answer_without_rag(question: str) -> str:
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    print("=== 벡터DB 준비 ===\n")
    # Day 4에서 만든 build_vector_store()를 그대로 재사용
    # (chroma_db 폴더에 이미 저장되어 있으면 재임베딩하지 않음)
    collection = vector_store.build_vector_store()

    questions = [
        "RAG 파이프라인은 몇 단계로 구성되어 있어?",
        "에이전트가 도구를 사용하는 방식의 기반이 되는 기능은 뭐야?",
    ]

    for question in questions:
        print("=" * 70)
        print(f"질문: {question}")
        print("=" * 70)

        # RAG 방식 답변
        result = answer_with_rag(collection, question, top_k=2)

        print("\n[검색된 참고자료]")
        print(result["context"])

        print("\n[RAG 답변]")
        print(result["answer"])

        # 비교용: RAG 없이 바로 물어본 답변
        plain_answer = answer_without_rag(question)
        print("\n[RAG 없이 바로 물어본 답변 (비교용)]")
        print(plain_answer)

        print()

    print("생각해볼 것:")
    print("- RAG 답변이 실제로 검색된 참고자료 내용을 근거로 나왔는가?")
    print("- RAG 없이 물어본 답변과 비교했을 때 어떤 차이가 있는가?")
    print("  (일반적으로는 비슷하게 답할 수도 있지만, 문서에만 있는 특정 정보를")
    print("   물어보면 RAG 없는 쪽은 '모른다'거나 틀린 답을 할 가능성이 높아짐)")
    print("- system 프롬프트의 '참고자료에 없으면 모른다고 답해' 지시를 지웠다면")
    print("  결과가 어떻게 달라졌을지 예상해보기")
