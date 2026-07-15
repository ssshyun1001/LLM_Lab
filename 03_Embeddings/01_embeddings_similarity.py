"""
Day 3: Embeddings - 텍스트 임베딩 생성 + 코사인 유사도로 문장 유사도 비교

이 코드에서 배우는 것:
- 임베딩(embedding)이란 무엇인가: 텍스트를 "의미를 담은 숫자 벡터"로 바꾸는 것
- OpenAI 임베딩 API 호출 방법
- 코사인 유사도(cosine similarity)로 두 벡터가 "얼마나 비슷한 의미인지" 계산하는 법

핵심 아이디어:
  chat completion은 "텍스트 -> 텍스트"였다면,
  embedding은 "텍스트 -> 숫자 벡터(리스트)"임.

  예: "강아지" -> [0.02, -0.15, 0.88, ...]  (실제로는 1536차원짜리 숫자 리스트)

  의미가 비슷한 문장일수록 이 벡터들이 "가까운 방향"을 가리키게 됨.
  이 "가까움"을 수치로 재는 방법이 코사인 유사도.
"""

import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 임베딩 전용 모델. chat completion 모델(gpt-4o-mini)과는 별도의 모델임.
# text-embedding-3-small: 저렴하고 빠름, 이번 실습엔 충분한 성능
EMBED_MODEL = "text-embedding-3-small"


# ------------------------------------------------------------------
# 텍스트 하나를 벡터(숫자 리스트)로 변환하는 함수
# ------------------------------------------------------------------
def get_embedding(text: str) -> list[float]: # 입력으로 text라는 문자열(str)을 받고, 결과로 float들로 이루어진 리스트(list[float])를 반환
    # embeddings.create()는 chat.completions.create()와 다른 엔드포인트
    # input에 텍스트를 넣으면, 그 텍스트의 "의미"를 담은 숫자 벡터를 돌려줌
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
    )
    # response.data는 리스트 (여러 텍스트를 한번에 넣으면 여러 개 반환됨)
    # 여기선 텍스트 1개만 넣었으니 data[0]에서 embedding 값을 꺼냄
    return response.data[0].embedding


# ------------------------------------------------------------------
# 코사인 유사도 계산
# 두 벡터가 "같은 방향을 가리키는 정도"를 -1 ~ 1 사이 값으로 나타냄
#   1에 가까울수록 -> 의미가 매우 비슷함
#   0에 가까울수록 -> 관련 없음
#  -1에 가까울수록 -> 반대되는 의미 (실제로는 거의 안 나옴)
#
# 공식: cos(theta) = (A · B) / (|A| * |B|)
#   A · B  : 두 벡터의 내적(dot product)
#   |A|,|B|: 각 벡터의 크기(norm)
# ------------------------------------------------------------------
def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)

    dot_product = np.dot(a, b)          # 내적: 두 벡터가 얼마나 같은 방향인지
    norm_a = np.linalg.norm(a)          # 벡터 A의 크기
    norm_b = np.linalg.norm(b)          # 벡터 B의 크기

    return dot_product / (norm_a * norm_b)


# ------------------------------------------------------------------
# 실습 1: 문장 쌍들의 유사도 직접 비교
# 의도적으로 "의미가 비슷한 쌍"과 "의미가 다른 쌍"을 섞어서 넣음
# ------------------------------------------------------------------
def compare_sentence_pairs():
    pairs = [
        ("강아지가 공원에서 뛰어논다", "개가 놀이터에서 달리고 있다"),      # 의미 비슷 (다른 단어)
        ("강아지가 공원에서 뛰어논다", "주식 시장이 오늘 크게 하락했다"),    # 의미 완전 다름
        ("파이썬은 프로그래밍 언어다", "Python is a programming language"),  # 같은 의미, 다른 언어(한/영)
        ("오늘 날씨가 좋다", "오늘 날씨가 좋다"),                          # 완전히 같은 문장
    ]

    print("=== 문장 쌍 유사도 비교 ===\n")

    for sent_a, sent_b in pairs:
        # 두 문장을 각각 임베딩으로 변환
        emb_a = get_embedding(sent_a)
        emb_b = get_embedding(sent_b)

        # 코사인 유사도 계산
        score = cosine_similarity(emb_a, emb_b)

        print(f"문장 A: {sent_a}")
        print(f"문장 B: {sent_b}")
        print(f"유사도: {score:.4f}\n")


# ------------------------------------------------------------------
# 실습 2: "검색" 흉내내보기
# 하나의 질문(query)과 여러 후보 문서(documents) 중에서
# 가장 의미가 비슷한 문서를 찾아내는 것 -> 이게 바로 RAG의 검색 원리
# ------------------------------------------------------------------
def mini_search_demo():
    query = "강아지 산책시키는 방법 알려줘"

    documents = [
        "고양이는 하루에 몇 번 밥을 줘야 하나요?",
        "강아지 하루 산책 횟수와 적절한 시간에 대한 가이드",
        "파이썬으로 웹 크롤러 만드는 법",
        "반려견과 함께 걷기 좋은 운동 루틴 추천",
        "오늘의 주식 시장 마감 시황",
    ]

    print("=== 미니 검색 데모 (RAG의 핵심 원리) ===\n")
    print(f"질문(query): {query}\n")

    # 1. 질문을 임베딩으로 변환
    query_embedding = get_embedding(query)

    # 2. 모든 문서도 각각 임베딩으로 변환
    #    (실제 RAG에서는 이 단계를 미리 해두고 벡터DB에 저장해둠 -> Day 4에서 다룸)
    results = []
    for doc in documents:
        doc_embedding = get_embedding(doc)
        score = cosine_similarity(query_embedding, doc_embedding)
        results.append((doc, score))

    # 3. 유사도 점수가 높은 순으로 정렬
    results.sort(key=lambda x: x[1], reverse=True)

    print("유사도 순위:")
    for rank, (doc, score) in enumerate(results, start=1):
        print(f"{rank}위 (유사도 {score:.4f}): {doc}")

    print(f"\n-> 가장 관련 있는 문서: \"{results[0][0]}\"")
    print("이렇게 '질문과 가장 비슷한 문서를 찾는 것'이 RAG의 검색(Retrieval) 단계예요.")


if __name__ == "__main__":
    compare_sentence_pairs()
    print("\n" + "=" * 60 + "\n")
    mini_search_demo()
