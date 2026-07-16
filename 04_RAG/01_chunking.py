"""
Day 4 (1/2): Chunking - 긴 문서를 작은 단위로 쪼개기

이 코드에서 배우는 것:
- 왜 문서를 통째로 임베딩하지 않고 작은 조각(chunk)으로 쪼개는지
- 가장 기본적인 chunking 방법: 고정 길이 + overlap(겹침)
- chunk 크기와 overlap이 검색 품질에 미치는 영향

핵심 아이디어:
  긴 문서를 그대로 임베딩하면 문서 전체의 "평균적인 의미"만 벡터에 담기게 되어,
  특정 부분(예: 한 문단)에 대한 구체적인 질문에는 검색이 잘 안 맞을 수 있음.

  그래서 문서를 작은 chunk로 쪼갠 뒤, chunk 각각을 따로 임베딩해서 저장함.
  질문이 들어오면 "그 질문과 가장 가까운 chunk"를 찾아낼 수 있게 됨.

  overlap(겹침)을 주는 이유:
  chunk 경계에서 문장이 뚝 끊기면 문맥이 손실될 수 있음.
  예: "...RAG는 정확도를 | 높여준다..." 처럼 중요한 부분이 잘리는 걸 막기 위해
  chunk끼리 살짝 겹치게 잘라줌.
"""

import os

# 이 파일 기준으로 data/sample_document.txt 경로를 찾음
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENT_PATH = os.path.join(BASE_DIR, "sample_document.txt")


# ------------------------------------------------------------------
# 고정 길이 chunking (가장 단순하고 기본적인 방법)
#
# chunk_size: 한 chunk에 들어갈 최대 글자 수
# overlap:    다음 chunk와 겹치게 할 글자 수 (문맥 손실 방지)
# ------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    # 문서 앞뒤 불필요한 공백 제거, 줄바꿈을 공백으로 통일해서 다루기 쉽게 함
    text = text.strip().replace("\n", " ")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:  # 빈 문자열은 저장하지 않음
            chunks.append(chunk)

        # 다음 시작 위치를 chunk_size만큼이 아니라 (chunk_size - overlap)만큼만 이동
        # -> 이렇게 해야 chunk끼리 overlap만큼 겹치게 됨
        start += chunk_size - overlap

    return chunks


# ------------------------------------------------------------------
# 문장 단위 chunking (조금 더 자연스러운 방법)
# 글자 수로 뚝 자르지 않고, 마침표(.) 기준으로 문장을 모아서
# chunk_size를 넘기지 않는 선에서 묶어줌 -> 문장이 중간에 잘리는 걸 방지
# ------------------------------------------------------------------
def chunk_by_sentence(text: str, chunk_size: int = 200) -> list[str]:
    text = text.strip().replace("\n", " ")

    # "다." 뒤에서 문장을 나눔 (한국어 문서 기준 간단한 방식, 완벽하진 않음)
    sentences = [s.strip() + "다." for s in text.split("다.") if s.strip()]

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # 현재 chunk에 문장을 추가했을 때 chunk_size를 넘는지 체크
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            # 넘으면 지금까지 모은 chunk를 저장하고 새로 시작
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


if __name__ == "__main__":
    with open(DOCUMENT_PATH, "r", encoding="utf-8") as f:
        document = f.read()

    print(f"원본 문서 길이: {len(document)}자\n")

    print("=" * 60)
    print("[방법 1-1] 고정 길이 chunking (chunk_size=200, overlap=50)")
    print("=" * 60)
    fixed_chunks = chunk_text(document, chunk_size=200, overlap=50)
    for i, chunk in enumerate(fixed_chunks):
        print(f"\n--- chunk {i} (길이: {len(chunk)}자) ---")
        print(chunk)

    print(f"\n총 chunk 개수: {len(fixed_chunks)}\n")

    print("[방법 1-2] 고정 길이 chunking (chunk_size=200, overlap=0)")
    print("=" * 60)
    fixed_chunks = chunk_text(document, chunk_size=200, overlap=0)
    for i, chunk in enumerate(fixed_chunks):
        print(f"\n--- chunk {i} (길이: {len(chunk)}자) ---")
        print(chunk)

    print(f"\n총 chunk 개수: {len(fixed_chunks)}\n")

    print("=" * 60)
    print("[방법 2] 문장 단위 chunking (chunk_size=200)")
    print("=" * 60)
    sentence_chunks = chunk_by_sentence(document, chunk_size=200)
    for i, chunk in enumerate(sentence_chunks):
        print(f"\n--- chunk {i} (길이: {len(chunk)}자) ---")
        print(chunk)

    print(f"\n총 chunk 개수: {len(sentence_chunks)}")

    print("\n" + "=" * 60)
    print("비교해서 확인해볼 것:")
    print("- 고정 길이 방식은 문장이 중간에 잘리는 경우가 있는가?")
    print("- 문장 단위 방식은 chunk마다 길이가 들쭉날쭉한가?")
    print("- overlap을 0으로 바꾸면 chunk 경계에서 문맥이 어떻게 손실되는가?")
    print("=" * 60)
