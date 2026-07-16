"""
Day 4 (2/2): Vector Store - chunk를 임베딩해서 벡터DB(Chroma)에 저장하고 검색하기

이 코드에서 배우는 것:
- 벡터DB가 왜 필요한지 (Day 3에서 직접 계산했던 코사인 유사도를 대신 처리해줌)
- Chroma를 이용해 chunk들을 저장하고, 질문과 유사한 chunk를 검색하는 법
>> chroma : 임베딩(벡터)을 저장하고 비슷한 벡터를 빠르게 찾아주는 벡터 데이터 베이스


핵심 아이디어:
  Day 3에서는 문서 몇 개를 놓고 우리가 직접 코사인 유사도를 계산해서 순위를 매겼음.
  문서가 수천~수만 개라면 이 방식은 너무 느림.

  벡터DB(Chroma, Pinecone, Weaviate 등)는 대량의 벡터를 효율적으로 저장하고,
  "이 벡터랑 가장 비슷한 벡터 top-k개"를 빠르게 찾아주는 전용 데이터베이스임.
  우리는 코사인 유사도를 직접 계산할 필요 없이, DB에 "이 질문이랑 비슷한 거 찾아줘"라고
  요청만 하면 됨.
"""

import os
import chromadb
from dotenv import load_dotenv
from openai import OpenAI

# 같은 폴더의 01_chunking.py에서 문장 단위 chunking 함수를 재사용
from importlib import import_module
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
chunking_module = import_module("01_chunking")

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED_MODEL = "text-embedding-3-small"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENT_PATH = os.path.join(BASE_DIR, "sample_document.txt")

# Chroma가 데이터를 저장할 로컬 폴더 (재실행해도 데이터가 남아있게 됨)
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")


# ------------------------------------------------------------------
# 텍스트를 임베딩 벡터로 변환 (Day 3에서 만든 것과 동일한 역할)
# ------------------------------------------------------------------
def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding


# ------------------------------------------------------------------
# 1단계: 문서를 chunk로 쪼개고, 각 chunk를 임베딩해서 Chroma에 저장
'''
| 파일                | 역할                                         |
| ----------------- | ------------------------------------------ |
| `chroma.sqlite3`  | Collection, 문서, ID 등 메타데이터를 저장하는 SQLite DB |
| `data_level0.bin` | 실제 임베딩 벡터 저장                               |
| `header.bin`      | 벡터 인덱스의 설정 및 헤더 정보                         |
| `length.bin`      | 각 벡터(노드)의 연결 개수 정보                         |
| `link_lists.bin`  | 벡터 간 연결 그래프(HNSW) 정보. 빠른 유사도 검색의 핵심        |

'''
# ------------------------------------------------------------------
def build_vector_store():
    # PersistentClient: 메모리가 아니라 디스크(CHROMA_DIR)에 저장 -> 재실행해도 유지됨
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    # collection = 벡터DB 안의 "테이블" 같은 개념. 이름으로 구분함.
    # 이미 있으면 가져오고, 없으면 새로 만듦
    collection = chroma_client.get_or_create_collection(name="llm_lab_docs")

    # 이미 데이터가 들어있으면 중복 저장하지 않도록 체크
    if collection.count() > 0:
        print(f"이미 {collection.count()}개의 chunk가 저장되어 있습니다. 다시 만들려면 chroma_db 폴더를 삭제하세요.\n")
        return collection

    with open(DOCUMENT_PATH, "r", encoding="utf-8") as f:
        document = f.read()

    # Day 4-1에서 만든 문장 단위 chunking 함수 재사용
    chunks = chunking_module.chunk_by_sentence(document, chunk_size=200)
    print(f"문서를 {len(chunks)}개의 chunk로 분할했습니다.\n")

    # 각 chunk를 하나씩 임베딩으로 변환
    embeddings = []
    ids = []
    for i, chunk in enumerate(chunks):
        print(f"chunk {i} 임베딩 생성 중...")
        embeddings.append(get_embedding(chunk))
        ids.append(f"chunk_{i}")  # 각 chunk를 구분할 고유 id

    # Chroma에 한 번에 저장
    # documents: 원문 텍스트 (나중에 검색 결과로 그대로 돌려받기 위함)
    # embeddings: 우리가 미리 계산한 벡터
    # ids: 각 항목의 고유 식별자
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
    )

    print(f"\n총 {len(chunks)}개 chunk를 벡터DB에 저장 완료.\n")
    return collection


# ------------------------------------------------------------------
# 2단계: 질문을 임베딩해서, 저장된 chunk 중 가장 유사한 것들을 검색
# ------------------------------------------------------------------
def search(collection, query: str, top_k: int = 3):
    query_embedding = get_embedding(query)

    # collection.query()가 내부적으로  유사 거리를 계산해서
    # 가장 가까운 top_k개를 알아서 찾아줌 -> Day 3에서 직접 짰던 로직을 대신 해주는 것
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    # results["documents"][0]에 검색된 chunk 텍스트들이 순위대로 들어있음
    retrieved_chunks = results["documents"][0]
    distances = results["distances"][0]  # 낮을수록 더 유사함 (거리이므로)

    return list(zip(retrieved_chunks, distances))


if __name__ == "__main__":
    print("=== 벡터DB 구축 ===\n")
    collection = build_vector_store()

    test_queries = [
        "RAG가 뭐야?",
        "프롬프트를 잘 쓰는 방법은?",
        "에이전트는 어떻게 도구를 사용해?",
    ]

    for query in test_queries:
        print("=" * 60)
        print(f"질문: {query}")
        print("=" * 60)

        top_results = search(collection, query, top_k=2)

        for rank, (chunk, distance) in enumerate(top_results, start=1):
            print(f"\n{rank}위 (거리: {distance:.4f})")
            print(chunk)

        print()

    print("생각해볼 것:")
    print("- 검색된 chunk들이 질문과 실제로 관련 있는 내용이었나?")
    print("- Day 3에서 직접 계산한 코사인 유사도와 결과가 비슷한 경향을 보이나?")
    print("- chroma_db 폴더가 생긴 걸 확인했나요? (재실행해도 다시 임베딩 안 함)")
