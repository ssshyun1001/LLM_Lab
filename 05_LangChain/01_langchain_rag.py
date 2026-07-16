"""
Day 6: LangChain으로 RAG 파이프라인 다시 만들기

이 코드에서 배우는 것:
- 04_RAG에서 직접 짰던 4단계(chunking -> 임베딩 -> 벡터DB 저장 -> 검색+생성)를
  LangChain이라는 프레임워크로 다시 구현해보고 직접 비교
- LangChain의 핵심 부품들: TextSplitter, Embeddings, VectorStore, Retriever, Chain
- "LCEL(LangChain Expression Language)" 문법으로 여러 단계를 파이프처럼 연결하는 법 (| 연산자)

핵심 아이디어:
  04_RAG에서 우리가 손으로 짠 것:
    - chunk_by_sentence()        -> 직접 만든 chunking 함수
    - get_embedding()            -> OpenAI API 직접 호출
    - chromadb.PersistentClient  -> Chroma를 직접 다룸
    - collection.query()         -> 검색 로직 직접 호출
    - client.chat.completions.create() -> 프롬프트 직접 구성해서 호출

  LangChain은 이 각각을 "표준화된 부품"으로 제공해서, 같은 걸 훨씬 적은 코드로
  구현하게 해줌. 대신 내부에서 무슨 일이 일어나는지는 잘 안 보이게 됨(추상화).
  그래서 04_RAG를 먼저 손으로 만들어본 게 의미가 있음 - "내부에서 뭐가 일어나는지"를
  이미 알고 있는 상태로 프레임워크를 쓰는 것과, 모르고 쓰는 것은 완전히 다름.
"""

import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 04_RAG 폴더의 샘플 문서를 그대로 재사용 (같은 문서로 비교해야 공정한 비교가 됨)
DOCUMENT_PATH = os.path.join(BASE_DIR, "..", "04_RAG", "sample_document.txt")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db_langchain")  # 04_RAG와 별도 폴더 사용


# ------------------------------------------------------------------
# 1단계: 문서 로드 + chunking
# 04_RAG에서는 chunk_by_sentence()를 직접 만들었지만,
# LangChain은 RecursiveCharacterTextSplitter를 기본 제공함.
#
# RecursiveCharacterTextSplitter의 동작 방식:
# "\n\n" -> "\n" -> " " -> "" 순서로 구분자를 우선순위대로 시도하면서,
# chunk_size를 넘지 않는 선에서 최대한 "자연스러운 경계"(문단/줄/단어)를 살려 자름.
# 우리가 손으로 짠 chunk_by_sentence()보다 범용적이지만, 원리는 비슷함.
# ------------------------------------------------------------------
def load_and_split():
    with open(DOCUMENT_PATH, "r", encoding="utf-8") as f:
        document = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,      # 04_RAG와 동일한 값으로 맞춰서 결과를 비교하기 쉽게 함
        chunk_overlap=50,    # 04_RAG의 overlap과 동일
    )

    # split_text()는 문자열을 받아 chunk 문자열 리스트를 반환
    chunks = splitter.split_text(document)
    return chunks


# ------------------------------------------------------------------
# 2단계 + 3단계: 임베딩 생성 + 벡터DB 저장
# 04_RAG에서는 get_embedding()으로 하나씩 벡터를 만들고,
# collection.add()로 Chroma에 직접 넣었음.
#
# LangChain에서는 Chroma.from_texts() 한 줄이 그 과정을 전부 대신함:
#   내부적으로 각 chunk마다 embedding.embed_query()를 호출해서 벡터를 만들고,
#   그 벡터들을 Chroma DB에 저장하는 걸 자동으로 처리해줌.
# ------------------------------------------------------------------
def build_vectorstore(chunks: list[str]):
    # OpenAIEmbeddings: 04_RAG의 get_embedding() 함수와 동일한 역할을 하는 LangChain 부품
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Chroma.from_texts: chunk 리스트를 받아서
    #   1) 각 chunk를 embeddings로 벡터화하고
    #   2) 벡터DB에 저장하는 것까지 한 번에 처리
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,  # 디스크에 저장 (04_RAG의 PersistentClient와 같은 역할)
    )

    return vectorstore


# ------------------------------------------------------------------
# 4단계: 검색 + 생성을 하나의 체인(chain)으로 연결
#
# 04_RAG의 answer_with_rag()에서는:
#   검색 -> context 문자열 조립 -> 프롬프트 조립 -> client.chat.completions.create()
# 를 순서대로 우리가 직접 호출했음.
#
# LangChain에서는 이 순서를 "|" 연산자로 이어붙여서 하나의 파이프라인(chain)을 만듦.
# 이 문법을 LCEL(LangChain Expression Language)이라고 부름.
# ------------------------------------------------------------------
def build_rag_chain(vectorstore):
    # retriever: "질문을 넣으면 관련 문서를 찾아주는" 객체
    # 04_RAG의 search() 함수와 동일한 역할. k=2는 top_k=2와 같은 의미.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 프롬프트 템플릿: 04_RAG에서 만들었던 system_prompt + user_prompt 조합과 동일한 내용
    # {context}와 {question}은 나중에 실제 값으로 채워지는 자리표시자(placeholder)
    prompt = ChatPromptTemplate.from_template(
        """너는 주어진 참고자료를 바탕으로 질문에 답하는 어시스턴트야.
참고자료에 없는 내용은 답하지 말고, 참고자료에서 근거를 찾을 수 없으면
'주어진 자료에서 찾을 수 없습니다'라고 답해. 답변은 3문장 이내로 간결하게.

참고자료:
{context}

질문: {question}"""
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 검색된 문서(Document 객체) 리스트를 하나의 문자열로 합치는 함수
    # 04_RAG의 build_context() 함수와 동일한 역할
    def format_docs(docs):
        return "\n\n".join(f"[참고자료 {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs))

    # LCEL 체인 구성: "|"는 "왼쪽의 출력을 오른쪽의 입력으로 전달한다"는 뜻
    #
    # 흐름:
    #   1) {"context": retriever | format_docs, "question": RunnablePassthrough()}
    #      -> 질문이 들어오면, retriever가 관련 문서를 찾고 format_docs로 문자열로 합침 (context)
    #         동시에 질문 원본은 그대로 통과시킴 (question)
    #   2) | prompt
    #      -> context와 question을 프롬프트 템플릿의 {context}, {question} 자리에 채워 넣음
    #   3) | llm
    #      -> 완성된 프롬프트를 LLM에게 전달해서 응답을 받음
    #   4) | StrOutputParser()
    #      -> LLM 응답 객체에서 순수 텍스트만 뽑아냄 (04_RAG의 .choices[0].message.content와 동일)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


if __name__ == "__main__":
    print("=== 1. 문서 로드 + chunking ===")
    chunks = load_and_split()
    print(f"총 {len(chunks)}개의 chunk로 분할됨\n")

    print("=== 2. 벡터DB 구축 ===")
    vectorstore = build_vectorstore(chunks)
    print(f"벡터DB 구축 완료 (저장 위치: {CHROMA_DIR})\n")

    print("=== 3. RAG 체인 구성 ===")
    rag_chain = build_rag_chain(vectorstore)
    print("체인 구성 완료\n")

    # 04_RAG의 03_rag_pipeline.py와 동일한 질문으로 테스트 -> 직접 결과 비교 가능
    questions = [
        "RAG 파이프라인은 몇 단계로 구성되어 있어?",
        "에이전트가 도구를 사용하는 방식의 기반이 되는 기능은 뭐야?",
    ]

    for question in questions:
        print("=" * 70)
        print(f"질문: {question}")
        print("=" * 70)

        # chain.invoke() 한 줄이 "검색 -> 프롬프트 구성 -> LLM 호출 -> 텍스트 추출"을 전부 수행
        # 04_RAG의 answer_with_rag() 함수 전체(약 15줄)가 이 한 줄로 압축된 것
        answer = rag_chain.invoke(question)

        print(f"\n[LangChain RAG 답변]\n{answer}\n")

    print("=" * 70)
    print("04_RAG (직접 구현) vs 05_LangChain (프레임워크) 비교 포인트")
    print("=" * 70)
    print("- 직접 구현: 각 단계(임베딩 호출, DB 저장, 검색, 프롬프트 조립)가 코드에 다 보임")
    print("             -> 무슨 일이 일어나는지 정확히 통제 가능, 대신 코드가 김")
    print("- LangChain: retriever | format_docs | prompt | llm | parser 로 압축됨")
    print("             -> 코드가 짧고 표준화됨, 대신 내부 동작은 추상화되어 덜 보임")
    print("- 실무에서는 보통 간단한 프로토타입은 LangChain, 세밀한 제어가 필요하면")
    print("  직접 구현하거나 LangChain 내부를 커스터마이징하는 방식을 섞어서 씀")
