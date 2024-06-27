---
title: "한국어 검색 엔진: Langchain과 Faiss를 이용한 RAG 체인 구축"
datePublished: Thu Jun 27 2024 01:33:37 GMT+0000 (Coordinated Universal Time)
cuid: clxwlc3m8000009jv1qdp95zk
slug: langchain-faiss-rag
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1719451872379/5a1d861d-7896-4f4e-bb26-1e9996ed2334.webp
tags: ai, mongodb, machine-learning, elasticsearch, nlu, langchain, llm-retrieval, openai-llms-langchain-promttemplate-promptengineering-python, faiss, data-indexing, retrieval-augmented-generation, korean-language-processing, search-engine-technology

---

### 서론

인공지능 기반의 검색 및 추천 시스템은 다양한 데이터 소스에서 유용한 정보를 신속하게 검색하고 제공하는 데 필수적입니다. 최근 프로젝트에서 Elasticsearch를 사용하여 벡터 임베딩을 색인화하고 검색하는 기존 방식에서 한국어 처리 성능의 한계와 높은 비용 문제를 경험했습니다. 이에 따라, 더 나은 성능과 비용 효율성을 제공하는 새로운 솔루션을 개발하기로 결정하였습니다.

이 글에서는 Elasticsearch 대신 `langchain`과 `faiss`를 이용하여 어떻게 더 효율적인 검색 엔진을 구축했는지, 그리고 한국어 토큰화와 앙상블 방식을 적용하여 어떻게 성능을 개선했는지 상세히 설명하겠습니다.

### 기술 스택과 아키텍처

이 프로젝트에서는 `langchain`, `faiss`, `MongoDB`, 그리고 한국어 토크나이저 `Kiwi`를 주요 기술로 선택했습니다. Elasticsearch의 강력한 KNN 검색 기능 대신, `faiss`를 이용해 비용 효율적이며 유사한 수준의 검색 결과를 제공하는 시스템을 구현했습니다. 이 시스템은 `Kiwi` 토크나이저를 통해 한국어 데이터의 토큰화 성능을 크게 향상시켜, 검색 정확도를 높였습니다.

### 상세 코드

#### 1\. FaissIndexManager

`FaissIndexManager`는 MongoDB에 벡터 인덱스를 적재하고 검색하는 클래스입니다. 이 클래스는 `faiss` 라이브러리를 사용하여 문서의 벡터 임베딩을 생성하고, 이를 인덱싱하여 MongoDB에 저장합니다. 아래는 `FaissIndexManager`에서 인덱스를 생성하고 MongoDB에 저장하는 과정을 단계별로 보여주는 코드입니다.

```python
import pickle
import bson
import concurrent.futures
import logging

import faiss
import numpy as np
from pymongo import MongoClient
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.docstore import InMemoryDocstore

from gpt_recruit_rag.docstore import SimpleIndexToDocstoreID


class FaissIndexManager:
    def __init__(self, mongo_uri, db_name, collection_name, embedding_uri):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.embedding = HuggingFaceEndpointEmbeddings(model=embedding_uri)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def create_faiss_index_in_batches(self, docs, batch_size=10):
        embedding_dim = len(self.embedding.embed_documents([docs[0].page_content])[0])
        faiss_index = faiss.IndexFlatL2(embedding_dim)
        docstore = InMemoryDocstore()
        index_to_docstore_id = SimpleIndexToDocstoreID()
        current_index = 0

        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i + batch_size]
            texts = [d.page_content for d in batch_docs]
            embeddings = self.embedding.embed_documents(texts)
            embeddings_np = np.array(embeddings).astype('float32')

            assert embeddings_np.shape[1] == embedding_dim, f"Embedding dimension mismatch: {embeddings_np.shape[1]} vs {embedding_dim}"
            
            faiss_index.add(embeddings_np)

            for j, doc in enumerate(batch_docs):
                doc_id = f"{i + j}"
                docstore.add({doc_id: doc})
                index_to_docstore_id.add(current_index, doc_id)
                current_index += 1
            
            print(f"Processed batch {i // batch_size + 1}/{len(docs) // batch_size + 1}")

        return faiss_index, docstore, index_to_docstore_id

    def save_faiss_index_to_mongo(self, faiss_index, docstore, index_to_docstore_id, chunk_size=10*1024*1024):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        futures = []

        # Save FAISS index
        index_bytes = faiss.serialize_index(faiss_index).tobytes()
        self._save_chunks(index_bytes, chunk_size, "faiss_index_chunk", executor, futures)
        self.collection.update_one({"_id": "faiss_index_metadata"}, {"$set": {"num_chunks": len(index_bytes) // chunk_size + 1}}, upsert=True)

        # Save docstore
        docstore_bytes = pickle.dumps(docstore._dict)
        self._save_chunks(docstore_bytes, chunk_size, "faiss_docstore_chunk", executor, futures)
        self.collection.update_one({"_id": "faiss_docstore_metadata"}, {"$set": {"num_chunks": len(docstore_bytes) // chunk_size + 1}}, upsert=True)

        # Save index_to_docstore_id mapping
        index_to_id_bytes = pickle.dumps(index_to_docstore_id.index_to_id)
        self._save_chunks(index_to_id_bytes, chunk_size, "faiss_index_to_docstore_id_chunk", executor, futures)
        self.collection.update_one({"_id": "faiss_index_to_docstore_id_metadata"}, {"$set": {"num_chunks": len(index_to_id_bytes) // chunk_size + 1}}, upsert=True)

        # Wait for all tasks to complete and handle exceptions
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Failed to save a chunk: {e}")

        logging.info("All chunks have been processed and stored.")

    def _save_chunks(self, data, chunk_size, prefix, executor, futures):
        num_chunks = len(data) // chunk_size + 1
        for i in range(num_chunks):
            chunk = data[i * chunk_size:(i + 1) * chunk_size]
            futures.append(executor.submit(self._save_chunk, i, chunk, prefix))

    def _save_chunk(self, chunk_id, data, prefix):
        try:
            self.collection.update_one(
                {"_id": f"{prefix}_{chunk_id}"},
                {"$set": {"chunk": bson.Binary(data)}},
                upsert=True
            )
            logging.info(f"Successfully saved chunk {chunk_id} for {prefix}")
        except Exception as e:
            logging.error(f"Failed to save chunk {chunk_id} for {prefix}: {e}")

    def load_faiss_index_from_mongo(self):
        try:
            # FAISS 인덱스 로드
            metadata = self.collection.find_one({"_id": "faiss_index_metadata"})
            if not metadata:
                raise Exception("FAISS index metadata not found")
            
            num_chunks = metadata["num_chunks"]
            index_bytes = b""
            for i in range(num_chunks):
                chunk_data = self.collection.find_one({"_id": f"faiss_index_chunk_{i}"})
                if not chunk_data:
                    raise Exception(f"Missing chunk {i} for FAISS index")
                index_bytes += chunk_data["chunk"]

            faiss_index = faiss.deserialize_index(np.frombuffer(index_bytes, dtype='uint8'))

            # docstore 로드
            metadata = self.collection.find_one({"_id": "faiss_docstore_metadata"})
            if not metadata:
                raise Exception("Docstore metadata not found")
            
            num_chunks = metadata["num_chunks"]
            docstore_bytes = b""
            for i in range(num_chunks):
                chunk_data = self.collection.find_one({"_id": f"faiss_docstore_chunk_{i}"})
                if not chunk_data:
                    raise Exception(f"Missing chunk {i} for docstore")
                docstore_bytes += chunk_data["chunk"]

            docstore_dict = pickle.loads(docstore_bytes)
            docstore = InMemoryDocstore()
            docstore._dict = docstore_dict

            # index_to_docstore_id 로드
            metadata = self.collection.find_one({"_id": "faiss_index_to_docstore_id_metadata"})
            if not metadata:
                raise Exception("Index to docstore ID metadata not found")
            
            num_chunks = metadata["num_chunks"]
            index_to_id_bytes = b""
            for i in range(num_chunks):
                chunk_data = self.collection.find_one({"_id": f"faiss_index_to_docstore_id_chunk_{i}"})
                if not chunk_data:
                    raise Exception(f"Missing chunk {i} for index to docstore ID mapping")
                index_to_id_bytes += chunk_data["chunk"]

            index_to_id_dict = pickle.loads(index_to_id_bytes)
            index_to_docstore_id = SimpleIndexToDocstoreID()
            index_to_docstore_id.index_to_id = index_to_id_dict

            return faiss_index, docstore, index_to_docstore_id

        except Exception as e:
            logging.error(f"Failed to load data from MongoDB: {e}")
            raise e

    def delete_collection(self):
        self.collection.drop()
        logging.info("Collection has been deleted.")

    def close(self):
        self.client.close()
        logging.info("MongoDB connection closed.")
```

#### 2\. KiwiBM25FaissEnsembleRetriever

`KiwiBM25FaissEnsembleRetriever`는 한국어 토큰화를 담당하는 `Kiwi`와 BM25 알고리즘, 그리고 `faiss` 검색 엔진을 조합한 앙상블 리트리버입니다. 이 리트리버는 토큰화된 텍스트를 사용하여 BM25 알고리즘을 통해 초기 검색을 수행하고, `faiss`를 사용하여 세부적인 벡터 기반 검색을 수행합니다. 이 앙상블 접근 방식은 검색의 정확성을 높이면서 다양한 유형의 검색 쿼리에 효과적으로 대응할 수 있습니다.

```python
from typing import List

from kiwipiepy import Kiwi
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from gpt_recruit_rag.faiss import FaissIndexManager

kiwi = Kiwi()


def kiwi_tokenize(text):
    return [token.form for token in kiwi.tokenize(text)]


def build_kiwi_bm25(docs: List[Document]) -> BM25Retriever:
    print("Building Kiwi BM25...")
    kiwi_bm25 = BM25Retriever.from_documents(
        docs,
        preprocess_func=kiwi_tokenize
    )
    return kiwi_bm25


class KiwiBM25FaissEnsembleRetriever:
    def __init__(
        self,
        mongo_uri: str,
        db_name: str,
        collection_name: str,
        embedding_uri: str,
        docs: List[Document] = None,
        kiwi_bm25: BM25Retriever = None,
        weights: List[float] = [0.7, 0.3],
        create_index: bool = False
        ):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.embedding_uri = embedding_uri

        self.fim = FaissIndexManager(
            self.mongo_uri,
            self.db_name,
            self.collection_name,
            self.embedding_uri
        )

        self.docs = docs or []
        self.kiwi_bm25 = kiwi_bm25 or build_kiwi_bm25(self.docs)
        self.weights = weights
        if create_index:
            self._create_save_new_index()

    def build_ensemble_retriever(self):
        try:
            faiss_index, docstore, index_to_docstore_id = self.fim.load_faiss_index_from_mongo()
            print("Loaded Faiss index from MongoDB.")
        except:
            print("Creating new Faiss index...")
            faiss_index, docstore, index_to_docstore_id = self._create_save_new_index()

        faiss_retriever = FAISS(
            embedding_function=self.fim.embedding,
            index=faiss_index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        ).as_retriever()

        self.fim.close()

        return EnsembleRetriever(
            retrievers=[self.kiwi_bm25, faiss_retriever],  # 사용할 검색 모델의 리스트
            weights=self.weights,  # 각 검색 모델의 결과에 적용할 가중치
            search_type="mmr",  # 검색 결과의 다양성을 증진시키는 MMR 방식을 사용
        )

    def _create_save_new_index(self):
        faiss_index, docstore, index_to_docstore_id = self.fim.create_faiss_index_in_batches(self.docs)
        try:
            print("Deleting existing Faiss index...")
            self.fim.delete_collection()
        except:
            pass
        finally:
            self.fim.save_faiss_index_to_mongo(faiss_index, docstore, index_to_docstore_id)
            print("Saved New Faiss index to MongoDB.")
        return faiss_index, docstore, index_to_docstore_id
```

### RAG 체인 상세 설명

이 섹션에서는 [`chain.py`](http://chain.py) 파일에 정의된 RAG 체인의 구성 및 동작 방식을 설명합니다. 이 체인은 사용자와의 상호작용을 통해 질문을 받고, 해당 질문에 대한 문서를 검색한 후, 최종적으로 관련있는 답변을 생성합니다.

#### 체인의 구성 요소

1. **RunnableMap**:
    
    * `RunnableMap`은 여러 개의 실행 가능한 단계를 연결하여 데이터 흐름을 구성합니다. 이를 통해 사용자의 입력에서 시작하여 최종 출력까지의 여러 단계를 순차적으로 처리합니다.
        
2. **RunnablePassthrough**:
    
    * 이 컴포넌트는 입력을 직접적으로 다음 단계로 전달합니다. 여기서는 사용자의 대화 이력(`chat_history`)을 받아서 포매팅 함수에 전달하여 처리합니다.
        
3. **formatting functions** (`_format_chat_history`, `_combine_documents`):
    
    * `_format_chat_history`: 사용자와 봇 간의 대화 이력을 문자열로 포맷팅합니다. 이는 후속 처리를 위해 구조화된 데이터를 문자열로 변환하는 데 사용됩니다.
        
    * `_combine_documents`: 검색된 문서들을 결합하여 하나의 긴 문자열로 만듭니다. 이 문자열은 질문에 대한 답변을 생성하는 데 사용됩니다.
        
4. **ChatOpenAI**:
    
    * `ChatOpenAI` 객체는 OpenAI의 GPT 모델을 사용하여 자연어 처리를 수행합니다. 이 객체는 구성된 URI와 모델 정보를 바탕으로 API 호출을 통해 질문에 대한 답변을 생성합니다.
        
5. **KiwiBM25FaissEnsembleRetriever**:
    
    * 이 리트리버는 BM25와 Faiss 검색 엔진을 결합한 앙상블 방식을 사용하여 관련 문서를 검색합니다. 한국어 텍스트에 최적화된 `Kiwi` 토크나이저를 통해 입력된 질문을 효과적으로 처리하고, 더 정확한 문서 검색을 가능하게 합니다.
        

#### 체인의 실행 흐름

* 사용자의 질문이 입력되면, 첫 번째로 `RunnablePassthrough`를 통해 대화 이력이 포매팅됩니다.
    
* 포매팅된 대화 이력은 `KiwiBM25FaissEnsembleRetriever`로 전달되어 관련 문서를 검색합니다.
    
* 검색된 문서들은 `_combine_documents` 함수를 통해 결합되며, 이렇게 생성된 문서 집합은 `ChatOpenAI` 모델에 전달되어 최종적인 답변을 생성합니다.
    

이 체인을 사용함으로써, 복잡한 검색 및 응답 생성 과정을 효율적으로 관리하고, 사용자 질문에 대해 정확하고 관련성 높은 답변을 신속하게 제공할 수 있습니다. 이 구조는 또한 새로운 기능이나 다른 모델로의 확장성을 고려하여 설계되었습니다.

#### 소스 코드

```python
import os
import pickle
from dotenv import dotenv_values

from operator import itemgetter
from typing import List, Tuple

from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import format_document
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough, RunnableMap

from gpt_recruit_rag.prompts import CONDENSE_QUESTION_PROMPT, DOCUMENT_PROMPT, LLM_CONTEXT_PROMPT
from gpt_recruit_rag.retrievers import KiwiBM25FaissEnsembleRetriever

config = dotenv_values("packages/gpt-recruit-rag/.env")

MONGO_URI = config["MONGO_URI"]
db_name = "faiss_index"
collection_name = "wanted_job_details_index_v1"
EMBEDDING_URI = config["EMBEDDING_URI"]

kiwi_bm25_path = 'packages/gpt-recruit-rag/kiwi_bm25.pkl'
with open(kiwi_bm25_path, 'rb') as inp:
    kiwi_bm25 = pickle.load(inp)

kiwibm25_faiss_73 = KiwiBM25FaissEnsembleRetriever(
    mongo_uri=MONGO_URI,
    db_name=db_name,
    collection_name=collection_name,
    embedding_uri=EMBEDDING_URI,
    kiwi_bm25=kiwi_bm25,
    weights=[0.7, 0.3],
    create_index=False,
)
my_retriever = kiwibm25_faiss_73.build_ensemble_retriever()

os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
llm = ChatOpenAI(
    base_url=config["RUO_LLM_URI"],
    model=config["RUO_LLM_MODEL"],
    temperature=0.7
)


def _combine_documents(
    docs, document_prompt=DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def _format_chat_history(chat_history: List[Tuple]) -> str:
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer


_inputs = RunnableMap(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: _format_chat_history(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | llm
    | StrOutputParser(),
)

_context = {
    "context": itemgetter("standalone_question") | my_retriever | _combine_documents,
    "question": lambda x: x["standalone_question"],
}


class ChatHistory(BaseModel):
    """Chat history with the bot."""

    chat_history: List[Tuple[str, str]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "question"}},
    )
    question: str

conversational_qa_chain = (
    _inputs | _context | LLM_CONTEXT_PROMPT | llm | StrOutputParser()
)

chain = conversational_qa_chain.with_types(input_type=ChatHistory)
```

### 결과 및 테스트

시스템을 여러 한국어 데이터셋에 대해 테스트한 결과, `Kiwi` 토크나이저를 사용함으로써 검색 정확도가 기존 시스템 대비 상당히 향상되었습니다. 또한, `faiss`와 `KiwiBM25`의 조합은 Elasticsearch의 KNN 기능과 비교하여 비용 효율성이 뛰어나면서도 우수한 검색 결과를 제공했습니다.

### 감사의 말

이 프로젝트를 진행하면서 많은 기술 문서와 커뮤니티의 지원을 받았습니다. 특히, `langchain`과 관련된 다양한 구현 사례를 다룬 유튜버 teddynote님의 "랭체인 코리아" 시리즈는 이 프로젝트에 매우 큰 도움이 되었습니다. [teddynote님의 Kiwi-BM25Retriever 구현 예시](https://github.com/teddylee777/langchain-kr/blob/main/11-Retriever/10-Kiwi-BM25Retriever.ipynb)는 한국어 텍스트 처리 및 검색 성능 향상에 관한 훌륭한 참고 자료였습니다. 이 자료를 바탕으로 더 효율적인 검색 엔진을 개발할 수 있었으며, 프로젝트의 성공적인 완성에 크게 기여했습니다.

### 최종 결과 테스트

이 프로젝트의 최종 구현 결과는 [GPT Recruit 웹사이트](https://gpt-recruit.com/)에서 직접 테스트해볼 수 있습니다. 이 웹사이트를 통해 사용자는 직접 질문을 입력하고, 구현된 검색 엔진이 어떻게 관련 문서를 검색하고 응답을 생성하는지 경험할 수 있습니다. 이 테스트를 통해 프로젝트의 실용성과 효율성을 직접 확인해보세요.

### 결론

이 프로젝트를 통해 한국어 데이터에 대한 효과적인 검색 시스템을 구축할 수 있었습니다. 무엇보다 `Kiwi` 토크나이저의 도입과 인덱스의 재사용 방법이 큰 성공을 거두었으며, `faiss`의 사용은 높은 비용의 상용 솔루션에 의존하지 않고도 효과적인 검색 결과를 얻을 수 있음을 입증했습니다. 앞으로 이 시스템을 다른 언어와 도메인에 적용하여 그 범위를 확장할 계획입니다. 프로젝트에 도움을 준 모든 자료와 커뮤니티에 감사드립니다.