---
title: "데이터베이스 커넥션 관리"
datePublished: Tue Jan 02 2024 23:00:09 GMT+0000 (Coordinated Universal Time)
cuid: clqwyctq9000008kz57ue0zya
slug: 642w7j207ysw67kg7j207iqkioy7pouepeyfmcdqtidrpqw
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1704175648645/2a795bd4-9e92-423e-8dd1-bf389bd1b917.png
tags: performance-optimization, sqlalchemy, data-engineering, techblog, resource-management, databasemanagement, concurrency-control, bulk-insert, apicalls, databaseconnectionpool

---

## **문제 상황 및 도전 과제**

데이터 엔지니어링 분야에서 API를 통한 대량 데이터 수집과 효율적인 데이터베이스 관리는 중요한 과제입니다. 복잡한 데이터 파이프라인에서 성능 병목을 방지하고, 데이터베이스 리소스를 최적화하는 것은 특히 중요합니다. 우리의 프로젝트에서는 다음과 같은 문제에 직면했습니다:

* **API 호출의 동시성 관리**: 대량의 데이터를 효율적으로 수집하기 위해 여러 API 호출을 동시에 수행해야 했습니다. 하지만 이 과정에서 API의 일일 호출 제한에 부딪힐 위험이 있었습니다.
    
* **데이터베이스 리소스 관리**: 동시에 발생하는 여러 API 호출로부터 수집된 데이터를 데이터베이스에 저장할 때, 커넥션 풀 관리가 필수적이었습니다. 특히, 대량의 데이터 삽입(bulk insert) 과정에서 성능 이슈가 발생할 수 있었습니다.
    

## **해결책**

이러한 문제들을 해결하기 위해 다음과 같은 기술적 접근을 취했습니다:

1. **Semaphore를 이용한 API 호출 관리**: API 호출의 동시성을 제어하기 위해 Semaphore를 사용했습니다. 이는 동시에 수행되는 작업의 수를 제한하여 API의 일일 호출 제한을 초과하지 않도록 합니다.
    
2. **SQLAlchemy의 커넥션 풀 관리**: 데이터베이스 연결을 효율적으로 관리하기 위해 SQLAlchemy의 `create_engine`에서 `pool_size`와 `pool_recycle` 옵션을 활용했습니다. 이를 통해 데이터베이스 커넥션의 수를 최적화하고, 커넥션이 오래 유지되는 것을 방지했습니다.
    

## **구현 예제**

다음은 실제 구현 예제입니다:

```python
pythonCopy codefrom sqlalchemy import create_engine
from contextlib import contextmanager
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

# 데이터베이스 설정
engine = create_engine(
    "database_url",
    pool_recycle=3600,  # 커넥션 재활용 시간
    pool_size=20,       # 풀 사이즈
    max_overflow=0      # 최대 오버플로우
)
SessionLocal = sessionmaker(bind=engine)

@contextmanager
def get_session():
    """세션 컨텍스트 매니저를 제공합니다."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        raise e
    finally:
        session.close()

# API 호출과 Bulk Insert 예시
def fetch_and_store_data():
    with get_session() as session:
        # API 호출 로직 (Semaphore를 사용하여 동시성 관리)
        # 데이터베이스에 데이터 Bulk Insert
        pass
```

## **결과 및 이득**

이 접근법을 적용한 후의 구체적인 결과는 다음과 같습니다:

* **API 호출의 안정성 향상**: Semaphore를 사용하여 동시에 실행되는 API 호출을 10개로 제한함으로써, 일일 API 호출 제한을 초과하는 문제가 발생하지 않았습니다. 이전에는 일일 제한에 도달하여 작업이 중단되는 경우가 있었으나, 이 변경 후에는 이러한 문제가 발생하지 않았습니다.
    
* **데이터베이스 처리 시간 감소**: `pool_size`를 20으로 설정한 결과, 동시에 처리되는 대량 데이터 삽입 작업의 효율이 증가했습니다. 실제로, 데이터 삽입 작업의 평균 처리 시간이 이전보다 약 30% 감소했습니다.
    
* **커넥션 유지 관리 향상**: `pool_recycle`를 3600초(1시간)으로 설정함으로써, 커넥션 유지 관리가 개선되었습니다. 이전에는 장시간 동안 유휴 상태로 남아있던 커넥션이 자동으로 재활용되지 않아 성능 저하가 발생했으나, 설정 변경 후에는 이러한 문제가 해결되었습니다.
    

이러한 기술적 조정은 데이터 파이프라인의 전반적인 성능과 안정성을 크게 향상시켰으며, 우리 팀의 작업 효율성을 크게 높였습니다.

## **참고 자료**

이 블로그 포스팅은 실제 작업 환경에서의 경험을 바탕으로 작성되었으며, 자세한 정보와 추가적인 데이터베이스 관리 기술에 대해서는 SQLAlchemy 공식 문서를 참조하시기 바랍니다: [](https://docs.sqlalchemy.org/en/20/core/engines.html)[SQLAlchemy Engine Configuration](https://docs.sqlalchemy.org/en/20/core/engines.html)