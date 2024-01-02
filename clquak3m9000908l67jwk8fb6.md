---
title: "비동기 프로그래밍"
datePublished: Mon Jan 01 2024 02:18:26 GMT+0000 (Coordinated Universal Time)
cuid: clquak3m9000908l67jwk8fb6
slug: 67me64z6riwio2uhouhnoq3uouemouwjq
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1704075088771/eef21388-8085-4b81-a4d2-ead1311a31ae.webp
tags: python, async, semaphore, fastapi, 67me64z6riw

---

## 비동기 프로그래밍의 이해

비동기 프로그래밍은 병렬 처리의 일종으로, I/O 작업이나 네트워크 요청과 같이 오래 걸리는 작업을 기다리는 동안 다른 작업을 계속 진행할 수 있게 해줍니다. 이는 프로그램의 효율성과 반응성을 크게 향상시킬 수 있습니다.

### 원리

기본적으로, 비동기 프로그래밍은 작업이 완료될 때까지 기다리지 않고, 작업이 끝나면 콜백을 통해 결과를 반환받습니다. 이는 '이벤트 루프'라는 구조를 통해 관리되며, Python에서는 `asyncio` 라이브러리가 이를 담당합니다.

## FastAPI의 비동기 지원

FastAPI는 비동기 I/O를 네이티브로 지원하는 현대적인 Python 웹 프레임워크입니다. Flask와 달리, FastAPI는 Python 3.6+의 타입 힌트와 함께 `async`와 `await` 키워드를 사용하여 비동기 프로그래밍을 쉽게 구현할 수 있도록 설계되었습니다.

### 선택 이유

Flask는 기본적으로 동기적으로 작동하기 때문에, 동시에 많은 요청을 처리해야 하는 경우 성능상의 제한을 가집니다. 반면 FastAPI는 비동기 처리를 통해 이러한 제한 없이 높은 동시성을 제공합니다.

## 비동기를 효율적으로 사용하기 위한 방법

비동기 프로그래밍을 효율적으로 사용하기 위해서는, 동시에 실행되는 작업의 수를 조절하고, 공유 자원에 대한 접근을 관리해야 합니다.

### 세마포어의 이해와 설정 방법

세마포어는 동시에 실행할 수 있는 작업의 수를 제한하는 메커니즘입니다. FastAPI와 `asyncio`에서 세마포어를 설정하는 방법은 다음과 같습니다:

```python
semaphore = asyncio.Semaphore(10)
```

위 코드는 동시에 최대 10개의 작업만 실행되도록 제한합니다.

## 코드 예제 설명

비동기 작업을 관리하는 클래스 `DartFinanceScraper`를 만들어 API 호출을 비동기적으로 처리하는 방법입니다.

```python
class DartFinanceScraper:
    # ...
    async def _get_company_finance_info(self, ...):
        # ...
```

## 경험한 비동기 관련 에러들과 해결책

API 호출 횟수를 관리하는 과정에서 발생한 경쟁 조건을 해결하기 위해, 공유 상태의 동기화를 보장하는 방법을 적용했습니다. 또한, 자정에 API 호출 카운트를 리셋하는 로직을 도입하여 API 제한에 효과적으로 대응했습니다.

```python
if self._api_call_count >= self._api_call_limit:
    await self._wait_until_midnight()
    # ...
```

## 결론

비동기 프로그래밍은 막강한 동시성과 효율성을 제공하지만, 공유 자원의 동기화와 작업 관리에 주의가 필요합니다. FastAPI는 이러한 비동기 프로그래밍을 Python에서 쉽고 강력하게 구현할 수 있게 해주며, 오늘의 경험은 앞으로의 비동기 프로그래밍 작업에 큰 도움이 될 것입니다.