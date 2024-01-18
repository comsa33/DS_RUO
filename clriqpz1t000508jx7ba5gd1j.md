---
title: "멀티스레딩과 셀레니움 그리드를 활용한 고성능 웹 스크래핑"
datePublished: Thu Jan 18 2024 04:57:22 GMT+0000 (Coordinated Universal Time)
cuid: clriqpz1t000508jx7ba5gd1j
slug: 66ma7yuw7iqk66ci65sp6ro8ioyfgougioulioybgcdqt7jrpqzrk5zrpbwg7zmc7jqp7zwcioqzooyeseukpsdsm7kg7iqk7ygs656y7zwr
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1705553715923/7db2f28a-14e7-442d-b78c-f9acdc1753ca.png
tags: data-science, multithreading, automation, selenium, selenium-grid, data-scraping, web-crawling, techblog, docker-container, python-web-scraper, 7ju5ioykpo2brouemo2vkq

---

### 서론

웹 스크래핑은 데이터 수집의 필수적인 방법 중 하나로 자리잡았습니다. 대용량 데이터를 효율적으로 수집하기 위해, 멀티스레딩과 셀레니움 그리드를 결합한 스크래핑 기법을 소개합니다. 이 글에서는 도커를 이용해 셀레니움 그리드를 설정하는 방법과 멀티스레딩으로 스크래핑 속도를 극대화하는 전략을 공유합니다.

### 셀레니움 그리드 설정

셀레니움 그리드는 여러 브라우저 인스턴스를 관리하고 동시에 다수의 작업을 수행할 수 있게 해줍니다. 이를 통해 스크래핑의 병렬 처리가 가능해져, 대규모 웹 사이트에서도 빠르게 데이터를 수집할 수 있습니다. 도커를 이용한 셀레니움 그리드의 구성은 다음과 같습니다:

```yaml
# docker-compose.yml
version: '3.3'
services:
  selenium-hub:
    image: selenium/hub:latest
    ports:
      - "4442:4442"
      - "4443:4443"
      - "4444:4444"
    environment:
      GRID_MAX_SESSION: 16
    security_opt:
      - seccomp:unconfined
  # Chrome 노드 설정 예시 (원하는 만큼 생성)
  chrome-node_1:
    image: selenium/node-chrome:latest
    depends_on:
      - selenium-hub
    ...
```

각 서비스는 셀레니움 그리드의 한 부분을 담당하며, 이러한 설정을 통해 여러 브라우저 세션을 동시에 실행할 수 있습니다.

#### 셀레니움 그리드와 도커 보안 옵션

도커를 사용할 때, 특정 버전에서 자바 애플리케이션의 메모리 할당과 관련된 문제가 발생할 수 있습니다. 이는 컨테이너가 자바 프로세스에 필요한 충분한 메모리를 할당하지 못할 때 나타나는 현상입니다. 특히, 도커 버전 20.10.10 이전 버전에서는 컨테이너의 기본 보안 프로파일 때문에 이러한 문제가 발생하는 것으로 알려져 있습니다.

#### 문제의 원인

##### 도커 컨테이너의 기본 보안 프로파일은 `seccomp`이라는 리눅스 커널 기능을 활용합니다. `seccomp`은 컨테이너가 수행할 수 있는 시스템 호출을 제한하여 보안을 강화합니다. 하지만, 일부 자바 애플리케이션은 많은 양의 메모리와 여러 시스템 호출을 필요로 하는데, 이 제한으로 인해 자바 프로세스가 필요한 메모리를 할당받지 못하고 실패하는 경우가 있습니다.

#### 해결 방안

이 문제에 대한 한 가지 해결 방안은 `seccomp` 프로파일을 비활성화하는 것입니다. `security_opt` 설정에 `- seccomp:unconfined`를 추가하여 컨테이너가 필요로 하는 모든 시스템 호출을 허용함으로써, 자바 애플리케이션이 필요한 메모리를 할당받을 수 있도록 합니다.

```yaml
security_opt:
  - seccomp:unconfined
```

이 설정은 메모리 관련 문제를 해결할 수 있으나, 보안상의 리스크를 증가시킬 수 있기 때문에, 가능하다면 도커를 최신 버전으로 업데이트하는 것이 권장됩니다. 최신 버전의 도커는 이러한 메모리 할당 문제를 해결하고, 동시에 보안을 강화한 기본 설정을 제공합니다.

#### docker 업데이트가 불가능할 때의 대처법

#### 도커를 업데이트할 수 없는 상황에서는 위의 `security_opt` 설정을 사용하는 것이 현실적인 선택입니다. 하지만 이 경우, 보안상의 위험을 최소화하기 위해 해당 설정이 적용된 컨테이너의 사용을 최소한으로 제한하고, 사용 후에는 반드시 컨테이너를 삭제하는 등의 조치를 취해야 합니다.

### 멀티스레딩 구현

멀티스레딩은 파이썬의 `ThreadPoolExecutor`를 사용해 구현할 수 있습니다. 이를 통해 동시에 여러 스크래핑 작업을 수행하여 성능을 개선할 수 있습니다. 아래는 멀티스레딩을 구현하는 방법을 보여주는 코드의 일부입니다:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

class VntrScraper:
    def scrape(self):
        executors_list = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            # 스크래핑 작업을 분배
            executors_list.append(executor.submit(self._scrape_vntr, vntr_list_1, 'SCP 1'))
            executors_list.append(executor.submit(self._scrape_vntr, vntr_list_2, 'SCP 2'))
            executors_list.append(executor.submit(self._scrape_vntr, vntr_list_3, 'SCP 3'))
            executors_list.append(executor.submit(self._scrape_vntr, vntr_list_4, 'SCP 4'))
            executors_list.append(executor.submit(self._scrape_vntr, vntr_list_5, 'SCP 5'))
            executors_list.append(executor.submit(self._scrape_vntr, vntr_list_6, 'SCP 6'))
            executors_list.append(executor.submit(self._scrape_vntr, vntr_list_7, 'SCP 7'))
            executors_list.append(executor.submit(self._scrape_vntr, vntr_list_8, 'SCP 8'))
            ...
```

스크래퍼 클래스 내에서 멀티스레딩을 구현하고 각 스레드가 벤처기업 목록의 일부를 스크래핑하도록 분배합니다.

### 실제 스크래핑 프로세스

실제 스크래핑 과정에서는 캡챠 인식, 로그인 처리, 예외 상황 대처 등 다양한 도전이 있습니다. 이러한 도전을 극복하기 위해 다음과 같은 전략을 사용했습니다:

1. 캡챠 인식 실패 시 재시도
    
2. 예외 발생 시 로그 기록 및 재시도
    
3. 스레드 안정성을 고려한 설계
    

### 성능 평가

이 방법을 사용함으로써, 웹 스크래핑 속도가 기존 대비 8배 향상되었습니다. (멀티스레드를 8개로 확장함) 벤처기업 데이터를 수집하는 데 걸리는 시간이 크게 줄어들었으며, 이는 비즈니스 결정을 내리는 데 사용되는 인사이트를 더 빠르게 제공할 수 있게 해줍니다.

### 결론

셀레니움 그리드와 멀티스레딩을 사용한 스크래핑 기법은 대규모 데이터 수집 프로젝트에 큰 이점을 제공합니다. 이 글을 통해 소개한 기법과 전략이 여러분의 프로젝트에도 도움이 되기를 바랍니다.