---
title: "Selenium WebDriver 최적화"
datePublished: Wed Jan 24 2024 02:43:36 GMT+0000 (Coordinated Universal Time)
cuid: clrr6l2qc00010al5400fc7da
slug: selenium-webdriver
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1706063754347/ec643155-9f55-4de3-91fd-43d993815ccc.png
tags: selenium, selenium-grid, webscraping, data-engineering, techblog, web-crawler, zombie-process, webdriver-optimization

---

**서론**:

* 웹 스크래핑은 데이터 수집에 필수적인 작업이지만, 성능과 안정성에 관한 문제가 종종 발생합니다. 최근 프로젝트에서도 이러한 문제를 경험했습니다. 특히, Selenium WebDriver를 사용할 때 좀비 프로세스 발생과 메모리 부족 문제가 빈번했습니다.
    

**문제 발생**:

1. 좀비 프로세스 증가: 스크래퍼가 실행되면서 시간이 지날수록 프로세스 ID(PID)가 급격히 증가하고, 시스템 리소스 소비가 늘어났습니다.
    
    ```bash
      root     32358  0.0  0.0      0     0 pts/0    Z    09:36   0:00 [chrome] <defunct>
    ```
    
2. 메모리 부족: 스크래핑 동안 브라우저 인스턴스가 메모리를 과도하게 사용하며, 이로 인해 시스템이 불안정해지고 크래시가 발생했습니다.
    

**해결 과정**:

* 문제 원인 탐색: WebDriver 인스턴스가 적절히 종료되지 않아 발생하는 리소스 누수 문제로 판단
    
* 해결책 탐구: ChromeDriver 설정 변경, Selenium Grid 환경 최적화 등 다양한 시도
    

**구체적 해결책**:

1. `chrome_options.add_argument('--no-zygote')` 옵션 추가: 이 옵션은 ChromeDriver의 좀비 프로세스 생성을 방지합니다.
    
2. Selenium 그리드 최적화: Selenium 그리드에서 노드의 수를 8개에서 4개로 줄이고, 멀티스레드 수도 4개로 조정했습니다.
    
3. `with webdriver.Remote()` 사용: 컨텍스트 매니저를 사용하여 드라이버 인스턴스를 안정적으로 관리하도록 했습니다.
    

```python
# Chrome 옵션 설정
chrome_options = Options()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--headless=new")
chrome_options.add_argument('--no-zygote')  # 중요: 좀비 프로세스 문제 해결
chrome_options.add_argument('--disable-gpu')
command_executor = 'http://localhost:4444/wd/hub'

# WebDriver 시작 -> Remote 드라이버를 통해 selenium Grid를 사용 (분산 스크래핑)
with webdriver.Remote(
    command_executor=command_executor,
    options=chrome_options
) as driver:
    # 셀레니움 작업 실행
```

**결과 및 효과**:

* 좀비 프로세스 생성이 중단되고 시스템 리소스 사용량이 안정화됨
    
* PID 수가 안정적으로 유지되며, 이전보다 시스템 성능 및 안정성 향상
    
* 코드 최적화로 인한 시스템의 지속 가능한 운영 가능
    

**성능 향상 수치**

* 좀비 프로세스 및 메모리 문제 해결 후, 스크래핑 도구의 안정성이 크게 향상되었습니다. 메모리 사용량이 1.8GiB로 안정적으로 유지되었으며, PID의 증가가 없어 시스템 리소스를 보다 효율적으로 활용할 수 있게 되었습니다. 이는 장기적으로 스크래핑 도구의 성능을 유지하는 데 큰 도움이 되었습니다.
    

**결론**:

* 웹 스크래핑 프로세스의 성능과 안정성을 향상시키는 것은 중요합니다. 본 글에서 제시한 해결책은 유사한 문제에 직면한 다른 개발자들에게도 유용할 수 있으며, 이를 통해 보다 효율적이고 안정적인 스크래핑 환경을 구축할 수 있습니다.
    

**참고 사이트:**

[https://dev.to/styt/resolving-seleniums-zombie-process-issue-pak](https://dev.to/styt/resolving-seleniums-zombie-process-issue-pak)