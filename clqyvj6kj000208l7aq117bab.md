---
title: "FastAPI를 활용한 인증 서버 구축 및 토큰 기반 인증 시스템 구현하기"
datePublished: Thu Jan 04 2024 07:16:39 GMT+0000 (Coordinated Universal Time)
cuid: clqyvj6kj000208l7aq117bab
slug: fastapi
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1704352496102/57776dbc-e873-4916-b163-0382bd494598.png
tags: authentication, backend-development, python-programming, fastapi, websecurity, api-development, api-security, jwt-tokenjson-webtokentoken-authenticationaccess-tokenjson-tokenjwt-securityjwt-authenticationtoken-based-authenticationjwt-decodingjwt-implementation, json-web-tokens-jwt, server-architecture

---

## **개요**

오늘날 데이터 보안은 웹 서비스 개발에서 중요한 요소 중 하나입니다. FastAPI를 사용하여 안전한 인증 시스템을 구축하는 방법을 살펴보겠습니다. 이 글에서는 인증 서버의 개발, 리소스 서버에 토큰 인증 기능 추가, 그리고 클라이언트에서 이를 사용하는 방법까지 단계별로 설명합니다.

## **1\. 인증 서버 개발**

### **인증 서버란 무엇인가**

인증 서버는 사용자의 자격 증명을 검증하고, 유효한 사용자에게 접근 토큰을 제공하는 서비스입니다. 이 토큰은 사용자가 다른 서비스(리소스 서버)에 접근할 수 있도록 하는 열쇠와 같은 역할을 합니다.

### **FastAPI와 JWT를 사용한 인증 서버 구축**

FastAPI는 비동기 Python 웹 프레임워크로, 강력한 인증 및 보안 기능을 제공합니다. JWT (JSON Web Tokens)는 사용자 인증에 널리 사용되는 방식으로, 토큰 자체에 정보를 담고 있어 높은 효율성을 제공합니다.

#### 주요 구현 단계:

1. **환경 설정**: 필요한 라이브러리 설치 (`fastapi`, `uvicorn`, `pyjwt`)
    
2. **JWT 토큰 생성 및 검증 로직 구현**: 사용자의 자격 증명에 대한 검증 후 JWT 토큰 발급
    
3. **사용자 인증 라우트 개발**: 사용자가 자격 증명을 전송할 수 있는 엔드포인트 생성
    

#### 코드 예시:

```python
from datetime import datetime, timedelta
from typing import Optional
from passlib.context import CryptContext

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import jwt

from app.settings import SECRET_KEY, ALGORITHM, USERNAME, PASSWORD

ACCESS_TOKEN_EXPIRE_MINUTES = 60*24*7

# 비밀번호 해싱을 위한 설정
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 관리자 자격 증명 설정
admin_credentials = {
    USERNAME: pwd_context.hash(PASSWORD)
}

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str, credentials_exception):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError:
        raise credentials_exception


# 사용자 인증 함수
def authenticate_user(username: str, password: str):
    hashed_password = admin_credentials.get(username)
    if not hashed_password:
        return False
    if not pwd_context.verify(password, hashed_password):
        return False
    return True


@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user_authenticated = authenticate_user(form_data.username, form_data.password)
    if not user_authenticated:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": form_data.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me")
async def read_users_me(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    return verify_token(token, credentials_exception)
```

### **보안 고려사항**

* **HTTPS 사용**: 데이터 전송 시 암호화를 보장하기 위해 HTTPS 사용
    
* **토큰 보안**: 토큰의 유효기간 설정, 민감 정보는 토큰에 포함하지 않기
    

## **2\. 리소스 서버에 토큰 인증 추가**

### **리소스 서버의 역할**

리소스 서버는 보호된 데이터와 기능을 호스팅하는 서버입니다. 사용자가 인증 서버로부터 발급받은 토큰을 사용하여 리소스 서버에 접근할 수 있습니다.

### **토큰 검증 로직 구현**

리소스 서버는 들어오는 요청에 포함된 토큰을 검증하여, 유효한 요청에만 서비스를 제공합니다.

#### 구현 방법:

* FastAPI의 `Depends` 기능을 사용하여 API 엔드포인트에 토큰 검증 로직 적용
    
* 토큰이 유효하지 않은 경우 접근 거부
    

#### 코드 예시:

```python
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
import jwt
from jwt import PyJWTError

from app.config.settings import SECRET_KEY, ALGORITHM

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except PyJWTError:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials"
        )
```

```python
@router.get("/dart/finance/business/{bizNum}", response_model=NewCompanyFinanceResponse)
async def get_company_finance_info(bizNum: str, token: str = Depends(verify_token)):
    """사업자등록번호로 기업 재무정보를 조회하는 함수
    Args:
        bizNum (str): 사업자등록번호
    Returns:
        NewCompanyFinanceResponse: 기업 재무정보
    """
    try:
        data = get_company_info(bizNum=bizNum)
        if not data:
            return NewCompanyFinanceResponse(newCompanyFinance=[])
        return NewCompanyFinanceResponse(newCompanyFinance=data)
    except Exception as e:
        err_msg = traceback.format_exc()
        logger.error(err_msg)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
```

### **보안 고려사항**

* **토큰 누출 방지**: API 요청 시 토큰을 안전하게 전송
    
* **적절한 권한 부여**: 사용자에게 필요한 최소한의 권한만 부여
    

## **3\. 클라이언트에서의 토큰 사용**

### **클라이언트의 역할**

클라이언트는 사용자가 인증 서버에 자격 증명을 제공하고, 발급받은 토큰을 사용하여 리소스 서버에 접근하는 인터페이스를 제공합니다.

### **토큰 발급 및 사용 방법**

1. **토큰 발급 요청**: 클라이언트는 인증 서버에 사용자 이름과 비밀번호를 전송하여 토큰을 요청
    
2. **토큰 저장 및 사용**: 발급받은 토큰을 저장하고, 리소스 서버에 요청을 보낼 때 헤더에 포함
    

#### 클라이언트 코드 예시 (Python, Java):

```python
import requests

# 인증 서버에서 토큰 발급
auth_response = requests.post("http://호스트:포트/token", data={
    "username": "아이디",
    "password": "비밀번호"
})
token = auth_response.json()["access_token"]
```

```java
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.Map;

public class Main {
    public static void main(String[] args) {
        HttpClient client = HttpClient.newHttpClient();
        String url = "http://호스트:포트/token";
        String formParams = "username=" + URLEncoder.encode("아이디", StandardCharsets.UTF_8) +
                            "&password=" + URLEncoder.encode("비밀번호", StandardCharsets.UTF_8);

        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(url))
            .header("Content-Type", "application/x-www-form-urlencoded")
            .POST(HttpRequest.BodyPublishers.ofString(formParams))
            .build();

        try {
            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
            System.out.println("Response status code: " + response.statusCode());
            System.out.println("Response body: " + response.body());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

### **보안 고려사항**

* **토큰 보안**: 클라이언트 측에서 토큰을 안전하게 저장 및 관리
    
* **HTTPS 사용**: 데이터 전송 시 암호화 보장
    

---

이 포스팅은 FastAPI를 활용한 인증 시스템의 구축과 클라이언트에서의 토큰 사용 방법을 자세하게 설명합니다. 보안을 최우선으로 고려하여 각 단계를 신중하게 설계하는 것이 중요합니다.