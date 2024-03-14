---
title: "[해결] [microk8s] metallb tls 인증서 오류"
datePublished: Thu Mar 14 2024 02:45:45 GMT+0000 (Coordinated Universal Time)
cuid: cltqmof8q000209jwg3mseszs
slug: microk8s-metallb-tls
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1710384224410/76068d83-c61d-4170-8553-a88b0a2b2adc.webp
tags: error-handling, tls, ssl-certificate, k8s, network-security, clustering, load-balancing, data-engineering, microk8s, ssl-certificateverifyfailed-error, metallb, ipaddresspool

---

### 문제

* 온프레미스 클러스터 운영 중 로드밸런싱 기술을 도입하기 위해 metallb를 설치
    
* ip address pool 설정을 적용하는 중 아래와 같은 에러 발생
    
    ```python
    apiVersion: metallb.io/v1beta1
    kind: IPAddressPool
    metadata:
      name: ruo-network-pool-1
      namespace: metallb-system
    spec:
      addresses:
      - 192.168.1.240-192.168.1.250 
    ```
    
    ```plaintext
    ❯ kubectl apply -f ruo-network.yaml
    Error from server (InternalError): error when creating "ruo-network.yaml": Internal error occurred: failed calling webhook "ipaddresspoolvalidationwebhook.metallb.io": failed to call webhook: Post "https://webhook-service.metallb-system.svc:443/validate-metallb-io-v1beta1-ipaddresspool?timeout=10s": tls: failed to verify certificate: x509: certificate signed by unknown authority
    ```
    
* 세부 로그를 확인하니 `2024/03/13 13:00:02 http: TLS handshake error from 10.1.119.192:48444: remote error: tls: bad certificate` TLS 핸드셰이크 과정에서 클라이언트의 인증서가 유효하지 않거나 신뢰할 수 없는 상황임.
    

### 이 문제를 해결하기 위한 몇 가지 접근 방법

1. **인증서 확인**: 해당 IP 주소로부터 요청하는 클라이언트의 TLS 인증서가 유효한지, 만료되지 않았는지, 신뢰할 수 있는 CA에 의해 서명되었는지 확인하세요.
    
2. **서버 설정 검토**: 서버가 올바른 인증서를 사용하고 있는지, 그리고 서버의 TLS 설정이 클라이언트 인증서와 호환되는지 확인하세요.
    
3. **네트워크 트래픽 분석**: 네트워크 패킷 분석 도구(예: Wireshark)를 사용하여 TLS 핸드셰이크 과정을 보다 자세히 분석하고, 어떤 단계에서 문제가 발생하는지 확인할 수 있습니다.
    
4. **인증서 로그 확인**: 서버 또는 클라이언트의 로그에서 추가적인 오류 메시지나 경고를 확인하여 문제의 원인을 좀 더 구체적으로 파악할 수 있습니다.
    
5. **클라이언트 구성 검토**: 문제가 발생하는 클라이언트의 TLS 구성을 검토하고, 필요하다면 적절한 인증서를 구성하거나 갱신하세요.
    

### **쿠버네티스에서 인증서 확인하기**

쿠버네티스 환경에서 `ingress-nginx`를 사용한다면, SSL/TLS 인증서는 쿠버네티스의 시크릿(Secrets) 형태로 관리될 가능성이 높습니다. 이 경우, 인증서 파일이 서버의 파일 시스템에 직접 저장되지 않고, 쿠버네티스 클러스터 내에서 관리됩니다.

#### **1\. Webhook 서비스의 TLS 인증서 확인**

먼저, 해당 webhook 서비스가 사용하는 TLS 인증서의 상태를 확인합니다. 이는 쿠버네티스의 시크릿으로 저장되어 있을 가능성이 높습니다.

```plaintext
❯ kubectl get secret -n metallb-system
```

이 명령어를 통해 `metallb-system` 네임스페이스 내의 시크릿들을 확인할 수 있습니다. 여기에서 `webhook-service`와 관련된 TLS 인증서를 찾으세요.

#### **2\. 인증서의 상세 정보 확인**

인증서의 상세 정보를 확인하여 유효기간, 발급자 등의 정보를 검토합니다.

```plaintext
❯ kubectl describe secret <webhook-service-secret-name> -n metallb-system
```

#### **3\. 인증서 만료 여부 확인**

인증서가 만료되었거나 신뢰할 수 없는 CA에 의해 발급된 경우, 인증서를 갱신하거나 새로운 인증서를 발급받아야 합니다. 이 과정은 인증서를 발급한 방식에 따라 다릅니다.

#### **4\. Webhook 설정 검토**

`metallb-system`의 webhook 설정도 검토해보세요. webhook 구성이 올바르게 설정되어 있는지 확인합니다.

```plaintext
❯ kubectl get mutatingwebhookconfigurations,validatingwebhookconfigurations -A
```

#### **5\. 인증서 갱신 및 적용**

만약 인증서가 만료되었거나 문제가 있다면, 새로운 인증서를 생성하고 적용해야 합니다. 새 인증서를 적용한 후에는 관련 서비스를 재시작하거나 클러스터를 업데이트하여 변경 사항을 적용합니다.

내 경우 5번에 해당함.

### 인증서 갱신

웹훅 시크릿 인증서를 확인해보자.

```plaintext
❯ kubectl get secret -n metallb-system webhook-server-cert -o yaml
```

```plaintext
apiVersion: v1
data:
  ca.crt: LS0tLS...[BASE64로 인코딩 된 코드]
  ca.key: LS0tLS...[BASE64로 인코딩 된 코드]
  tls.crt: LS0tLS...[BASE64로 인코딩 된 코드]
  tls.key: LS0tLS...[BASE64로 인코딩 된 코드]
kind: Secret
...(나머지 설정)
```

위 결과 중 `tls.crt > ca.crt` 이 두개의 파일이 서로 서명 인증이 제대로 안되었거나 뭔가가 잘못된거임. (왜 안맞는지 이유는 파악을 못함...)

그래서 다시 새로 만들어줘야함.

이제 웹훅 인증하는 애를 확인해보자.

```plaintext
❯ kubectl get validatingwebhookconfiguration.admissionregistration.k8s.io/metallb-webhook-configuration -o yaml
```

```plaintext
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfiguration
metadata:
  annotations:
    kubectl.kubernetes.io/last-applied-configuration: |
      {"apiVersion": ...(생략)...
creationTimestamp: "2024-03-13T11:12:42Z"
  generation: 18
  name: metallb-webhook-configuration
  resourceVersion: "88450654"
  uid: ee55b562-c288-4e7b-a3f5-9bc72614cd8e
webhooks:
- admissionReviewVersions:
  - v1
  clientConfig:
    caBundle: LS0tLS...[BASE64로 인코딩 된 코드]
... (생략)
...
  clientConfig:
    caBundle: LS0tLS...[BASE64로 인코딩 된 코드]
...
...
  clientConfig:
    caBundle: LS0tLS...[BASE64로 인코딩 된 코드]
...
...
  clientConfig:
    caBundle: LS0tLS...[BASE64로 인코딩 된 코드]
    service:
      name: webhook-service
      namespace: metallb-system
      path: /validate-metallb-io-v1beta1-addresspool
...
```

여기 caBundle 이후 인코딩된 인증서 정보도 다 새로 만든걸로 바꿔줘야한다.

직접 디코딩해서 비교해보니, 인증서의 아래 부분 등, 서명정보 및 Issuer, Basic constraints 등 정보들이 달랐다.

```plaintext
X509v3 extensions:
            X509v3 Subject Alternative Name: 
                DNS:webhook-service.metallb-system.svc
            X509v3 Key Usage: critical
                Digital Signature, Key Encipherment
            X509v3 Basic Constraints: critical
                CA:TRUE
            X509v3 Subject Key Identifier: 
                99:9E:98:07:51:02:B8:F8:37:15:03:F3:1E:6D:0F:34:CE:66:0A:B4
```

이제 새로 만들어보자.

파이썬으로 간단하게 CA에서 발급한 파일이 쿠버네티스 시크릿 정보로 들어있으니 이를 파일로 만들어줘야한다.

```plaintext
import base64


def decode_base64_and_save(encoded_string, file_path):
    # Decode the base64 encoded string
    decoded_bytes = base64.b64decode(encoded_string.encode('utf-8'))

    # Write the decoded bytes to the specified file
    with open(file_path, 'wb') as file:
        file.write(decoded_bytes)

# Base64 encoded strings for ca.crt and ca.key
# The actual base64 strings are omitted for brevity

ca_crt_base64 = "이전 웹훅 시크릿에서 ca.crt 인코딩 된 정보"
ca_key_base64 = "이전 웹훅 시크릿에서 ca.key 인코딩 된 정보"

# File paths to save the decoded certificates
ca_crt_file_path = "ca.crt"
ca_key_file_path = "ca.key"

# Decode and save the certificates
decode_base64_and_save(ca_crt_base64, ca_crt_file_path)
decode_base64_and_save(ca_key_base64, ca_key_file_path)

ca_crt_file_path, ca_key_file_path
```

파일이 만들어졌으면 이제 이를 기반으로 해당 CA로부터 서명받은 웹훅crt와 key를 만들어야한다.

만들기 전 `openssl.snf`라는 파일을 만들어 다음처럼 입력

```plaintext
[ req ]
default_bits = 2048
defautl_md = sha256
distinguished_name = req_distinguished_name
req_extensions = req_ext
x509_extensions = v3_ca
prompt = no

[ req_distinguished_name ]
O = metallb
CN = webhook-service.metallb-system.svc

[req_ext]
subjectAltName = @alt_names

[ v3_ca ]
subjectAltName = @alt_names
keyUsage = critical, digitalSignature, keyEncipherment
basicConstraints = critical,CA:TRUE
extendedKeyUsage = serverAuth
authorityKeyIdentifier = keyid,issuer:always

[ alt_names ]
DNS.1 = webhook-service.metallb-system.svc
```

위 정보는 ca.crt를 디코딩해서 정보를 직접 눈으로 확인 후 해당 정보들을 맞춰서 넣은거다.

```plaintext
# 인증서와 키 생성: 이제 설정 파일을 사용하여 인증서 요청(CSR)과 개인 키를 생성
❯ openssl req -new -nodes -out webhook.csr -newkey rsa:2048 -keyout webhook.key -config openssl.cnf

# CA에 의한 인증서 서명: 생성된 CSR을 사용하여 CA에 의해 서명된 인증서를 생성
# 이 과정에서 CA의 인증서와 키를 사용
# 여기서 ca.crt와 ca.key는 CA의 인증서와 개인 키의 경로임. -CAcreateserial 옵션은 CA가 각 인증서에 대해 고유한 일련 번호를 부여하는 데 사용
❯ openssl x509 -req -in webhook.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out webhook.crt -days 365 -extensions req_ext -extfile openssl.cnf

# 확인: 생성된 인증서에 원하는 확장이 포함되었는지 확인
❯ openssl x509 -in webhook.crt -text -noout

# 서명된 인증서를 ca.crt로 인증확인
❯ openssl verify -CAfile ca.crt webhook.crt
webhook.crt: OK <<< 이렇게 ok뜨면 됨.
```

ok가 안뜨면 뭔가 인증서 만들때 디코딩된 ca.crt 정보를 다시 잘보고 이것저것 잘 맞춰서 바꿔봐야함.

이제 새로만든 `webhook.crt`, `webhook.key` 가 ca인증이 잘 된걸 확인했으니, 다시 쿠버네티스 시크릿과 웹훅 인증확인 yaml을 새 인증서 정보로 업데이트 해줘야함.

```plaintext
# 웹훅 시크릿 파일내, tls.crt, tls.key 에 해당하는 인증서 데이터를 새로만든 인증서 데이터로 업데이트
❯ kubectl create secret generic webhook-server-cert \                                                          
  --from-file=tls.crt=webhook.crt \
  --from-file=tls.key=webhook.key \
  -n metallb-system \
  --dry-run=client -o yaml | kubectl apply -f -
```

```plaintext
# 웹훅 인증해주는 config 리소스도 업데이트
❯ kubectl patch validatingwebhookconfiguration metallb-webhook-configuration \                                 
  --type='json' \
  -p="[{\"op\": \"replace\", \"path\": \"/webhooks/0/clientConfig/caBundle\", \"value\":\"${base64_encoded_ca}\"}]"
```

`"/webhooks/0/clientConfig/` 여기에서 주의할 부분은 `0`이 부분이 아까 웹훅 인증 야믈을 보면 7개 서비스에 caBundle: 아래로 정보가 들어있었음. 그렇기 때문에 0부터 7까지 바꿔가면서 전부 업데이트 해줘야함.

이렇게 하면 드디어 완료.

```plaintext
❯ kubectl apply -f ruo-network.yaml
ipaddresspool.metallb.io/ruo-network-pool-1 created
```

다시 ipaddresspool 리소스를 생성. 성공!!!

이것 때문에 몇일을 검색하고 삽질을 수십번함... 나같은 삽질하지마시고 다들 행복한 코딩하시길 ㅋㅋㅋ

그럼 20000