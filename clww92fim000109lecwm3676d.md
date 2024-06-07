---
title: "쿠버네티스에서 PostgreSQL 데이터베이스 미러링 및 주기적 백업 설정"
datePublished: Sat Jun 01 2024 15:10:28 GMT+0000 (Coordinated Universal Time)
cuid: clww92fim000109lecwm3676d
slug: postgresql
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1717254481470/7accc521-57d7-4e18-9546-83af4bd92ded.webp
tags: postgresql, docker, kubernetes, devops, it, backup, docker-compose, kubernetes-cluster, database-mirroring, 642w7j207ysw67kg7j207iqkiouwseyxhq

---

### 본문

#### 1\. 서론

**프로젝트 개요** 이번 프로젝트에서는 쿠버네티스 클러스터에서 PostgreSQL 데이터베이스를 미러링하고, 주기적으로 백업하는 방법을 설명합니다. 이를 통해 데이터의 신뢰성과 가용성을 높일 수 있습니다.

**환경 설정 및 요구사항**

* 쿠버네티스 클러스터 (노드 3대)
    
* PostgreSQL 14
    
* 백업 서버 (맥미니, Ubuntu Server 설치)
    
* Docker 및 Docker Compose
    

#### 2\. 준비 단계

**쿠버네티스 클러스터 설정** 먼저 쿠버네티스 클러스터를 설정합니다. 쿠버네티스 설치 및 설정은 이 블로그의 범위를 벗어나므로, 설치된 상태에서 시작합니다.

**PostgreSQL 배포 및 설정** 쿠버네티스 클러스터에 PostgreSQL을 배포합니다. 아래는 PostgreSQL 배포를 위한 YAML 파일 예제입니다.

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: postgres-config
  namespace: postgres
data:
  postgresql.conf: |
    listen_addresses = '*'
    wal_level = replica
    max_wal_senders = 3
    wal_keep_segments = 64
    archive_mode = on
    archive_command = 'cp %p /var/lib/postgresql/data/archive/%f'
    hot_standby = on
  pg_hba.conf: |
    local   all             all                                     trust
    host    all             all             0.0.0.0/0               md5
    host    replication     all             0.0.0.0/0               md5
    host    replication     all             ::/0                    md5

---

apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: postgres
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
        - name: postgres
          image: postgres:14
          ports:
            - containerPort: 5432
          env:
            - name: POSTGRES_USER
              value: postgres
            - name: POSTGRES_PASSWORD
              value: yourpassword
          volumeMounts:
            - mountPath: /var/lib/postgresql/data
              name: postgredb
      volumes:
        - name: postgredb
          persistentVolumeClaim:
            claimName: pg-nfs-pvc
```

#### 3\. PostgreSQL 미러링 설정

**주 서버 및 복제 서버 설정** 먼저 주 서버와 복제 서버를 설정합니다. 주 서버의 PostgreSQL 설정 파일을 수정합니다.

`postgresql.conf` 파일 수정:

```plaintext
listen_addresses = '*'
wal_level = replica
max_wal_senders = 3
wal_keep_size = 1024
archive_mode = on
archive_command = 'cp %p /var/lib/postgresql/data/archive/%f'
hot_standby = on
```

`pg_hba.conf` 파일 수정:

```plaintext
local   all             all                                     trust
host    all             all             0.0.0.0/0               md5
host    replication     all             0.0.0.0/0               md5
host    replication     all             ::/0                    md5
```

**복제 사용자 생성 및 권한 설정**

주 서버에서 복제 사용자를 생성합니다.

```sql
CREATE ROLE replica WITH REPLICATION LOGIN PASSWORD 'replica_password';
```

**ConfigMap 및 Deployment 설정**

쿠버네티스 ConfigMap과 Deployment를 설정하여 PostgreSQL 컨테이너를 배포합니다. 이 과정은 위의 준비 단계에서 이미 설명한 YAML 파일을 참조하세요.

#### 4\. 데이터 복제

`pg_basebackup`을 사용한 초기 데이터 복제

복제 서버에서 `pg_basebackup`을 사용하여 초기 데이터를 복제합니다. 아래는 초기화 스크립트 예제입니다.

```bash
#!/bin/bash
set -e

ARCHIVE_DIR="/var/lib/postgresql/data/archive"

# 초기화할 때만 데이터 복제
if [ -z "$(ls -A /var/lib/postgresql/data)" ]; then
    echo "Data directory is empty, starting pg_basebackup..."
    PGPASSWORD='replica_password' pg_basebackup -h <주 서버 IP> -p 5432 -D /var/lib/postgresql/data -U replica -v -P --wal-method=stream
    echo "primary_conninfo = 'host=<주 서버 IP> port=5432 user=replica password=replica_password'" >> /var/lib/postgresql/data/postgresql.auto.conf
    touch /var/lib/postgresql/data/recovery.signal
else
    echo "Data directory is not empty, skipping pg_basebackup..."
fi

echo "Starting PostgreSQL server..."
pg_ctl start -D /var/lib/postgresql/data -l /var/lib/postgresql/data/logfile
```

#### 5\. 주기적 백업 스크립트 작성

백업 스크립트를 작성하여 주기적으로 데이터를 백업합니다. 아래는 백업 스크립트 예제입니다.

```bash
#!/bin/bash

BACKUP_ROOT_DIR=/home/ruo/postgres_backups
TIMESTAMP=$(date +%F_%H-%M-%S)
DATE_DIR=$(date +%F)

# 날짜별 백업 디렉토리 생성
BACKUP_DIR="$BACKUP_ROOT_DIR/$DATE_DIR"
mkdir -p $BACKUP_DIR

# 백업할 데이터베이스 목록
DBS=("db1", "db2", ...)

# 각 데이터베이스를 백업
for DB in "${DBS[@]}"
do
    BACKUP_FILE="$BACKUP_DIR/${DB}_backup_$TIMESTAMP.sql"
    docker exec postgres_backup pg_dump -U postgres -d $DB > $BACKUP_FILE
done

# 오래된 백업 삭제 (3일 이상 된 백업 폴더 삭제)
find $BACKUP_ROOT_DIR -maxdepth 1 -type d -mtime +3 -exec rm -rf {} \;
```

#### 6\. 크론잡 설정

크론잡을 설정하여 백업 스크립트를 주기적으로 실행합니다.

```bash
crontab -e
```

크론탭 파일에 다음 라인을 추가합니다 (매일 새벽 3시에 실행):

```bash
0 3 * * * /home/ruo/postgres_backup_all.sh
```

크론잡 설정을 확인합니다.

```bash
crontab -l
```

#### 7\. 연결 상태 확인

원본 서버와 복제 서버 간의 연결 상태를 확인하는 과정은 중요합니다. 이를 통해 미러링 구성이 정상적으로 작동하고 있는지 확인할 수 있습니다.

#### 원본 서버에서 확인

원본 PostgreSQL 서버에서는 `pg_stat_replication` 뷰를 통해 복제 상태를 확인할 수 있습니다. 이 뷰는 복제 중인 각 세션에 대한 정보를 제공합니다. 아래 SQL 명령을 실행하여 복제 상태를 확인합니다:

```sql
SELECT * FROM pg_stat_replication;
```

이 쿼리는 복제를 수행하고 있는 프로세스의 상세 정보를 반환합니다. 예상한 복제 슬롯과 연결된 세션 정보가 표시되어야 합니다. 정보가 보이지 않는 경우, 복제 설정을 다시 검토해야 할 수 있습니다.

#### 복제 서버에서 확인

복제 서버에서는 `pg_stat_wal_receiver` 뷰를 사용하여 현재 WAL 수신 상태를 확인할 수 있습니다. 이 뷰는 원본 서버로부터 WAL 데이터를 받고 있는지, 그리고 그 상태가 어떠한지를 보여줍니다:

```sql
SELECT * FROM pg_stat_wal_receiver;
```

복제가 올바르게 설정되어 있다면, 이 쿼리는 원본 서버의 연결 정보와 함께 WAL 데이터의 수신 상태를 보여줄 것입니다. 데이터가 업데이트되지 않거나, 연결 상태에 문제가 있는 경우 추가적인 트러블슈팅이 필요할 수 있습니다.

#### 네트워크 연결 테스트

복제에 문제가 발생하는 경우, 네트워크 연결을 테스트하여 두 서버 간의 통신이 원활하게 이루어지고 있는지 확인해야 합니다. 예를 들어, `telnet`을 사용하여 원본 서버의 PostgreSQL 포트로 연결을 시도할 수 있습니다:

```bash
telnet <원본 서버 IP> <PostgreSQL 포트>
```

이 커맨드가 성공한다면 네트워크 연결은 정상입니다. 실패한다면 네트워크 설정, 방화벽 규칙 또는 원본 서버의 PostgreSQL 설정을 다시 확인해야 합니다.

#### 8\. 결론

**요약 및 추가 고려사항** 이번 포스트에서는 쿠버네티스에서 PostgreSQL 데이터베이스를 미러링하고 주기적으로 백업하는 방법을 다뤘습니다. 이를 통해 데이터의 신뢰성과 가용성을 높일 수 있습니다.

**참고 자료**

* [PostgreSQL 공식 문서](https://www.postgresql.org/docs/)
    
* [쿠버네티스 공식 문서](https://kubernetes.io/docs/)
    
* [Docker 공식 문서](https://docs.docker.com/)
    

이제 이 블로그 포스트를 통해 쿠버네티스에서 PostgreSQL 데이터베이스를 효과적으로 관리할 수 있을 것입니다. 추가적인 질문이나 도움이 필요하면 언제든지 댓글로 문의해 주세요.