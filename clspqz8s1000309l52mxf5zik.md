---
title: "쿠버네티스에서 Kubeflow와 GPU: Cgroup v1과 v2의 문제 해결기"
datePublished: Sat Feb 17 2024 07:18:40 GMT+0000 (Coordinated Universal Time)
cuid: clspqz8s1000309l52mxf5zik
slug: kubeflow-gpu-cgroup-v1-v2
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1708154181908/ef5a6ff3-9c2e-4658-9389-f8dc190e308d.webp
tags: ai, linux, docker, kubernetes, devops, containers, nvidia, gpu, kubeflow, containerd, system-administration, cgroups, tech-tutorial

---

최근 쿠버네티스 환경에서 Kubeflow를 사용하는 동안 GPU 인식 문제에 직면했습니다. 이 글에서는 이 문제를 해결한 과정과 배운 교훈을 공유하고자 합니다.

## 문제 상황

Kubeflow를 쿠버네티스 클러스터에 통합하려고 할 때, 특정 노드에서 NVIDIA GPU를 인식하지 못하는 문제에 직면했습니다. NVIDIA 관련 플러그인 파드들이 `failed to get device cgroup mount path: no cgroup filesystem mounted for the devices subsystem in mountinfo file`라는 오류 메시지를 계속해서 보여주었습니다.

## 원인 분석

문제의 원인을 파악하기 위해 리눅스의 cgroup(Control Groups)에 주목했습니다. Cgroup은 프로세스의 리소스 사용을 제한하고 관리하는 리눅스 커널의 기능입니다.

쿠버네티스 환경에서는 cgroup이 컨테이너의 리소스 사용을 관리하는 데 중요한 역할을 합니다. 이 경우, NVIDIA 컨테이너 런타임은 GPU 리소스를 컨테이너에 할당하는 데 cgroup을 사용합니다. 그런데, 시스템에 cgroup v2가 활성화되어 있었고, NVIDIA 컨테이너 런타임은 cgroup v1을 기반으로 작동합니다.

## 해결 방법

해결책은 시스템의 cgroup 버전을 v1으로 전환하는 것이었습니다. `/etc/default/grub` 파일에서 `GRUB_CMDLINE_LINUX` 항목에 `systemd.unified_cgroup_hierarchy=0`를 추가하고, `update-grub` 명령어로 GRUB 설정을 업데이트한 후 시스템을 재부팅했습니다.

이 변경 후, NVIDIA GPU를 성공적으로 인식할 수 있었고, Kubeflow 파드들이 정상적으로 작동하기 시작했습니다.

## 결론

이 경험은 쿠버네티스와 같은 복잡한 시스템에서는 하드웨어와 소프트웨어 간의 상호작용이 중요하다는 것을 보여줍니다. 특히, 리눅스 시스템의 핵심 기능인 cgroup의 버전 차이가 NVIDIA 컨테이너 런타임과 같은 중요한 컴포넌트의 동작에 영향을 줄 수 있음을 알 수 있었습니다.

이러한 문제를 해결하는 과정에서 시스템의 깊은 이해와 세심한 분석이 필요하다는 것을 다시 한번 깨달았습니다.