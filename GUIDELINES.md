# 미분기하 Study Set 작성 가이드라인 (Agent용)

- 작성일: 2026-09-30
- 대상: 이 저장소에서 내용을 **작성·검토·수정**하는 모든 agent
- 결과물 형식: **HTML** (이 문서만 Markdown)

---

## 0. 이 문서를 읽는 법

- 이 문서는 study set 전체의 **정본(canonical) 규약**이다. 모든 agent는 작업을 시작하기 전에 이 문서를 **처음부터 끝까지** 읽는다. 규약들이 서로 얽혀 있으므로 필요한 부분만 골라 읽지 않는다.
- 우선순위: ① 사용자의 명시적 지시 → ② 이 문서 → ③ 참고 교재의 관행 → ④ agent 자신의 판단.
- §6(표기·규약)과 §7(기준 예제)을 바꾸려면 **사용자 승인**이 필요하다. 바꿔야 할 이유를 발견해도 작업을 멈추지 않는다. 해당 위치에 `[확인 필요]` 표시를 남기고 최종 보고에 적는다.
- 이 문서에 없는 상황에서는 가장 가까운 규칙의 취지를 따르고, 내린 결정을 최종 보고에 적는다.

### 한눈에 보기

1. 결과물은 **정적 HTML**이다. 한 절이 한 파일이고, 수식은 MathJax 3으로 쓰며, 공용 CSS와 설정 파일만 사용한다 (§11).
2. 본문 문장은 한국어 "~이다"체로 쓰고, 대학 수학 용어와 라벨·제목은 영어로 쓴다. 용어를 처음 정의할 때 한국어 번역어를 괄호에 한 번 적는다 (§8).
3. **아주 기초부터 차근차근** 쓴다. 순서는 동기 → 직관(그림) → 정의 → 예·비예 → 정리·증명 → 계산 → 요약 → 연습이다 (§2).
4. 26장이 **한 권의 책처럼** 이어지게 쓴다. 큰 줄기, 이어받는 것·넘겨주는 것, 한 개념 한 정의, 필수 호환성 명제, 기준 예제의 여정이 그 장치다 (§3).
5. 모든 정의와 정리에 가정을 빠짐없이 적는다. 이유 없는 "자명하다"는 쓰지 않는다 (§4).
6. 표기와 부호 규약은 §6 하나만 쓴다 (Lee와 do Carmo 기반). 교재와 다른 점은 본문에 명시한다.
7. §7 기준 예제의 매개화와 값을 모든 장에서 그대로 재사용한다.
8. 그림은 교재 그림을 **참고해서 직접 다시 그린다**. 실제 수식으로 계산해서 그린다 (§12).
9. 비자명한 계산은 sympy나 numpy 검증 스크립트로 확인하고 파일로 남긴다 (§13).
10. 교재의 정리·쪽 번호는 **확인한 것만** 적는다. 불확실하면 장 단위로만 인용하거나 `[확인 필요]`로 표시한다 (§4.4).
11. 작업을 끝내기 전에 §16 체크리스트를 확인하고 `bash tools/check_all.sh`를 통과시킨다.

---

## 1. 목표와 독자

- **목표**: 미분기하를 처음 배우는 사람이 혼자서도 교재 수준의 엄밀성에 도달할 수 있는 학습 자료를 만든다. 미분기하 교과서처럼 기초부터 순서대로 쌓아 올린다.
- **가정하는 독자**: 다변수를 포함한 미적분과 선형대수를 한 번 배운 학부생이다. 위상수학과 해석학은 몰라도 된다. 필요한 만큼은 Part 0에서 다룬다. 독자가 증명 읽기에 익숙하지 않다고 가정한다.
- **도달 목표**
  - Part I을 마치면 do Carmo의 곡선·곡면론 수준에 이른다.
  - Part II를 마치면 Lee, *Introduction to Smooth Manifolds*의 핵심 장 수준에 이른다.
  - Part III을 마치면 Lee, *Introduction to Riemannian Manifolds* 전반부(측지선, 곡률, 가우스-보네) 수준에 이른다.
- 이 자료는 교재를 대체하지 않고 **교재와 나란히 읽도록** 만든다. 그래서 각 절에 대응하는 교재 절을 명시한다.

---

## 2. 교육 원칙: "차근차근"

### 2.1 한 절의 흐름

순서는 필요하면 조정해도 되지만 항목을 빠뜨리지는 않는다.

1. **이 절의 목표**(소제목 Goals): 2–4개 항목. "이 절을 마치면 …을 할 수 있다" 형식으로 쓴다.
2. **선수 지식**(Prerequisites): 이전 절의 정의·정리 번호를 링크로 건다.
3. **동기**(Motivation): 이 개념이 왜 필요한지 쓴다. 첫 문단은 **앞 절이 끝난 지점에서** 출발한다(§3.2). 앞 절에서 해결하지 못한 질문이나 ℝ³·ℝⁿ의 익숙한 상황이 출발점이 된다.
4. **직관**: 그림을 1개 이상 넣는다. 정의보다 먼저 "그림으로 보면 무엇인지"를 보여준다.
5. **정의**: 정확하게 진술한다. 바로 뒤에 가장 간단한 **예**와 **비예(non-example)**를 둔다. 비예가 부자연스러운 정의(예: 행렬식, 쌍대사상)에는 비예 대신 **흔한 오해**를 `box warning`(제목 Common misconception)으로 둔다.
6. **정리**: 진술 → "이 정리가 말하는 것"(평이한 문장 1–2개) → 증명 아이디어(2–3문장) → 증명 순서로 쓴다.
7. **좌표 표현**: 좌표 없는 정의와 좌표 공식을 모두 준다. 좌표 공식은 **좌표를 어떻게 골라도 같은 결과가 나오는지**(well-definedness) 확인한다.
8. **계산 예제**: §7의 기준 예제로 계산한다. 어떤 계산 기법이 처음 나올 때는 중간 단계를 하나도 생략하지 않는다.
9. **주의 / 흔한 오해 / 규약 비교**(상자 제목 Warning / Common misconception / Conventions): 교재와 규약이 다를 때 넣는다.
10. **요약**(Summary): 3–6개 항목으로 쓴다. 끝에 "다음 절에서는" 한두 문장으로 다음 절의 질문을 예고한다.
11. **연습문제**(Exercises): §14를 따른다.

### 2.2 원칙

- **구체에서 추상으로.** ℝ³의 곡면 → 추상 다양체, ℝⁿ의 방향미분 → derivation 순서로 간다. 추상 정의를 먼저 던지지 않는다.
- **새 개념은 한 번에 하나만.** 한 문단에 새 정의를 둘 넣지 않는다.
- **같은 대상을 반복해서 쓴다.** 예제를 매번 새로 만들지 말고 §7 기준 예제를 다시 써서 독자가 비교할 수 있게 한다.
- **정의의 각 조건이 왜 필요한지 보여준다.** 조건 하나를 빼면 무엇이 깨지는지 비예로 보인다.
- **계산 생략 기준.** 어떤 계산 기법이 처음 나오는 곳에서는 모든 단계를 쓴다. 두 번째부터는 "(6.2.3)과 같은 방식으로"처럼 앞 계산을 참조하고 줄여도 된다.
- **앞뒤를 연결한다.** Part I의 개념이 Part II·III에서 일반화될 때는 앞에서 "Looking ahead" 상자(`box forward`)로 예고한다. 뒤에서는 "Looking back" 상자(`box backward`)에서 "Part I의 …와 비교"로 되짚는다. 구체적인 규칙은 §3에 있다.
- **과하게 일반화하지 않는다.** Part I에서 벡터다발이나 접속의 일반론을 꺼내지 않는다. 그 자리에 필요한 만큼만 다룬다.
- **분량 기준.** 한 절에는 핵심 개념 1개 안팎을 담고, 읽는 데 30–60분이 걸리게 한다. 정의 1–4개, 예 3개 이상, 그림 1개 이상, 연습 4–8개가 기준이다. 이보다 커지면 절을 나눈다.
  - Part 0의 복습 장은 기본 정의가 몰려 있어 정의가 5–7개여도 된다(01장 파일럿).
  - 절을 나누는 것은 **작성 단계에서만** 한다. 다른 장이 이미 링크한 절은 번호를 바꾸지 않는다(§15.6).

### 2.3 Chapter 0 (Tour)의 원칙 (2026-10-06 사용자 결정)

Chapter 0 「A Guided Tour of Differential Geometry」는 깊은 장(01–26) **앞에 두는 가벼운 입구**다. 깊은 장은 그대로 둔다.

- **목적**: 전체 개념을 한 바퀴 훑어 직관과 지도를 얻는 것. 관심이 생긴 주제만 Go deeper 링크로 깊은 장에 들어간다.
- **독자**: 컴퓨터비전·그래픽스를 공부하는 CS 학부생. 미적분과 행렬·벡터 계산은 알지만, 증명을 따라가지 않아도 읽을 수 있어야 한다.
- **분량**: 절 하나에 15–25분. 핵심 정의 1–3개, 그림 2개 이상. 예는 기준 예제(§7) 위주로 든다.
- **엄밀성**: 정확하되 가볍게 쓴다.
  - 단순화할 때는 "여기서는 …로 단순화한다"고 밝히고, 정확한 진술은 Go deeper로 보낸다. **틀린 단순화는 하지 않는다.**
  - 증명은 쓰지 않는다. 필요하면 "Proof idea" 한 문단으로 쓴다.
  - 정의와 정리는 상자로 두되, 가정을 줄인 형태라면 그렇다고 적는다.
- **절의 구성**
  - Goals, Prerequisites(앞 Tour 절, 고교·학부 기초), Motivation
  - 본문: 그림 중심, 핵심 정의·정리 상자
  - **In computer vision 상자**(`aside.box.application`) 1–3개: 컴퓨터비전·그래픽스·기계학습의 실제 기법과 연결한다. 연결이 정확해야 하고 과장하지 않는다(예: normal estimation, mesh curvature, SO(3)와 카메라 자세, 동차좌표와 ℝP², Poisson image editing, Riemannian optimization).
  - **Go deeper 상자**(`aside.box.deeper`): 깊은 장의 해당 절·정리를 링크한다. 아직 쓰이지 않은 장은 장 개요로 링크한다.
  - Summary(끝에 다음 절 예고 `p.next`)
  - Exercises: 제목은 "Quick check". ★ 문제 2–4개로 개념 확인 위주이며, 풀이를 단다.
- **용어 표시**: Tour에서는 `<dfn>`을 쓰지 않는다. 공식 정의는 깊은 장에 있다(§3.3 한 개념 한 정의). 처음 나올 때 `<strong>curvature</strong>(곡률)`처럼 굵게 쓰고 한국어를 괄호에 적는다. 가능하면 깊은 장의 정의로 링크한다.
- **수식**: 필요한 만큼만 쓰고 계산 과정은 짧게 한다. 기호는 §6을 따르고, 새 기호는 꼭 필요할 때만 쓴다.
- **그림**: §12를 따른다. 직관을 위한 그림을 아끼지 않는다. 3D 곡면에는 인터랙티브 그림(§12.7)을 권장한다. 정적 SVG는 항상 둔다.
- **번호**: 장 번호는 0이다(Definition 0.3.1, Figure 0.4.2 …).
- **verify**: Tour 절의 수치와 공식도 계산이 있으면 verify 스크립트로 확인한다.
- **연결**: Tour 절끼리는 §3.2대로 앞뒤를 잇는다. 깊은 장은 해당 Tour 절을 링크로 가리킬 수 있다(선택).

---

## 3. 유기적 연결과 일관성

26장은 **한 권의 책처럼** 읽혀야 한다. 여러 agent가 나눠 쓰더라도 독자가 저자가 바뀐 것을 느끼지 못해야 한다. 앞에서 배운 것은 뒤에서 다시 쓰이고, 뒤에서 배우는 것은 앞의 것을 넓히거나 다시 설명해야 한다. 이를 위한 규칙과 장치를 아래에 둔다.

### 3.1 큰 줄기 (threads)

study set 전체를 꿰는 이야기 줄기다.

- 모든 장은 적어도 하나의 줄기를 앞으로 진행시킨다.
- 장 개요(`index.html`)에는 이 장이 어느 줄기의 어느 지점에 있는지 적는다.
- 줄기와 무관한 장이 생기면 로드맵(§9)을 다시 검토할 신호이므로 최종 보고에 적는다.
- 페이지(`p.threads`)에는 줄기의 영어 이름을 쓴다(§8.2). 예: `Threads: <strong>T3</strong> Intrinsic vs. extrinsic (start)`

| 줄기 (영어 이름) | 흐름 (장 번호) |
|---|---|
| T1. 곡률이란 무엇인가 (What is curvature?) | 곡선의 κ, τ (04–05) → 법곡률·주곡률·가우스 곡률 (08) → 빼어난 정리: K는 내재적이다 (09) → 곡률텐서·단면곡률 (24) → 가우스 방정식으로 Part I과 다시 만남 (25) |
| T2. 미분과 접공간 (Differentials and tangent spaces) | ℝⁿ의 전미분 DF(p) (02) → 접평면과 dφ_p (06) → 매끄러운 사상 (12) → derivation과 dF_p (13) → 몰입·침몰 (14) → 벡터장과 흐름 (15) → (선택) 분포와 프로베니우스 정리 (20) |
| T3. 내재적 vs 외재적 (Intrinsic vs. extrinsic) | 부분공간 위상과 몫공간: 공간 안에 놓인 것과 스스로 주어진 것 (03) → 제1기본형식 (07) → 빼어난 정리 (09) → 둘러싼 공간 없는 추상 다양체 (11) → 리만 계량 (21) → 부분다양체의 기하 (25) |
| T4. 평행이동과 측지선 (Parallel transport and geodesics) | 곡면 위의 공변미분·측지선 (09) → 접속 (22) → 측지선·지수사상 (23) → 야코비 장 (26) |
| T5. 적분과 위상 (Integration and topology) | 다중적분과 변수변환 공식 (02) → 컴팩트성·연결성 (03) → 호의 길이·넓이 (04, 07) → 가우스-보네 (10) → 단위분할 (12) → 미분형식·스토크스 (17–18) → 드람 코호몰로지 (19) → 가우스-보네 재방문 (26) |
| T6. 쌍대성과 텐서 (Duality and tensors) | 쌍대공간 (01) → 여접공간·1-형식 (16) → 미분형식 (17) → 음악적 동형 (21) → 곡률텐서 (24) |

### 3.2 장과 절을 잇는 장치

- **장 개요** `index.html`에는 부록 A-2 템플릿대로 다음을 반드시 쓴다.
  - **이어받는 것**(소제목 Builds on): 앞 장들에서 가져오는 질문과 결과. 번호로 링크한다.
  - **이 장의 질문**(Questions): 이 장이 답하는 중심 질문 1–2개
  - **넘겨주는 것**(Leads to): 이 장이 남기는 질문, 그리고 뒤 장에서 쓰이는 결과와 그 장
  - **줄기 위치**(`p.threads`, Threads): §3.1의 어느 줄기, 어느 지점인지
- **절의 시작**: "동기"의 첫 문단은 앞 절(장의 첫 절이면 앞 장)이 끝난 지점에서 출발한다.
  - 예: "Section 6.3에서 surface 위의 함수가 매끄럽다는 것의 뜻을 정했다. 그렇다면 그런 함수의 '미분'은 무엇이어야 하는가?"
- **절의 끝**: 요약 다음에 "다음 절에서는"으로 다음 절의 질문을 한두 문장 예고한다(`<p class="next">`).
- **앞으로 가리킬 때(예고)**
  - **다른 장**의 개념은 `box forward` 안에서만 예고한다. 같은 장 안의 뒤 절은 일반 링크로 가리켜도 된다(예: "Section 1.4에서 다룬다").
  - 예고한 개념은 **정의나 증명에 쓰지 않는다**.
  - 예고할 때는 해당 장을 링크한다.
- **뒤로 가리킬 때(회고)**
  - 같은 대상이 다시 나오면 반드시 이전 위치를 링크한다.
  - 이전 결과는 **다시 유도하지 않고 인용한다**.
  - 새 관점에서 다시 보는 것이 목적이면 `box backward`에 "무엇이 새로워졌는가"를 쓴다.
- **같은 내용을 두 곳에 쓰지 않는다.** 복사하지 말고, 링크를 걸고 한 문장으로 요약한다.
- **Part 0의 예외 (02장 ↔ 03장)**: 02장(미적분)이 03장(위상)보다 앞에 있지만, 02장의 증명은 컴팩트성에 관한 미적분학 수준의 사실을 쓴다. 그런 사실은 "미적분학에서 알려진 사실로 받아들인다"고 밝히고 증명이 있는 3.6절의 항목에 링크한다. 해당 사실은 유계 닫힌 집합 위 연속함수의 최대·최소(정리 3.6.10), 하이네-보렐(정리 3.6.7), 균등연속(따름정리 3.6.19), 중간값 정리(따름정리 3.7.6)다. 이 예외는 이 네 사실에만 적용하며, 02장은 이것들을 2.1절 「준비」 한 곳에 모아 둔다. 03장이 다루지 않는 미적분학의 사실(ℝⁿ의 완비성 등)도 2.1절 「준비」에 "받아들이는 사실"로 모아 두고 거기에 링크한다.
- **한 문장짜리 지시문**: 뒤 장을 가리키기만 하는 한 문장("이 결과는 Chapter 8에서 다시 쓴다", "이 개념은 Chapter 13에서 다시 만난다")은 동기 문단이나 예·비고의 끝 어디에 써도 된다. 장 링크를 건다. 뒤 장의 개념을 **설명하거나 쓰려면** `box forward`로 옮긴다.

### 3.3 한 개념, 한 정의

- 각 개념은 study set 전체에서 **한 곳에서만 정의**한다. 다른 곳에서는 그 정의를 링크한다.
- 이미 정의된 개념을 다시 정의하지 않는다. 필요하면 "Definition 6.2.1을 상기하자"처럼 **상기**만 한다. 상기할 때는 `<dfn>`을 쓰지 않고 원래 위치를 링크한다.
- **일반화**(Part I의 개념을 Part II·III에서 넓힐 때)는 새 정의로 쓴다.
  1. 무엇을 일반화하는지 표시한다. 예: `<dfn data-generalizes="../06-regular-surfaces/6-2-tangent-plane.html#i-6-2-1">tangent space</dfn>(접공간)`
  2. 특수한 경우에 이전 정의와 **일치함을 명제로 증명한다**(호환성 명제, §3.4).
- **동치인 두 정의**가 있을 때도 어느 쪽이 공식 정의인지 정한다. 다른 쪽은 명제로 둔다.
  - 예: 13장에서는 derivation 정의가 공식 정의이고, 곡선의 동치류 정의는 동치임을 보이는 명제로 둔다.
- 정의를 도입할 때 **왜 이 정의가 앞의 것과 같은 이름을 가질 자격이 있는지** 설명한다. 예: 추상 접공간이 왜 "접"공간인가.

### 3.4 필수 호환성 명제

Part I의 개념이 Part II·III에서 일반화될 때, 아래 명제를 해당 장에서 **반드시 진술하고 증명한다**. 이 명제들이 세 Part를 하나로 묶는 이음매다. 증명 없이 "같다"고만 쓰지 않는다. 호환성 명제에는 `compat` 클래스를 붙인다(`<div class="thm proposition compat" id="…">`). `build_index.py`가 표의 장마다 이 항목이 있는지 검사한다.

| 장 | 호환성 명제 | 이전 위치 |
|---|---|---|
| 11 | ℝ³의 정칙곡면은 2차원 매끄러운 다양체이고, 매개화의 역사상은 매끄러운 차트다 | 06 |
| 12 | 정칙곡면 위 함수의 매끄러움(06장의 정의)은 다양체로서의 매끄러움과 같다 | 06 |
| 13 | 정칙곡면의 접평면 T_pS ⊆ ℝ³은 derivation 공간 T_pS와 자연스럽게 동형이다. 이 동형에서 dφ_p(06)는 dF_p(13)와 같다 | 06 |
| 14 | 정칙곡면은 ℝ³의 2차원 매장 부분다양체와 같은 것이다. 06장의 "정칙값의 역상" 정리는 정칙값 정리의 특수한 경우다 | 06 |
| 16 | 제1기본형식은 유클리드 계량을 포함사상으로 당긴 것이다 | 07 |
| 18 | 곡면의 넓이(07)는 넓이 형식 ι_**N**(dx∧dy∧dz)의 적분과 같다. ℝ³의 고전적인 스토크스·발산 정리는 일반 스토크스 정리의 특수한 경우다 | 07 |
| 21 | 곡면 사이의 등거리사상(07)은 유도된 리만 계량에 대한 등거리사상과 같다. 18장의 넓이 형식은 유도 계량의 리만 체적형식 dV_g와 같다 | 07, 18 |
| 22 | 곡면 위의 공변미분 D_t(09)는 유도된 레비치비타 접속의 공변미분과 같다 | 09 |
| 23 | 곡면의 측지선(09)은 유도된 계량에 대한 리만 측지선과 같다 | 09 |
| 24 | 2차원에서 단면곡률은 가우스 곡률 K(08)와 같다 | 08 |
| 25 | 형태작용소 W_N은 W_p = −d**N**_p(08)와 같고, 빼어난 정리(09)는 가우스 방정식의 특수한 경우다 | 08, 09 |
| 26 | 가우스-보네 정리(10)를 리만 기하의 언어로 다시 해석한다 | 10 |

### 3.5 기준 예제의 여정

기준 예제(§7)는 장을 거듭하며 **누적적으로** 발전한다.

- 앞 장에서 계산한 값은 다시 계산하지 않고 인용해서 이어 간다. 예: "Example 7.1.2에서 구한 E, F, G를 쓰면 …"
- 아래 표는 계획이다. 각 장은 자기 칸의 내용을 반드시 다룬다.
- 칸에 없는 내용을 추가했다면 최종 보고에 적는다(통합 담당이 표를 갱신).

| 예제 | 여정 (장 번호) |
|---|---|
| 구면 S²(r) | 편미분 벡터와 그람 행렬 (01 연습), 전미분·등위집합·부피 요소 (02) → 입체사영이 위상동형사상임 (03) → 대원·위도원의 곡률 (04), 구면 위 원의 프레네 틀 (05) → 정칙곡면임을 보임, 접평면 (06) → E, F, G, 넓이 4πr², 띠 넓이 2πrh, 입체사영·메르카토르의 등각성, 바깥쪽 𝐍 (07) → L, M, N과 K = 1/r² (08) → 크리스토펠 기호, 측지선 = 대원, 평행이동 회전각 (09) → 가우스-보네로 χ = 2 (10) → Sⁿ의 매끄러운 구조, 입체사영의 좌표변환 u/|u|², 반구 차트와 같은 구조, S¹의 각 차트 (11) → 높이함수의 입체사영 좌표표현, 두 입체사영 영역의 단위분할 (12) → 정칙값 정리로 본 Sⁿ (14) → 둥근 계량 (21) → 측지선, 지수사상, 절단점 (23) → 상수곡률 1/r² (24) |
| 원환면 | 계수 2, 입체 부피 2π²Rr² (02) → 토러스 T² = S¹ × S¹ ≅ 원환면, 정사각형의 몫공간, 연결성 (03) → 정칙값의 역상, 𝐱_u × 𝐱_v는 안쪽 (06) → E, F, G, 넓이 4π²Rr, 𝐍_𝐱는 안쪽 (07) → K의 부호 분포 (08) → ∫K dA = 0, χ = 0 (10) → 곱다양체 T² = S¹ × S¹, 각 차트 α × α와 원환면 매개화의 관계 (11) → T² ≅ T_{R,r}(미분동형), cos u의 ℝ³ 확장 (12) → 드람 코호몰로지 (19) |
| 원기둥 | 정칙곡면, 𝐱_u × 𝐱_v는 바깥쪽 (06) → E, F, G와 평면과의 국소 등거리 (07) → K = 0이지만 H ≠ 0 (08) → 측지선 = 나선 (09) |
| 나선 | γ′ × γ″ (01 연습), 속도 (02) → 속도·길이(한 바퀴 2π√(a²+b²))·호의 길이 매개화 (04) → κ, τ (05) → 원기둥 위의 측지선 (09) |
| 원 | 속도 (02) → 부호곡률 κ_s = ±1/r, 접촉원 (04) → κ = 1/r, τ = 0, 회전지수 ±1, 넓이 πr² (05) |
| 회전면 | 정칙곡면의 예 (06) → E, F, G (07) → K와 주곡률 (08) |
| 그래프 곡면 | 정칙곡면의 예 (06) → K 공식 (08) |
| 안장면 | K = −4, H = 0, 쌍곡점 (08) |
| 쌍곡평면 𝕌² | 둘러싼 공간 없이 계량만 주어진 "곡면", K = −1 (09 끝, T3 줄기) → 리만 다양체로 재정의 (21) → 측지선 (23) → 상수곡률 −1 모형공간 (24) |
| ℝℙⁿ | 몫위상, Sⁿ/± ≅ ℝℙⁿ, 차트 φ_i의 위상동형성, 하우스도르프·제2가산·컴팩트·연결 (03) → 차트와 매끄러운 구조, 좌표변환 행렬식의 부호 (11) → Sⁿ → ℝℙⁿ은 2대1 국소 미분동형 (12) → 방향 불가능성 (18) |

### 3.6 일관성 유지 장치

규칙을 기억에만 맡기지 않는다. 정본을 한 곳에 두고 도구로 검사한다.

| 대상 | 정본 위치 | 검사 방법 |
|---|---|---|
| 기호·부호 규약 | §6 | 검토자 |
| 용어 | §8, `tools/terms.tsv`, `tools/terms_keep.txt` | 검토자, `concepts.html`. 변환 작업에서는 `term_guard.py` |
| 라벨·제목 | §8.2, §9 | `check_site.py`: 영어 머리 단어·상태 라벨은 오류, 남은 한국어 라벨·제목·번호 참조는 경고 |
| 개념이 정의된 위치 | `concepts.html` (Index, 자동 생성) | `build_index.py`: 같은 영어 용어(대소문자 무시)의 `<dfn>`이 두 번 나오면 오류 (`data-generalizes` 표시는 제외). 괄호 안 한국어 번역어가 서로 다르면 동음이의어로 보고 경고만 한다(예: trace = 대각합 / 자취) |
| 기준 예제의 식과 값 | §7과 `tools/dgsym.py`의 `EXAMPLES` | verify 스크립트는 매개화를 `EXAMPLES`에서 가져온다. 식을 직접 다시 쓰지 않는다 |
| 상호참조 | 각 페이지의 링크 | `check_site.py`: 깨진 링크, 번호 불일치 |
| 역참조 ("쓰이는 곳") | `concepts.html` | `build_index.py`가 링크 그래프로 생성 |
| 문체와 밀도 | 부록 B | 검토자 |

**변경 전파**: 정의, 기호, 예제 값을 바꿨다면 `concepts.html`의 "쓰이는 곳"을 보고 영향받는 절을 **모두** 다시 확인한다. 영향받은 절의 목록은 최종 보고에 적는다.

### 3.7 연결을 위한 작성 순서

- **같은 장의 절은 순서대로 쓴다.** 앞 절이 `draft`가 된 뒤에 다음 절을 시작한다. 그래야 "앞 절이 끝난 지점"에서 출발할 수 있다.
- **병렬 작업은 서로 독립인 장끼리만 한다.** 독립 여부는 §9의 의존 관계로 판단한다. 예: Part I(04–10)과 Part II 앞부분(11–15)
- 장이 끝나면 **장 통합 검토**를, Part가 끝나면 **Part 통합 검토**를 한다(§15.6).

---

## 4. 엄밀성, 출처, 검증

### 4.1 진술

- 모든 정의와 정리에 가정을 명시한다. 확인할 항목은 다음과 같다.
  - 매끄러움의 정도. 이 study set에서 "매끄러운"은 C^∞를 뜻한다.
  - 차원, 하우스도르프·제2가산 조건, 연결성, 컴팩트성, 방향, 경계 유무
  - 곡선이 정칙인지, 호의 길이로 매개화되었는지
  - 곡면의 법벡터를 어느 쪽으로 골랐는지
- 한정사를 정확히 쓴다. "모든 p에 대하여 어떤 근방 U가 존재하여 …"처럼 쓴다. "국소적으로"라고 쓸 때는 무엇이 국소적인지 밝힌다.
- 정의에 쓰인 선택(차트, 매개화, 기저, 법벡터 방향)에 결과가 **무관함**을 확인한다. 선택에 의존한다면 의존한다고 쓴다.

### 4.2 증명

- 핵심 정리에는 완전한 증명을 쓴다. 각 단계에 근거(앞 정리의 번호나 가정)를 붙인다.
- "자명하다", "쉽게 알 수 있다", "명백하다"는 한 줄로 확인할 수 있는 경우에만 쓰고, 그 한 줄의 이유를 함께 적는다.
- 증명을 생략해도 되는 "무거운 정리"는 아래와 같다. 진술은 정확히 하고 증명은 참고문헌으로 돌린다.
  - 상미분방정식 해의 존재·유일성, 초기값에 대한 매끄러운 의존성
  - Sard 정리
  - Whitney 매장 정리
  - 곡면 분류 정리
  - 삼각분할의 존재
  - 역함수 정리: Part 0에서 증명 스케치만 허용한다.
  - Part 0의 다중적분 복습: 적분가능성 판정, 푸비니 정리, 변수변환 공식, 그린 정리. 진술 또는 증명 스케치와 참고문헌으로 둔다(02장 2.6절).
  - ODE 흐름의 정의역이 열린집합이고 흐름이 매끄럽다는 것(ODE 항목에 포함한다).
- 위 목록에 없는 증명을 생략했다면 최종 보고에 이유를 적는다.
- 증명 스케치에는 반드시 "Proof sketch."라고 표시한다(§11.3). 완전한 증명처럼 보이게 쓰지 않는다.

### 4.3 출처의 신뢰도 순서

1. 사용자가 `refs/`에 넣어 둔 교재 PDF. 있으면 가장 먼저 쓴다. Read 도구로 필요한 쪽만 읽는다.
2. 공개된 믿을 만한 자료: 저자가 공개한 교재·강의노트(예: Shifrin의 곡선·곡면 교재, Axler LADR 4판 open access), 대학 강의노트
3. 백과사전류(Wikipedia, nLab, MathWorld). **규약 차이를 확인할 때만** 쓰고, 단독 근거로 삼지 않는다.
4. 직접 유도한 결과와 그 기호·수치 검증 (§13)

기억에만 의존한 사실은 4번 방법으로 확인한 뒤에 쓴다.

### 4.4 인용 규칙 (환각 방지 — 중요)

- 교재의 정리·절·쪽 번호는 `refs/`의 PDF나 믿을 만한 목차로 **직접 확인한 경우에만** 쓴다. 확인하지 못했다면 "[Lee-ISM, Ch. 3]"처럼 장 수준까지만 적거나 `[확인 필요]`로 표시한다.
- 교재 문장을 번역해서 그대로 옮기지 않는다(저작권). 내용은 자기 말로 다시 쓴다.
- 교재 그림을 스캔하거나 캡처하거나 따라 그리지 않는다. 구도와 관례만 참고하고 새로 계산해서 그린다 (§12).
- 교재의 연습문제를 쓸 때는 **실질적으로 변형**하고 출처를 적는다. 예: "[dC 2장 연습문제 변형]". 대상·조건·묻는 것을 바꾸거나 일반화한다. 숫자만 바꾸거나 번역만 한 것은 변형이 아니다(06장 검토에서 7문항을 고쳤다).

### 4.5 확인 필요 표시

확신이 없는 곳에는 모두 `<span class="todo">[확인 필요: 이유]</span>`를 넣는다. 숨기지 않는다. 검사 도구가 이 표시의 개수를 센다.

---

## 5. 참고 교재

본문에서는 아래 약어로 인용한다. `references.html`의 각 항목 id는 약어와 같다. 예: `references.html#Lee-ISM`

| 약어 | 서지 | 역할 |
|---|---|---|
| dC | M. P. do Carmo, *Differential Geometry of Curves and Surfaces*, 2nd ed., Dover, 2016 (초판 Prentice-Hall, 1976) | Part I 주교재 |
| Pr | A. Pressley, *Elementary Differential Geometry*, 2nd ed., Springer (SUMS), 2010 | Part I 보조. 더 쉬운 설명과 계산 예 |
| ON | B. O'Neill, *Elementary Differential Geometry*, rev. 2nd ed., Academic Press, 2006 | Part I 보조. 틀(frame)과 형식 관점 |
| Sh | T. Shifrin, *Differential Geometry: A First Course in Curves and Surfaces* (저자 공개 강의노트) | Part I 보조. 공개 자료 |
| Nd | T. Needham, *Visual Differential Geometry and Forms*, Princeton UP, 2021 | 직관과 그림 참고 |
| Cr | K. Crane, *Discrete Differential Geometry: An Applied Introduction* (공개 강의노트) | 직관과 그림 참고 |
| Ax | S. Axler, *Linear Algebra Done Right*, 4th ed., Springer, 2024 (open access) | Part 0 선형대수 |
| Sp-CoM | M. Spivak, *Calculus on Manifolds*, 1965 | Part 0 해석 |
| Lee-ITM | J. M. Lee, *Introduction to Topological Manifolds*, 2nd ed., GTM 202, Springer, 2011 | Part 0 위상 |
| Lee-ISM | J. M. Lee, *Introduction to Smooth Manifolds*, 2nd ed., GTM 218, Springer, 2013 | Part II 주교재이자 **표기 기준** |
| Tu-IM | L. W. Tu, *An Introduction to Manifolds*, 2nd ed., Universitext, Springer, 2011 | Part II 보조. 더 완만한 설명 |
| Lee-IRM | J. M. Lee, *Introduction to Riemannian Manifolds*, 2nd ed., GTM 176, Springer, 2018 | Part III 주교재이자 **곡률 부호 기준** |
| dC-RG | M. P. do Carmo, *Riemannian Geometry*, Birkhäuser, 1992 | Part III 보조. 곡률 부호가 반대이므로 주의 |
| BT | R. Bott, L. W. Tu, *Differential Forms in Algebraic Topology*, GTM 82, Springer, 1982 | 드람 코호몰로지 심화 |

### 5.1 `refs/`에 있는 교재 (2026-09-30 확인)

사용자가 넣어 둔 PDF는 두 권이다. 둘 다 위 표의 판본과 일치하고, 본문이 빠짐없이 들어 있다.

| 약어 | 파일 | 판본 확인 | PDF 쪽 = 인쇄 쪽 + 오프셋 |
|---|---|---|---|
| dC | `refs/(Dover Books on Mathematics) Manfredo P. do Carmo - Differential Geometry of Curves and Surfaces-Dover Publications (2016).pdf` | Revised & Updated 2nd ed., Dover 2016, ISBN 978-0-486-80699-0. 529쪽 | 전 구간 **+16** (인쇄 1–510쪽) |
| Lee-IRM | `refs/(Graduate Texts in Mathematics) John M. Lee - Introduction to Riemannian Manifolds-Springer (2018).pdf` | 2nd ed., GTM 176, Springer 2018, ISBN 978-3-319-91754-2. 447쪽 | 인쇄 1–83쪽 **+14**, 86–113쪽 **+13**, 116–191쪽 **+12**, 194–317쪽 **+11**, 320–437쪽 **+10**. 빈 쪽 84, 114, 192, 318은 PDF에 없다 |

- **읽는 법**
  - 텍스트: `pdftotext -f P -l P -layout 파일 -` (P는 PDF 쪽 번호)
  - 쪽 이미지: Read 도구의 `pages` 인자
  - Lee-IRM에서 추출한 텍스트는 기호 글꼴이 깨진다(= → D, ∇ → r, 마이너스 누락 등). **수식은 반드시 쪽 이미지로 확인한다.** dC의 텍스트는 대체로 깨끗하다.
- **인용할 때**: 이 두 권은 절·정리·식 번호를 원문에서 확인한 뒤 정확히 인용할 수 있다(§4.4). 쪽 번호는 인쇄 쪽 번호로 쓴다.
- **나머지 교재**(Lee-ISM, Tu-IM, Pr, ON, dC-RG 등)는 `refs/`에 없다. 장 수준까지만 인용하거나 `[확인 필요]`로 표시한다.
- `refs/`의 내용은 참조만 한다. 문장, 그림, 연습문제를 복사하지 않는다(§4.4).

---

## 6. 표기법과 규약 (정본)

### 6.0 기본 원칙

- Part I(곡선·곡면)은 **dC**를 따른다. 단, 아래에서 바꾼 두 가지(비틀림률 부호, 제2기본형식 계수 문자)는 이 문서를 따른다.
- Part II(다양체)는 **Lee-ISM**, Part III(리만 기하)은 **Lee-IRM**을 따른다.
- **한 기호에는 한 의미만** 준다. 적어도 한 절 안에서는 반드시 지킨다. 충돌이 불가피하면 §6.7을 따른다.
- 이 study set의 규약이 교재와 다른 곳에는 본문에 "규약 비교" 상자를 넣는다 (§11.3).
- TeX 표기는 부록 C의 프로젝트 매크로(`\R`, `\inner`, `\abs` 등)를 쓴다.
- 글자 모양을 통일한다.
  - φ는 항상 `\varphi`로 쓴다(`\phi` 금지).
  - ε은 항상 `\varepsilon`으로 쓴다.
  - 굵은 벡터는 `\mathbf{…}`로, 굵은 그리스 문자는 `\boldsymbol{…}`로 쓴다.

### 6.1 일반

- **수 체계**: 실수 `\R`, 정수 `\Z`, 유리수 `\Q`.
- **ℝⁿ의 점**
  - Part I에서는 `(x, y, z)`로 쓴다.
  - Part 0·II·III에서는 `(x^1, \dots, x^n)`처럼 위첨자를 쓴다.
  - 단, ℝ²·ℝ³의 구체적인 예에서는 어느 Part든 `(x, y)`, `(x, y, z)`를 써도 된다. 일반 n차원 진술과 성분 계산에서는 위첨자를 쓴다.
  - 위첨자가 제곱과 헷갈리면 제곱을 괄호로 쓴다: `(x^2)^2`.
- **표준기저**: `e_1, \dots, e_n`.
- **단위행렬**: `I_n`. 크기가 분명하면 `I`도 되지만, 그 절에서 I가 구간·반전 사상 등 다른 뜻으로 쓰이면 반드시 `I_n`으로 쓴다(E는 제1기본형식 계수이므로 단위행렬로 쓰지 않는다).
- **그래프**: 함수 f의 그래프는 `\Gamma(f)`로 쓴다(Lee와 같음).
- **유클리드 내적·노름**: `\inner{v}{w}`, `\abs{v}`. 내적은 항상 이 표기로 쓰고, 점곱 `v \cdot w`는 쓰지 않는다.
- **외적(cross product)**: `v \times w`. 쐐기곱과 혼동하지 않는다.
- **부분집합**: `\subseteq`. 진부분집합은 `\subsetneq`로 쓴다. `\subset`은 쓰지 않는다.
- **사상**
  - `F\colon M \to N`, `p \mapsto F(p)`
  - 합성 `G \circ F`, 항등사상 `\id_M`, 제한 `F|_U`, 역상 `F^{-1}(c)`
- **미분**
  - ℝⁿ 사이 사상의 전미분(야코비 행렬): `DF(p)`
  - 편미분: `\pd{f}{x^i}` 또는 `\partial_i f`
  - 1변수 함수의 미분: `\gamma'(t)`
  - 점 표기 `\dot\gamma^i`는 Part III 측지선 방정식의 성분에서만 쓴다.
- **매끄러움**: "매끄러운" = `C^\infty`. `C^k`는 필요할 때만 명시한다.
- **인덱스 위치** (Part 0 1장에서 도입하고 계속 유지)
  - 벡터 성분은 위첨자 `v^i`, 여벡터 성분은 아래첨자 `\omega_i`
  - 좌표함수는 위첨자 `x^i`, 좌표벡터는 아래첨자 `\partial_i`, 쌍대기저는 위첨자 `dx^i`
  - 행렬 성분은 1장(비고 1.4.9)부터 `A^i_j`(i행 j열)로 쓴다. Part I에서는 `a_{ij}`도 허용한다.
- **아인슈타인 합 규약**
  - 13장(접공간)에서 명시적인 상자로 도입한다. 그 전(Part 0·I 포함)에는 합 기호 Σ를 직접 쓴다.
  - 도입한 뒤에도 각 장에서 처음 쓸 때 한 번 상기시킨다.
  - 위 인덱스와 아래 인덱스가 짝을 이룰 때만 더한다.

### 6.2 곡선 (Part I)

- **곡선**: `\gamma\colon I \to \R^3`. I는 열린구간이다. 일반 매개변수는 t, 호의 길이 매개변수는 s로 쓴다.
- **호의 길이**: `s(t) = \int_{t_0}^{t} \abs{\gamma'(\tau)}\,d\tau`. 곡선의 길이는 `L(\gamma)`로 쓴다.
- **프레네 틀**: 단위접벡터·주법선벡터·종법선벡터를 `\mathbf{t}, \mathbf{n}, \mathbf{b}`로 쓴다. 곡률은 `\kappa \ge 0`, 비틀림률은 `\tau`로 쓴다.
- **프레네-세레 공식** (호의 길이 매개화):
  ```
  \mathbf{t}' = \kappa\mathbf{n}, \quad \mathbf{n}' = -\kappa\mathbf{t} + \tau\mathbf{b}, \quad \mathbf{b}' = -\tau\mathbf{n}
  ```
  > ⚠ dC는 `\mathbf{b}' = \tau\mathbf{n}`으로 정의하므로 τ의 부호가 **반대**다. 이 study set은 Pr, ON, Sh와 같은 부호를 쓴다(오른손 나선의 τ > 0). 5장에서 "규약 비교" 상자로 반드시 알린다.
- **일반 매개변수 공식**:
  ```
  \kappa = \frac{\abs{\gamma' \times \gamma''}}{\abs{\gamma'}^3}, \qquad \tau = \frac{\inner{\gamma' \times \gamma''}{\gamma'''}}{\abs{\gamma' \times \gamma''}^2}
  ```
- **평면곡선의 부호곡률**: `\kappa_s`. `\mathbf{t}' = \kappa_s \mathbf{n}_s`이고, `\mathbf{n}_s`는 **t**를 반시계 방향으로 90° 돌린 벡터다.

### 6.3 곡면 (Part I)

- **정칙곡면과 매개화**
  - 정칙곡면 `S \subseteq \R^3`
  - (국소) 매개화 `\mathbf{x}\colon U \to S`. U ⊆ ℝ²는 열린집합이고 매개변수는 (u, v)다.
  - 편미분은 `\mathbf{x}_u, \mathbf{x}_v, \mathbf{x}_{uv}`로 쓴다.
  - 매개화 자체는 ℝ²의 열린집합에서 ℝ³로 가는 사상이므로 그 미분은 §6.1대로 `D\mathbf{x}(q)`로 쓴다(02장과 같음). dC는 `d\mathbf{x}_q`로 쓰므로 06장에서 규약 비교로 알린다. 곡면 **사이**의 사상의 미분만 `d\varphi_p`로 쓴다.
  - 인덱스가 필요하면 `(u^1, u^2) = (u, v)`, `\mathbf{x}_i = \partial\mathbf{x}/\partial u^i`로 쓴다.
  - ⚠ 매개화 **x**는 Part II의 차트 φ와 **방향이 반대**인 사상이다(φ = **x**⁻¹). 11장 첫 부분에서 명시적으로 연결한다.
- **접평면과 사상의 미분**
  - 접평면 `T_pS`
  - Part I에서 곡면 사이의 사상은 `\varphi, \psi\colon S_1 \to S_2`로 쓴다(dC). 미분은 `d\varphi_p\colon T_pS_1 \to T_{\varphi(p)}S_2`다.
- **단위법벡터와 가우스 사상**: `\mathbf{N} = \dfrac{\mathbf{x}_u \times \mathbf{x}_v}{\abs{\mathbf{x}_u \times \mathbf{x}_v}}`. 가우스 사상은 `\mathbf{N}\colon S \to S^2`다.
- **제1기본형식**: `\mathrm{I}_p(w) = \inner{w}{w}`. 계수는 `E, F, G`이며 각각 `g_{11}, g_{12}, g_{22}`다.
- **형태작용소**(바인가르텐 사상): `W_p = -d\mathbf{N}_p\colon T_pS \to T_pS`.
- **제2기본형식**: `\mathrm{II}_p(w) = \inner{W_p(w)}{w}`. 계수는 다음과 같고, 각각 `h_{11}, h_{12}, h_{22}`다.
  ```
  L = \inner{\mathbf{x}_{uu}}{\mathbf{N}}, \quad M = \inner{\mathbf{x}_{uv}}{\mathbf{N}}, \quad N = \inner{\mathbf{x}_{vv}}{\mathbf{N}}
  ```
  > ⚠ dC는 이 계수를 e, f, g로 쓴다. 이 study set은 g를 리만 계량 전용으로 남겨 두려고 Pr의 L, M, N을 쓴다(e = L, f = M, g = N). 계수 N(이탤릭)과 법벡터 **N**(굵은 글씨)을 구분한다.
- **여러 곡률**
  - 주곡률 `\kappa_1 \ge \kappa_2`: W_p의 고윳값
  - 주방향: 주곡률에 대응하는 고유벡터의 방향
  - 법곡률 `\kappa_n`, 측지곡률 `\kappa_g`
- **가우스 곡률과 평균곡률**:
  ```
  K = \kappa_1\kappa_2 = \det W_p = \frac{LN - M^2}{EG - F^2}, \qquad
  H = \frac{\kappa_1 + \kappa_2}{2} = \frac12\tr W_p = \frac{EN - 2FM + GL}{2(EG - F^2)}
  ```
  > ⚠ **N**을 −**N**으로 바꾸면 II, κᵢ, H의 부호가 바뀌고 K는 바뀌지 않는다. 부호가 나오는 **모든** 예제에 **N**의 방향을 명시한다.
- **넓이요소**: `dA = \sqrt{EG - F^2}\,du\,dv`.
- **가우스 공식과 크리스토펠 기호**: `\mathbf{x}_{ij} = \sum_k \Gamma^k_{ij}\mathbf{x}_k + h_{ij}\mathbf{N}`. Part I에서는 합 기호 Σ를 명시한다.
- **곡선을 따른 공변미분**: `D_t w`. dC는 Dw/dt로 쓰므로 규약 비교에 적는다.
- **오일러 지표**: `\chi(S) = n_0 - n_1 + n_2`. n₀, n₁, n₂는 각각 꼭짓점·모서리·면의 개수다. 흔히 쓰는 V − E + F는 제1기본형식의 E, F와 충돌하므로 쓰지 않는다.

### 6.4 다양체 (Part II, Lee-ISM)

- **다양체**: M, N으로 쓰고 n = dim M이다. 필요하면 차원을 `M^n`처럼 적는다.
- **차트와 아틀라스**
  - 차트 `(U, \varphi)`, `\varphi\colon U \to \hat U = \varphi(U) \subseteq \R^n`
  - 좌표함수는 `\varphi = (x^1, \dots, x^n)`이다.
  - 좌표변환 `\psi \circ \varphi^{-1}`, 아틀라스 `\mathcal{A}`
- **매끄러운 사상**: `F\colon M \to N`. 좌표표현은 `\hat F = \psi \circ F \circ \varphi^{-1}`다. `C^\infty(M)`은 M 위의 매끄러운 실함수 전체다.
- **상반공간**: `\mathbb{H}^n = \{x \in \R^n : x^n \ge 0\}`. 경계를 갖는 다양체의 모형으로만 쓰고, 쌍곡공간(§6.5)과 혼동하지 않는다.
- **접공간과 미분**
  - 접공간 `T_pM`은 p에서의 derivation 공간으로 정의한다.
  - 좌표기저 `\left.\pd{}{x^i}\right|_p`, 줄여서 `\partial_i|_p`
  - 미분 `dF_p\colon T_pM \to T_{F(p)}N`, `dF_p(v)f = v(f \circ F)`
  - ⚠ Tu는 `F_{*,p}`로 쓴다. 규약 비교에 언급한다.
- **접다발**: `TM`, 사영 `\pi\colon TM \to M`.
- **곡선의 속도**: `\gamma'(t) = d\gamma_t\big(\left.\tfrac{d}{dt}\right|_t\big)`.
- **여접공간과 여접다발**
  - `T_p^*M`, `T^*M`, 쌍대기저 `dx^i|_p`
  - 함수의 미분 `df`, `df_p(v) = vf`
- **벡터장과 리 괄호**
  - 벡터장 전체 `\mathfrak{X}(M)`. 좌표로는 `X = X^i\partial_i`, 함수에 대한 작용은 `Xf`
  - 리 괄호: `[X,Y]f = X(Yf) - Y(Xf)`
- **흐름과 리 미분**
  - 흐름 `\theta\colon \mathcal{D} \to M`, `\theta_t(p) = \theta(t,p)`, 적분곡선 `\theta^{(p)}(t)`
  - 리 미분 `\Lie_X`
- **텐서**
  - (k,l)형 텐서는 반변 k개, 공변 l개다: `T^{(k,l)}(V) = V^{\otimes k} \otimes (V^*)^{\otimes l}`. ⚠ 일부 문헌은 (k,l)의 순서가 반대다.
  - 공변 k-텐서장 공간은 `\mathcal{T}^k(M)`로 쓴다.
  - 대칭곱: `\omega\eta = \tfrac12(\omega\otimes\eta + \eta\otimes\omega)`. 따라서 `(dx)^2 = dx \otimes dx`다.
- **미분형식과 쐐기곱** (결정식 규약)
  - 미분형식 `\Omega^k(M)`, 교대 k-텐서 `\Lambda^k(V^*)`
  - 쐐기곱은 다음과 같이 정의하며, 따라서 아래 두 번째 식이 성립한다.
    ```
    \omega\wedge\eta = \frac{(k+l)!}{k!\,l!}\Alt(\omega\otimes\eta), \qquad
    dx^{i_1}\wedge\cdots\wedge dx^{i_k}(v_1,\dots,v_k) = \det\big(dx^{i_a}(v_b)\big)
    ```
  - 이 규약에서는 `d\omega(X,Y) = X(\omega(Y)) - Y(\omega(X)) - \omega([X,Y])`가 성립한다.
  - ⚠ 정규화가 다른 문헌(예: Kobayashi–Nomizu의 Alt 규약)과는 상수배가 다르다.
- **형식 연산**
  - 외미분 `d`, 당김 `F^*`, 내부곱 `\iota_X`. Lee는 내부곱을 `X \lrcorner \omega`로 쓴다.
  - 카르탕 공식: `\Lie_X = d\circ\iota_X + \iota_X\circ d`
- **방향과 스토크스 정리**
  - 경계는 `\partial M`으로 쓴다.
  - 스토크스 정리: `\int_M d\omega = \int_{\partial M}\omega`. ∂M에는 바깥쪽 법벡터를 첫 번째로 두는 유도 방향(Stokes orientation)을 준다.
- **드람 코호몰로지**: `H^k_{\mathrm{dR}}(M)`.
- **리 군과 리 대수**: G, `\mathfrak{g}`, `\GL(n,\R)`.
- **북극**: `N = (0,\dots,0,1)`. 입체사영에서 쓰며, 법벡터가 없는 Part 0(3장)과 Part II에서만 쓴다. Part I의 곡면 절에서는 쓰지 않는다.

### 6.5 리만 기하 (Part III, Lee-IRM)

- **리만 계량**
  - 계량 g, 내적 `\inner{v}{w}_g`, 노름 `\abs{v}_g`
  - 좌표로 `g = g_{ij}\,dx^i dx^j`, 역행렬의 성분 `g^{ij}`
- **표준 계량**
  - 유클리드 계량 `\bar g`
  - 반지름 R인 구면의 둥근 계량 `\mathring g_R`. R = 1이면 `\mathring g`
  - 쌍곡 계량 `\breve g`
- **쌍곡공간의 모형**
  - 상반공간 모형 `\mathbb{U}^n = \{x \in \R^n : x^n > 0\}`, `\breve g = \dfrac{(dx^1)^2 + \cdots + (dx^n)^2}{(x^n)^2}`
  - 푸앵카레 공 모형 `\mathbb{B}^n`
  - Part I의 쌍곡평면도 `\mathbb{U}^2`로 쓴다. `\mathbb{H}^n`은 §6.4의 상반공간 전용이다.
- **음악적 동형과 미분 연산자**
  - 음악적 동형 `\flat`, `\sharp`
  - 기울기 `\grad f = (df)^\sharp`, 발산 `\divg X`
  - 라플라시안 `\Delta u = \divg(\grad u)`. ℝⁿ에서는 Σ ∂²u/∂(xⁱ)²다. ⚠ 호지 라플라시안 dd* + d*d(함수에서 −Δ)와 부호가 반대이므로 쓸 때 명시한다.
  - 헤세 `\Hess u = \nabla^2 u`
- **체적형식**: `dV_g = \sqrt{\det(g_{ij})}\,dx^1\wedge\cdots\wedge dx^n`. 방향이 양인 차트에서 쓴다.
- **접속과 크리스토펠 기호**
  - 접속 ∇, 크리스토펠 기호 `\nabla_{\partial_i}\partial_j = \Gamma^k_{ij}\partial_k`
  - 레비치비타 접속: `\Gamma^k_{ij} = \tfrac12 g^{kl}(\partial_i g_{jl} + \partial_j g_{il} - \partial_l g_{ij})`
  - 비틀림: `T(X,Y) = \nabla_XY - \nabla_YX - [X,Y]`
- **곡선과 측지선**
  - 곡선을 따른 공변미분 `D_t`, 평행이동 `P^{\gamma}_{t_0t_1}`
  - 측지선 방정식 `\ddot\gamma^k + \Gamma^k_{ij}\dot\gamma^i\dot\gamma^j = 0`
  - 지수사상 `\exp_p`, 단사반지름 `\inj(p)`
  - 리만 거리 `d_g`, 길이 `L_g(\gamma)`
- **곡률** (Lee-IRM 규약)
  - 곡률 연산자: `R(X,Y)Z = \nabla_X\nabla_YZ - \nabla_Y\nabla_XZ - \nabla_{[X,Y]}Z`
  - 곡률텐서: `\Rm(X,Y,Z,W) = \inner{R(X,Y)Z}{W}_g`
  - 성분: `R(\partial_i,\partial_j)\partial_k = R_{ijk}{}^{l}\partial_l`, `R_{ijkl} = g_{lm}R_{ijk}{}^{m}`
  - 단면곡률: `K(v,w) = \dfrac{\Rm_p(v,w,w,v)}{\abs{v}_g^2\abs{w}_g^2 - \inner{v}{w}_g^2}`. Lee-IRM은 같은 값을 `\sec(v,w)`로 쓴다(식 (8.28)). 이 study set은 2차원에서 가우스 곡률과 같다는 점을 살려 K를 쓴다(§6.7).
  - 리치 곡률: `\Rc(Y,Z) = \tr\big(X \mapsto R(X,Y)Z\big)`, 성분 `R_{jk} = R_{ijk}{}^{i}`
  - 스칼라곡률: `S = g^{jk}R_{jk}`. R은 곡률 연산자 전용이므로 스칼라곡률은 S로 쓴다.
  - **자기점검**: 상수곡률 c인 공간에서는 `R(X,Y)Z = c(\inner{Y}{Z}X - \inner{X}{Z}Y)`다. 반지름 R인 Sⁿ에서는 K = 1/R², Rc = ((n−1)/R²) g, S = n(n−1)/R²다.
  - ⚠ dC-RG는 R의 부호가 **반대**다: `R(X,Y)Z = \nabla_Y\nabla_XZ - \nabla_X\nabla_YZ + \nabla_{[X,Y]}Z`. 대신 단면곡률을 ⟨R(x,y)x, y⟩로 정의하므로 **단면곡률 값은 같다**. dC-RG의 성분 공식을 옮길 때 특히 주의한다.
- **야코비 방정식**: `D_t^2 J + R(J,\gamma')\gamma' = 0`.
- **리만 부분다양체**
  - 둘러싼 다양체 쪽의 대상에는 **물결표**를 붙인다: `(\tilde M, \tilde g)`, `\tilde\nabla`, `\widetilde{\Rm}`. Lee-IRM과 같다. 막대(`\bar g`)는 유클리드 계량 전용이다.
  - 제2기본형식 `\mathrm{II}(X,Y) = (\tilde\nabla_XY)^\perp`
  - 형태작용소 `W_N X = -(\tilde\nabla_X N)^\top`. Part I의 `W_p = -d\mathbf{N}_p`를 일반화한 것이다. Lee-IRM은 형태작용소를 `s_N`으로 쓴다.
  - 바인가르텐 방정식 `\inner{W_N X}{Y} = \inner{\mathrm{II}(X,Y)}{N}`
  - 가우스 방정식은 아래와 같다. ℝ³ 안의 곡면에서 이 식이 K = (LN − M²)/(EG − F²)가 됨을 25장에서 확인한다.
    ```
    \Rm(W,X,Y,Z) = \widetilde{\Rm}(W,X,Y,Z) + \inner{\mathrm{II}(W,Z)}{\mathrm{II}(X,Y)} - \inner{\mathrm{II}(W,Y)}{\mathrm{II}(X,Z)}
    ```
    이것은 Lee-IRM 정리 8.5와 같은 식을 이항한 것이다(원문 대조 완료).

### 6.6 교재별 규약 차이 요약

"원문 대조" 열은 `refs/`의 원문으로 확인했는지를 뜻한다. ✓는 dC 또는 Lee-IRM 원문에서 확인한 것이고, "—"는 해당 교재가 없어 기억에 근거한 것이다. "—"인 항목을 본문에 쓸 때는 교재 이름을 빼고 "일부 교재는"이라고 쓰거나 `[확인 필요]`를 붙인다.

| 항목 | 이 study set | 다른 교재 | 원문 대조 |
|---|---|---|---|
| 비틀림률 | b′ = −τn | dC: b′ = τn (부호 반대, dC §1-5). dC도 §1-6 비고에서 b′ = −τn 규약이 흔하다고 적는다 | ✓ (dC) |
| 곡선·곡률·외적 표기 | γ, κ, v × w, ⟨v, w⟩ | dC: 곡선 α, 곡률 k(법곡률 k_n, 측지곡률 k_g), 외적 u ∧ v, 1장의 내적 u · v | ✓ (dC) |
| 제2기본형식 계수 | L, M, N | dC: e, f, g (dC §3-3) | ✓ (dC) |
| 형태작용소 | W_p = −dN_p, W_N | dC는 dN_p 자체를 쓰고 II_p(v) = ⟨−dN_p(v), v⟩, H = −½ tr dN_p = (k₁+k₂)/2 (같은 값). Lee-IRM은 s_N | ✓ (dC, Lee-IRM) |
| 공변미분 (곡선 따라) | D_t | dC: Dw/dt. Lee-IRM: D_t | ✓ (dC, Lee-IRM) |
| 미분 | Part I: dφ_p (dC와 같음). Part II 이후: dF_p (Lee와 같음) | Tu: F_{*,p} | — |
| (k,l)형 텐서 | 반변 k, 공변 l | 일부 문헌은 반대 | — |
| 쐐기곱 정규화 | 결정식 규약 (Lee-ISM) | Kobayashi–Nomizu 등은 상수배가 다름 | — |
| 곡률 R | ∇_X∇_Y − ∇_Y∇_X − ∇_[X,Y] (Lee-IRM과 같음. 성분식은 Lee-IRM 식 (7.4)) | dC-RG: 부호 반대 (단면곡률 값은 같음) | ✓ (Lee-IRM). dC-RG는 — |
| 라플라시안 | Δu = div(grad u) (Lee-IRM 2판 식 (2.20)과 같음) | ⚠ **Lee-ISM과 Lee-IRM 1판은 Δu = −div(grad u)**로 부호가 반대다(Lee-IRM 2판이 직접 밝힘). 호지 라플라시안 dd*+d*d는 함수에서 −Δ | ✓ (Lee-IRM) |
| 오일러 지표 | n₀ − n₁ + n₂ | 흔히 V − E + F | — |
| 단면곡률 표기 | K(v, w) | Lee-IRM: sec(v, w) (식 (8.28), 같은 값) | ✓ (Lee-IRM) |
| 둘러싼 다양체 | 물결표: M̃, g̃, ∇̃ | Lee-IRM과 같음 | ✓ (Lee-IRM) |
| 리치 곡률 | Rc(Y, Z) = tr(X ↦ R(X, Y)Z), R_jk = R_ijk^i | Lee-IRM: Rc(X, Y) = tr(Z ↦ R(Z, X)Y), R_ij = R_kij^k (문자만 다르고 같은 식) | ✓ (Lee-IRM) |
| 야코비 방정식 | D_t²J + R(J, γ′)γ′ = 0 | Lee-IRM 식 (10.1)과 같음 | ✓ (Lee-IRM) |

### 6.7 기호 충돌 규칙

알려진 충돌과 해결법은 다음과 같다.

- **N**: 제2기본형식 계수(이탤릭 N), 단위법벡터(굵은 **N**), 다양체 N(Part II), 북극 N(Part 0의 3장과 Part II)이 겹친다. 곡면을 다루는 절에서는 다양체와 북극을 N이라 부르지 않는다.
- **F**: 제1기본형식 계수 F와 사상 F가 겹친다. 그래서 Part I의 사상은 φ, ψ로 쓰고, 사상을 F, G로 쓰는 것은 Part II부터다.
- **φ**: 구면의 경도 φ와 사상·차트 φ가 겹친다. 일반 정의에서는 사상을 φ로 쓰고, 경도 φ가 함께 나오는 **한 항목(예·그림·연습·증명) 안에서만** 사상을 ψ로 부른다.
- **S**: 곡면 S, 스칼라곡률 S, 남극 S = −N(11장, Lee 관례)이 겹친다. Part III에서는 곡면도 (M, g)로 부른다. 남극 S는 Part II(11–20장)에서만 쓰고, 정칙곡면 S가 같은 절에 나오면 남극을 −N으로 쓴다.
- **L**: 길이 L(γ)와 계수 L이 겹친다. 인자가 있는지로 구분한다. 한 식에 둘 다 나오면 문장으로 구분해 준다.
- **R**: 곡률 R과 반지름 R이 겹친다. Part I의 반지름은 r이고, 원환면만 (R, r)을 쓴다. Part III에서 둘이 한 식에 나오면 반지름을 r로 바꾼다.
- **K**: 가우스 곡률과 단면곡률에 같은 문자를 쓴다. 2차원에서 둘이 같은 값이므로 **의도적으로** 같게 둔다.
- **H**: 평균곡률 H, 상반공간 ℍⁿ, 코호몰로지 Hᵏ는 서체로 구분한다.
- **E**: 계수 E와 벡터다발 E가 겹친다. 벡터다발은 Part II 이후에만 나온다.
- **∂**: 위상적 경계 ∂A(2.1절 준비, 명제 3.1.27)와 경계를 갖는 다양체의 경계 ∂M(11.5절, 18장)은 일반적으로 다르다(예: 열린 원판). 두 뜻이 한 절에 나오면 말로 구분한다.
- **≅**: 선형 동형(1장), 위상동형(3장), 미분동형(12장 이후)에 모두 쓴다. 한 절 안에서는 한 뜻으로만 쓰고, 처음 쓸 때 어느 뜻인지 밝힌다.

새 충돌을 발견하면 해당 절 안에서 해결하고, 결정한 내용을 최종 보고에 적는다(통합 담당이 notation.html에 반영).

---

## 7. 기준 예제 (running examples)

아래 매개화와 값을 **모든 장에서 그대로** 쓴다. 값은 2026-09-30에 sympy로 확인했다.

- **원** (반지름 r): γ(t) = (r cos t, r sin t, 0). κ = 1/r, τ = 0.
- **나선** (a > 0, b ≠ 0): γ(t) = (a cos t, a sin t, bt).
  - κ = a/(a² + b²), τ = b/(a² + b²)
  - b > 0이면 오른손 나선이고 τ > 0이다.
- **구면 S²(r)**: **x**(θ, φ) = (r sin θ cos φ, r sin θ sin φ, r cos θ). θ ∈ (0, π)는 여위도(colatitude), φ ∈ (0, 2π)는 경도다.
  - E = r², F = 0, G = r² sin²θ
  - **N** = **x**/r (바깥쪽)일 때 L = −r, M = 0, N = −r sin²θ
  - 따라서 κ₁ = κ₂ = −1/r, K = 1/r², H = −1/r
  - 크리스토펠 기호: Γ^θ_{φφ} = −sin θ cos θ, Γ^φ_{θφ} = Γ^φ_{φθ} = cot θ, 나머지는 0
- **원기둥** (반지름 r): **x**(u, v) = (r cos u, r sin u, v).
  - E = r², F = 0, G = 1
  - **N** 바깥쪽일 때 L = −r, M = N = 0
  - 따라서 K = 0, H = −1/(2r)
- **원환면** (0 < r < R): **x**(u, v) = ((R + r cos u) cos v, (R + r cos u) sin v, r sin u).
  - E = r², F = 0, G = (R + r cos u)²
  - K = cos u / (r(R + r cos u)). 바깥쪽 적도(u = 0)에서 K > 0, 안쪽 적도(u = π)에서 K < 0이다.
  - ⚠ 이 매개화에서 **x**_u × **x**_v는 **안쪽**(튜브의 중심원 쪽)을 향한다. L, M, N과 H를 줄 때 이 방향을 명시한다.
  - 원환면을 ℝ³의 점집합으로 부를 때는 `T_{R,r}`로 쓴다(06장). 토러스 T²(곱공간)와 구별한다.
- **회전면** (ρ > 0): **x**(u, v) = (ρ(u) cos v, ρ(u) sin v, z(u)).
  - E = ρ′² + z′², F = 0, G = ρ²
- **그래프 곡면**: **x**(u, v) = (u, v, f(u, v)).
  - K = (f_uu f_vv − f_uv²)/(1 + f_u² + f_v²)²
- **안장면**: z = x² − y², 즉 그래프 곡면에서 f = u² − v².
  - 원점에서 K = −4, H = 0
- **쌍곡평면** (상반평면 모형): 𝕌² = {(x, y) : y > 0}, g = (dx² + dy²)/y².
  - Part I(9장)에서는 제1기본형식의 언어로 E = G = 1/y², F = 0이라 쓴다. 기호 `\breve g`는 21장에서 도입한다.
  - K ≡ −1
  - 크리스토펠 기호: Γ^x_{xy} = Γ^x_{yx} = −1/y, Γ^y_{xx} = 1/y, Γ^y_{yy} = −1/y, 나머지는 0
- **입체사영**: σ: Sⁿ ∖ {N} → ℝⁿ, σ(x¹, …, xⁿ⁺¹) = (x¹, …, xⁿ)/(1 − xⁿ⁺¹). 여기서 N = (0, …, 0, 1)이다.
  - 남극에서의 입체사영은 σ̃(x) = −σ(−x) = (x¹, …, xⁿ)/(1 + xⁿ⁺¹)다.
- **실사영공간 ℝℙⁿ**: 동차좌표 [x¹ : ⋯ : xⁿ⁺¹]를 쓴다.
  - 차트 U_i = {xⁱ ≠ 0}, φ_i[x] = (x¹/xⁱ, …, x̂ⁱ, …, xⁿ⁺¹/xⁱ). 모자 표시는 그 성분을 뺀다는 뜻이다.
- **토러스**: Tⁿ = S¹ × ⋯ × S¹. T²는 위의 원환면과 미분동형이다.
  - **이름 구분**: 곱공간 S¹ × S¹은 "토러스 T²", 위 매개화로 주어지는 ℝ³의 곡면은 "원환면"이라 부른다. "회전 원환면"이라고 하지 않는다.
- **평행이동 검증 예** (S²(1)): 여위도 θ₀인 위도원을 따라 한 바퀴 평행이동하면 접벡터가 2π cos θ₀ (mod 2π)만큼 회전한다.
  - 가우스-보네와 비교하면, 위도원이 둘러싼 극관의 넓이 × K = 2π(1 − cos θ₀)다.

---

## 8. 문체, 용어, 라벨

**2026-10-04 사용자 결정**: 본문 문장은 한국어로 쓰고, **대학 수학 용어와 라벨·제목은 영어**로 쓴다.

### 8.1 문체와 용어

- 본문은 평서문 "~이다/~한다"로 쓴다. 독자에게 말을 거는 표현은 줄인다. "~해 보자"는 동기와 계산 안내에서만 허용한다.
- **대학 수학 용어는 영어로 쓴다.** 예: dual space, kernel, image, basis, linear map, tangent space, curvature, regular surface, diffeomorphism, manifold, chart.
- **고등학교 수학 수준의 말은 한국어로 둔다.** 목록은 `tools/terms_keep.txt`다. 예: 함수, 집합, 행렬, 벡터, 점, 직선, 평면, 원, 곡선, 길이, 넓이, 극한, 연속, 미분(하다), 적분, 접선, 속도.
  - **차원**: "n차원"처럼 숫자·기호와 붙여 쓸 때는 한국어로 쓰고("\(n\)차원 manifold"), 형용사로는 "n-dimensional"도 된다. "2dimension"처럼 붙여 쓰지 않는다. 명사 "차원" 자체는 dimension이다("dimension이 n이다").
  - **이름 붙은 정리·공식·방법은 고등학교에서 배운 것이라도 영어로 쓴다**: mean value theorem, intermediate value theorem, Rolle's theorem, extreme value theorem, Taylor's theorem, Cramer's rule, Gaussian elimination.
  - 같은 낱말이 두 뜻으로 쓰이면 뜻에 따른다. "미분한다"는 한국어지만, 사상으로서의 미분 dF_p는 differential이다. 직선의 "기울기"는 한국어지만, grad f는 gradient다.
- **영어 표기의 정본은 `tools/terms.tsv`다**(한국어 → 영어). 표에 없는 용어는 Lee·do Carmo의 표준 영어를 쓰고 최종 보고에 적는다. 통합 담당이 표와 glossary.html에 추가한다.
- 영어 용어는 소문자로 쓴다. 고유명사가 들어간 용어는 그 부분만 대문자로 쓴다: Gauss map, Lie bracket, Riemannian metric, Frenet frame, Theorema Egregium.
- 한국어 문장 안에서는 영어 명사를 단수형으로 쓴다("두 tangent vector", "모든 chart"). "-s"를 붙이지 않는다. 고정된 이름은 예외다: Christoffel symbols, normal coordinates.
- 한국어 수식어와 영어 용어를 한 용어 안에 섞지 않는다. "매끄러운 map"이 아니라 "smooth map"으로 쓴다. 문장 차원의 수식은 괜찮다("이 map은 매끄럽다").
- **처음 정의할 때**: `<dfn>dual space</dfn>(쌍대공간)`처럼 영어를 `<dfn>`에 넣고 한국어 번역어를 괄호에 한 번 적는다. 그 뒤로는 영어만 쓴다.
- **영어 용어 뒤 조사**는 영어 발음의 끝소리에 맞춘다.
  - 예: kernel**은**(커널), image**는**(이미지), dual space**는**(스페이스), basis**는**(베이시스), manifold**는**(매니폴드), chart**는**(차트), map**은**(맵), diffeomorphism**은**(-즘), curvature**는**(커버처), torsion**은**(토션), atlas**는**(아틀라스), Lie bracket**은**(브래킷), Gauss map**은**(맵)
  - 숫자는 한국어로 읽는다: Theorem 6.2.4**는**(사), Chapter 3**은**(삼), Section 1.3**에서**.
- **기호 뒤 조사**는 기호를 읽는 소리(영어 알파벳 또는 그리스 문자 이름)의 받침에 맞춘다.
  - 예: \(M\)은(엠), \(S\)는(에스), \(p\)는(피), \(L\)은(엘), \(n\)은(엔), \(R\)은(알), \(\gamma\)는(감마), \(\theta\)는(세타), \(\nabla\)는(나블라)
  - 첨자는 첨자 이름으로 읽는다(\(x^2\)는 "엑스 이", \(A^i_j\)는 "에이 아이 제이"). 거듭제곱은 "제곱"으로 읽는다.
  - 헷갈리면 "curve \(\gamma\)는"처럼 명사를 앞에 둔다.
- 조사는 수식과 영어 단어에 붙여 쓴다. 예: `\(M\)의`, `tangent space의`.

### 8.2 라벨과 제목 (영어)

| 대상 | 쓰는 말 |
|---|---|
| 번호 항목 (`.thm-head`) | Definition, Theorem, Proposition, Lemma, Corollary, Example, Non-example, Remark + `N.M.K` |
| 연습·그림 | Exercise N.M.K, Figure N.M.K. |
| 증명 | Proof. / Proof sketch. / Proof idea: |
| 풀이 접기 (`summary`) | Hint, Solution |
| 식 참조 | 단어 없이 "(N.M.K)". 문장 첫머리에서만 "Equation (N.M.K)" |
| 장·절 참조 | Chapter N, Section N.M ("N장", "N.M절"이라 쓰지 않는다) |
| 상자 제목 (`.box-title`) | Intuition, Warning, Common misconception, Conventions, Looking ahead, Looking back, History. 필요하면 뒤에 짧은 한국어 부제를 붙인다: "Intuition: 같은 대상, 다른 눈" |
| 절 구조 제목 | Goals, Prerequisites, Motivation, Summary, Exercises. 그 밖의 소제목도 영어로 쓴다 |
| 다음 절 예고 (`p.next`) | 본문 문장이므로 한국어로 쓴다: "다음 절에서는 …" |
| 장 개요 | Questions, Builds on, Sections, Leads to, Threads |
| 네비게이션 | breadcrumb: "Contents › 1. Linear Algebra Review › 1.3 Dual Spaces and Dual Bases". 개요 링크: "Chapter overview". pager: "← 1.2 Linear Maps" / "1.4 Change of Basis →" |
| 상태 표시 | Planned, Draft, Reviewed, Final |
| 메타 줄 | "References: [dC §2-5], …" |
| 공용 페이지 | Contents (index), Notation and Conventions, Glossary, References, Index (concepts) |
| 사이트 이름 | Differential Geometry Study (`<title>` 끝: "— DG Study") |

- **장·절 제목은 전부 영어**로 쓴다(Title Case). 장 제목의 정본은 §9의 표다. 절 제목은 장 개요의 toc가 정본이다.
- 그림 캡션과 alt는 본문처럼 한국어 문장에 영어 용어를 쓴다. 캡션의 머리만 "Figure N.M.K."다.
- 그림 **안**의 라벨은 지금처럼 수식 기호만 쓴다(§12.3).

### 8.3 용어표

- 정본은 `tools/terms.tsv`(한국어 → 영어, 비고)와 `tools/terms_keep.txt`(한국어로 두는 말)다.
- 독자용 판은 glossary.html이다(영어 → 한국어, 처음 나오는 곳).
- 혼동하기 쉬운 것
  - torus는 곱공간 T² = S¹ × S¹이고, torus of revolution은 ℝ³의 곡면 T_{R,r}이다(§7).
  - "상"(image)과 "핵"(kernel)은 다른 낱말 속의 같은 글자(이상, 대상, 상수, 핵심)와 혼동하지 않는다.
  - rank와 coefficient는 둘 다 예전에 "계수"라 불렀다. 이제 영어로 구분된다.
- 새 용어는 최종 보고에 적는다. 통합 담당이 terms.tsv와 glossary.html에 추가한다.

## 9. 로드맵

대응 교재 번호는 판본에 따라 다를 수 있다. 작성할 때 `refs/`로 다시 확인한다(§4.4). 표의 "(선택)" 장은 핵심 흐름이 끝난 뒤에 작성한다.

**장 제목과 폴더 (정본)**: 모든 페이지의 제목, 링크, breadcrumb는 아래 이름을 그대로 쓴다.

| Part | 장 | 제목 (English) | 폴더 |
|---|---|---|---|
| Tour | 00 | A Guided Tour of Differential Geometry | `00-tour` |
| 0. Preliminaries | 01 | Linear Algebra Review | `01-linear-algebra` |
| | 02 | Calculus on ℝⁿ | `02-calculus-rn` |
| | 03 | Basic Topology | `03-topology-basics` |
| I. Curves and Surfaces | 04 | Curves | `04-curves` |
| | 05 | Space Curves and the Frenet Frame | `05-space-curves` |
| | 06 | Regular Surfaces | `06-regular-surfaces` |
| | 07 | The First Fundamental Form | `07-first-fundamental-form` |
| | 08 | The Gauss Map and the Second Fundamental Form | `08-gauss-map` |
| | 09 | Intrinsic Geometry of Surfaces | `09-intrinsic-geometry` |
| | 10 | The Gauss–Bonnet Theorem | `10-gauss-bonnet` |
| II. Smooth Manifolds | 11 | Smooth Manifolds | `11-smooth-manifolds` |
| | 12 | Smooth Maps and Partitions of Unity | `12-smooth-maps` |
| | 13 | Tangent Spaces | `13-tangent-spaces` |
| | 14 | Immersions, Submersions, and Submanifolds | `14-submanifolds` |
| | 15 | Vector Fields and Flows | `15-vector-fields-flows` |
| | 16 | Covectors and Tensors | `16-covectors-tensors` |
| | 17 | Differential Forms | `17-differential-forms` |
| | 18 | Orientation, Integration, and Stokes's Theorem | `18-integration-stokes` |
| | 19 | Introduction to de Rham Cohomology | `19-de-rham` |
| | 20 | Lie Groups and the Frobenius Theorem (optional) | `20-lie-groups-frobenius` |
| III. Riemannian Geometry | 21 | Riemannian Metrics | `21-riemannian-metrics` |
| | 22 | Connections and the Levi-Civita Connection | `22-connections` |
| | 23 | Geodesics and Distance | `23-geodesics` |
| | 24 | Curvature | `24-curvature` |
| | 25 | Riemannian Submanifolds | `25-submanifold-geometry` |
| | 26 | Gauss–Bonnet Revisited and Jacobi Fields (optional) | `26-gauss-bonnet-jacobi` |

**Part 0. 준비**

| 장 | 폴더 | 핵심 내용 | 대응 교재 | 대표 그림 |
|---|---|---|---|---|
| 01 | `01-linear-algebra` | 벡터공간, 기저, 선형사상. 쌍대공간과 쌍대기저. 기저변환과 성분변환(반변·공변의 기원). 내적과 그람 행렬. 행렬식과 방향. 외적 | Ax. Lee-ISM 부록 B | 기저변환에 따른 성분 변화, 방향(오른손/왼손) |
| 02 | `02-calculus-rn` | 선형근사로서의 전미분, 야코비 행렬, 연쇄법칙. 역함수·음함수 정리. ℝⁿ에서의 계수정리. 변수변환 공식. ODE 존재·유일성(진술) | Sp-CoM. Lee-ISM 부록 C·D | 선형근사, 음함수 정리(등위곡선) |
| 03 | `03-topology-basics` | 열린집합, 연속, 위상동형. 부분공간·곱·몫 위상. 하우스도르프, 제2가산. 컴팩트, 연결 | Lee-ITM 2–4장. Lee-ISM 부록 A | 몫공간(정사각형 → 원환면) |

**Part I. 곡선과 곡면**

| 장 | 폴더 | 핵심 내용 | 대응 교재 | 대표 그림 |
|---|---|---|---|---|
| 04 | `04-curves` | 매개곡선, 정칙곡선, 호의 길이, 재매개화. 평면곡선의 부호곡률 | dC §1-2–1-3, 1-5 일부. Pr 1–2장 | 속도벡터, 접촉원 |
| 05 | `05-space-curves` | 곡률, 비틀림률, 프레네-세레 공식. 곡선론의 기본정리. 국소 표준형. 회전지수와 접선회전정리(10장의 가우스-보네에 필요). (선택) 등주부등식 | dC §1-5–1-7 | 나선의 프레네 틀, 접촉평면 |
| 06 | `06-regular-surfaces` | 정칙곡면의 정의와 예. 정칙값의 역상. 매개변수 변환. 곡면 위의 매끄러운 함수. 접평면, 사상의 미분 | dC §2-2–2-4 | 매개화 U → S, 접평면 |
| 07 | `07-first-fundamental-form` | 길이·각·넓이, 등거리사상, 등각사상, 방향 | dC §2-5, 2-6, 4-2 | 좌표곡선 격자와 각, 원기둥 펼치기 |
| 08 | `08-gauss-map` | 가우스 사상, 형태작용소. 법곡률과 뫼니에 정리. 주곡률·주방향. 가우스·평균곡률의 국소좌표 공식. 점의 분류 | dC §3-2–3-3 | 법단면, 원환면의 K 부호 색칠 |
| 09 | `09-intrinsic-geometry` | 크리스토펠 기호. 가우스 방정식, 코다치-마이나르디 방정식. 빼어난 정리. 공변미분, 평행이동, 측지곡률, 측지선 | dC §4-3–4-4 | 구면 위 평행이동, 측지선 |
| 10 | `10-gauss-bonnet` | 국소·전역 가우스-보네. 오일러 지표와 응용. (선택) 지수사상과 측지극좌표 | dC §4-5 (4-6) | 측지삼각형, 삼각분할 |

**Part II. 매끄러운 다양체**

| 장 | 폴더 | 핵심 내용 | 대응 교재 | 대표 그림 |
|---|---|---|---|---|
| 11 | `11-smooth-manifolds` | 위상다양체. 차트, 아틀라스, 매끄러운 구조. 예: ℝⁿ, Sⁿ, ℝℙⁿ, 곱, 그래프, 열린부분집합, GL(n,ℝ). 경계를 갖는 다양체. Part I 매개화와의 관계 | Lee-ISM 1장 | 차트와 좌표변환, 입체사영 |
| 12 | `12-smooth-maps` | 매끄러운 함수와 사상, 미분동형사상. 범프 함수, 단위분할 | Lee-ISM 2장 | 범프 함수, 단위분할 |
| 13 | `13-tangent-spaces` | derivation으로서의 접벡터, 미분 dF_p, 좌표 계산(합 규약 도입), 접다발. 곡선의 속도. 동치인 정의들 | Lee-ISM 3장 | p를 지나는 곡선들과 속도 |
| 14 | `14-submanifolds` | 몰입, 침몰, 매장. 계수정리. 정칙값 정리. 매장·몰입 부분다양체. (Sard는 진술만) | Lee-ISM 4–6장 | 8자 몰입, 정칙값의 역상 |
| 15 | `15-vector-fields-flows` | 벡터장, 리 괄호, 적분곡선, 흐름, 리 미분 | Lee-ISM 8–9장 | 흐름선, 리 괄호의 틈 |
| 16 | `16-covectors-tensors` | 여접다발, 1-형식, df, 당김, 선적분. 텐서, 대칭·교대. 텐서장. 유클리드 계량을 대칭 2-텐서장의 예로 제시 (리만 계량의 정의는 21장) | Lee-ISM 11–13장 | 1-형식의 등위면 그림 |
| 17 | `17-differential-forms` | 교대텐서, 쐐기곱, 외미분, 내부곱. 리 미분과 카르탕 공식 | Lee-ISM 14장 | 쐐기곱 = 부호 있는 넓이 |
| 18 | `18-integration-stokes` | 방향, 다양체 위의 적분, 스토크스 정리 | Lee-ISM 15–16장 | 경계의 유도 방향 |
| 19 | `19-de-rham` | 닫힌·완전 형식, 푸앵카레 보조정리, 호모토피 불변성, 마이어-비토리스, 계산 예 | Lee-ISM 17장. BT | ℝ² ∖ {0} 위의 dθ |
| 20 | `20-lie-groups-frobenius` | (선택) 리 군 입문, 분포와 프로베니우스 정리 | Lee-ISM 7, 19, 20장 | 적분가능/불가능 분포 |

**Part III. 리만 기하**

| 장 | 폴더 | 핵심 내용 | 대응 교재 | 대표 그림 |
|---|---|---|---|---|
| 21 | `21-riemannian-metrics` | 리만 계량과 예(유클리드, 둥근 구면, 쌍곡). 등거리사상. 길이와 거리. 음악적 동형. 체적형식. grad·div·Δ | Lee-IRM 2–3장. dC-RG 1장 | 𝕌² 위의 단위원들 |
| 22 | `22-connections` | 선형접속, 크리스토펠 기호. 곡선을 따른 공변미분, 평행이동. 비틀림, 계량 호환성. 기본정리(레비치비타) | Lee-IRM 4–5장. dC-RG 2장 | 평행이동과 홀로노미 |
| 23 | `23-geodesics` | 측지선, 지수사상, 정규좌표. 가우스 보조정리. 국소 최단성. 호프-리노프 정리 | Lee-IRM 6장. dC-RG 3, 7장 | exp_p와 측지공 |
| 24 | `24-curvature` | 곡률텐서와 대칭성. 단면·리치·스칼라 곡률. 상수곡률 모형공간 | Lee-IRM 7장. dC-RG 4, 8장 | 세 모형공간의 측지삼각형 |
| 25 | `25-submanifold-geometry` | 제2기본형식, 가우스 공식·방정식. Part I 곡면론의 재해석 | Lee-IRM 8장. dC-RG 6장 | 법성분·접성분 분해 |
| 26 | `26-gauss-bonnet-jacobi` | (선택) 가우스-보네 재방문, 야코비 장과 공액점 | Lee-IRM 9–10장. dC-RG 5장 | 구면 위 측지선의 수렴 |

**장 사이의 의존 관계**

- Part I에는 01–03장이 필요하다. 04–05장도 03장의 컴팩트성 결과(정리 3.6.10, 르베그 수, 균등연속)를 쓰고, 06장부터는 부분공간 위상과 위상동형이 본격적으로 쓰인다. Part II에도 01–03장이 필요하다.
- Part I과 Part II는 대부분 서로 독립이라 **병렬로 작성할 수 있다**. 11장의 "Part I과의 관계" 절만 06장 이후에 쓴다.
- Part III에는 Part II(특히 13–18장)가 필요하고, Part I을 자주 참조한다.

---

## 10. 디렉터리 구조와 파일 명명

```
DG-study/
├── GUIDELINES.md              ← 이 문서 (agent용, 유일한 Markdown 콘텐츠)
├── CLAUDE.md                  ← agent 진입점: 이 문서를 가리킴
├── PLAN.md                    ← 통합 담당용 진행 계획과 진행 기록
├── index.html                 ← 전체 목차(독자 진입점)
├── notation.html              ← 기호표 (§6의 독자용 판)
├── glossary.html              ← Glossary: 영-한 용어집 (§8)
├── references.html            ← 참고문헌 (§5, 항목 id = 약어)
├── concepts.html              ← Index: 영어 용어 → 정의 위치·일반화·쓰이는 곳 (자동 생성, §3.6)
├── assets/
│   ├── style.css              ← 유일한 스타일시트
│   └── mathjax-config.js      ← 유일한 MathJax 설정 (부록 C)
├── tools/
│   ├── dgfig.py               ← 그림 공용 스타일·헬퍼
│   ├── dgsym.py               ← §6 규약을 구현한 기호계산
│   ├── dgnum.py               ← 수치 헬퍼 (RK4 등)
│   ├── dgcheck.py             ← verify 스크립트용 check/sym_equal/close/summary
│   ├── check_site.py          ← HTML 구조·링크 검사
│   ├── build_index.py         ← concepts.html 생성, 중복 정의 검사
│   ├── check_math.mjs         ← 수식(TeX) 오류 검사
│   ├── term_guard.py          ← 용어·라벨 변환 전후 비교 (id·수식·구조·링크 대상, 남은 한국어)
│   ├── terms.tsv              ← 용어표 정본: 한국어 → 영어 (§8.3)
│   ├── terms_keep.txt         ← 한국어로 두는 말 (§8.1)
│   ├── package.json
│   ├── check_all.sh           ← 전체 검사
│   └── tests/                 ← 도구 자체의 테스트와 fixture (사이트 검사 대상 아님)
├── refs/                      ← (선택) 사용자가 넣는 교재 PDF. 인용·복사 금지, 참조만
└── chapters/
    └── 06-regular-surfaces/
        ├── index.html                      ← 장 개요
        ├── 6-1-definition.html             ← 절
        ├── 6-2-tangent-plane.html
        ├── figures/
        │   ├── fig-6-2-1-tangent-plane.py
        │   └── fig-6-2-1-tangent-plane.svg
        └── verify/
            └── v-6-2-tangent-plane.py
```

- 장 폴더는 `NN-slug`로 이름 짓는다. 두 자리 번호에 영어 kebab-case를 붙인다. 이름은 §9 표를 따른다.
- 절 파일은 `N-M-slug.html`로 이름 짓는다. 장 번호에 앞자리 0을 붙이지 않는다. 장 개요는 `index.html`이다.
- 그림
  - 스크립트는 `figures/fig-N-M-K-slug.py`로 짓고, 출력은 같은 이름의 `.svg`로 한다.
  - 필요하면 `.png`도 만든다. 인터랙티브 그림은 `-interactive.html`을 붙인다.
- 검증 스크립트는 `verify/v-N-M-slug.py`로 짓는다. 절마다 하나다.
- 모든 파일명은 영문 소문자, 숫자, 하이픈만 쓴다.

---

## 11. HTML 작성 규칙

### 11.1 기술 스택

- 순수 정적 HTML5에 공용 CSS(`assets/style.css`)와 MathJax 3을 쓴다.
  - MathJax는 `assets/mathjax-config.js`를 먼저 읽은 뒤 CDN의 `mathjax@3.2.2`를 불러온다.
- 빌드 도구는 없다. 파일을 브라우저로 **바로 열어서(file://)** 볼 수 있어야 한다.
- JavaScript는 MathJax 말고는 쓰지 않는다. 예외는 인터랙티브 그림 파일뿐이다(§12.7).
- 인라인 `style="…"`, 페이지 안의 `<style>`, 페이지별 매크로 정의는 **금지**한다. 새 스타일이 필요하면 최종 보고에 제안하고, 통합 담당이 style.css에 클래스로 추가한다.
- 오프라인 열람이 필요하면(사용자가 요청할 때) MathJax를 `assets/vendor/`에 두고 경로만 바꾼다.

### 11.2 페이지 골격

부록 A의 템플릿을 그대로 쓴다. 필수 요소는 다음과 같다.

- `<html lang="ko">`, `<meta charset="utf-8">`, viewport
- `<title>`, dg 메타데이터(§11.6)
- style.css → mathjax-config.js → MathJax CDN 순서의 로드
- 상단 breadcrumb (Contents › 장 › 절, §8.2)
- pager: 장의 첫 절의 이전 링크는 자기 장 개요로, 장의 마지막 절의 다음 링크는 다음 장 개요로 간다. 독자가 장을 넘어갈 때 개요의 "Builds on / Questions"를 먼저 보게 하기 위해서다.
- `<main>` 안에 Goals → Prerequisites → 본문 → Summary → Exercises
- 하단 이전/다음 절 링크

### 11.3 구성요소 (클래스 계약)

style.css는 아래 클래스를 전부 지원해야 하고, 페이지는 아래 클래스만 쓴다. `check_site.py`는 목록에 없는 클래스를 **오류**로 처리한다. 모든 구성요소가 쓰인 올바른 예는 `tools/tests/fixtures/site/`에 있으니, 페이지를 쓰기 전에 참고한다. 라벨·제목은 영어, 본문 문장은 한국어다(§8.2).

**HTML 작성 규칙**
- `<p>`, `<li>`를 포함한 모든 비어 있지 않은 태그는 명시적으로 닫는다.
- `<p>` 안에 블록 요소(`div.equation`, 목록, `aside`, `figure`)를 넣지 않는다. 문단을 닫고 블록을 둔 뒤 새 문단을 연다.
- id는 페이지 안에서 유일해야 한다.

- **정의·정리류**
  ```html
  <div class="thm definition" id="i-6-2-1">
    <p><span class="thm-head">Definition 6.2.1</span> <span class="thm-name">(tangent plane)</span>. 본문…</p>
  </div>
  ```
  - 두 번째 클래스 목록: `definition`, `theorem`, `proposition`, `lemma`, `corollary`, `example`, `nonexample`, `remark`
  - 호환성 명제(§3.4)에는 세 번째 클래스 `compat`을 붙인다.
  - 제목 단어: Definition, Theorem, Proposition, Lemma, Corollary, Example, Non-example, Remark. `thm-head`에는 "단어 번호"만 쓴다(예: `Non-example 6.2.3`).
  - `thm-name`은 괄호 안의 영어 이름이다. 용어처럼 소문자로 쓰고 고유명사만 대문자로 쓴다(예: `(first fundamental form)`, `(Theorema Egregium)`).
- **증명**
  ```html
  <div class="proof">
    <p class="proof-idea">Proof idea: …</p>          <!-- 선택 -->
    <p><span class="proof-head">Proof.</span> …</p>
  </div>
  ```
  - 끝 표시 ∎는 CSS가 붙이므로 직접 쓰지 않는다.
  - 스케치는 `<span class="proof-head">Proof sketch.</span>`로 쓴다.
- **상자**: `<aside class="box 종류">`로 쓰고, 첫 줄은 `<p class="box-title">…</p>`다. 제목은 아래 영어로 쓰고, 필요하면 짧은 한국어 부제를 붙인다(예: `Intuition: 같은 대상, 다른 눈`).
  - `intuition`: Intuition (직관)
  - `warning`: Warning / Common misconception (주의 / 흔한 오해)
  - `convention`: Conventions (규약 비교)
  - `forward`: Looking ahead (나중에 보게 될 일반화)
  - `backward`: Looking back (앞에서 본 것과 비교)
  - `history`: History (역사, 선택)
  - `application`: In computer vision (주로 Chapter 0 Tour, §2.3)
  - `deeper`: Go deeper (깊은 장으로 가는 링크, §2.3)
- **번호 붙은 식**
  ```html
  <div class="equation" id="eq-6-2-1">\[ … \tag{6.2.1} \]</div>
  ```
- **그림**
  ```html
  <figure class="dg-figure" id="fig-6-2-1">
    <img src="figures/fig-6-2-1-tangent-plane.svg" alt="(그림 내용 요약: 한국어 문장, 용어는 영어)">
    <figcaption><span class="fig-head">Figure 6.2.1.</span> 설명…</figcaption>
  </figure>
  ```
- **용어 정의**: `<dfn>tangent plane</dfn>(접평면)`. `<dfn>`에는 영어 용어를, 바로 뒤 괄호에는 한국어 번역어를 한 번 적는다(§8.1). `build_index.py`는 `<dfn>`의 글자를 용어로, 괄호 안의 한글을 번역어로 읽는다.
  - 일반화하는 정의에는 `data-generalizes="(이전 정의의 상대경로#id)"`를 붙인다(§3.3).
  - `<dfn>`은 정의 항목(`div.thm.definition`) 안이나 정의하는 문장 안에서만 쓴다. `build_index.py`가 가장 가까운 id를 정의 위치로 기록한다.
- **다음 절 예고**: 요약 section 끝의 `<p class="next">다음 절에서는 …</p>`
- **인용**: `<a class="cite" href="../../references.html#dC">[dC §2-4]</a>`
- **이 study set의 규약 도입**: 번호 붙은 비고(Remark)로 쓰고 제목에 "convention"을 넣는다. 예: `<div class="thm remark" id="i-1-4-9">`와 `(index position convention)`. 번호가 있어야 뒤에서 링크할 수 있다. 교재와 다른 점을 알리는 것은 따로 `box convention`(Conventions)으로 쓴다.
- **확인 필요**: `<span class="todo">[확인 필요: …]</span>`
- **검증 표시**: `<p class="verified">Verification: <code>verify/v-7-1-first-form.py</code></p>` (계산 예제 끝, 선택)
- **연습문제**
  ```html
  <div class="exercise" id="ex-6-2-1" data-level="1">
    <p><span class="thm-head">Exercise 6.2.1</span> <span class="level">★</span> 문제…</p>
    <details class="hint"><summary>Hint</summary><p>…</p></details>
    <details class="solution"><summary>Solution</summary><p>…</p></details>
  </div>
  ```
- **절 구조**: `<section class="goals">`(소제목 Goals), `<section class="prereq">`(Prerequisites), `<section class="summary">`(Summary), `<section class="exercises">`(Exercises). 동기 소제목은 Motivation이고, 그 밖의 소제목도 영어로 쓴다(§8.2).
- **장 개요의 절 목록**: `<ol class="toc">`. 각 항목에 상태 표시를 단다.
  - 상태 표시의 형식: `<span class="status 상태">라벨</span>`. 상태별 라벨은 `planned` Planned, `draft` Draft, `reviewed` Reviewed, `final` Final이다.
  - **아직 쓰지 않은 절**은 링크를 걸지 않는다. 예정 파일 이름은 `data-file`에 적는다.
    ```html
    <li id="toc-7-1" data-file="7-1-first-fundamental-form.html">
      <span class="toc-title">7.1 The First Fundamental Form</span> <span class="status planned">Planned</span>
      <p class="toc-note">Topics: 정의와 coordinate representation, 길이·각·넓이 · Running examples: sphere·cylinder의 E, F, G</p>
    </li>
    ```
  - 절을 쓰면 `toc-title`을 `<a class="toc-title" href="7-1-first-fundamental-form.html">`로 바꾸고 상태를 올린다.
  - `toc-note`의 머리는 "Topics:", "Running examples:"로 쓰고, 조각은 " · "로 나눈다.
- **장 개요의 연결 섹션**: `<section class="inherits">`(소제목 Builds on), `<section class="questions">`(Questions), `<section class="handoff">`(Leads to), `<p class="threads">`("Threads: …", 줄기 위치). 절 목록의 소제목은 Sections다. 남긴 질문은 "Open question: …"으로 쓴다. 부록 A-2를 따른다.

### 11.4 수식

- 인라인 수식은 `\( … \)`, 디스플레이 수식은 `\[ … \]`로 쓴다. **`$`는 구분자로 쓰지 않는다**(설정에서 꺼져 있음).
- 여러 줄 수식은 `\[ \begin{aligned} … \end{aligned} \]`로 쓴다. `align` 환경은 쓰지 않는다.
- `\tag{…}`는 `aligned` **밖**, `\end{aligned}` 뒤에 둔다(`\[ \begin{aligned} … \end{aligned} \tag{7.1.2} \]`). `aligned` 안에 두면 MathJax 오류가 난다. `\tag`는 `div.equation` 안에서만 쓴다.
- **HTML 특수문자 주의**
  - 수식 안의 `<`, `>`는 `\lt`, `\gt`로 쓴다. `<`가 HTML 태그로 파싱되어 수식이 깨질 수 있다.
  - 정렬 기호 `&` 바로 뒤에는 공백, `=`, `\` 중 하나를 둔다. 예: `&= `, `& \quad`
- **허용 패키지**: base, ams, amscd(`\begin{CD}` 가환도표), boldsymbol, 부록 C의 프로젝트 매크로.
  - 페이지 안에서 `\newcommand`나 `\def`를 쓰지 않는다.
  - 새 매크로가 필요하면 최종 보고에 제안한다. 통합 담당이 부록 C와 config에 **동시에** 반영한다.
- **가환도표**
  - 수평·수직 화살표만 있으면 `\begin{CD}`를 쓴다.
  - 대각선 화살표 한두 개는 `\begin{array}`에 `\nearrow`, `\searrow`로 처리한다.
  - 더 복잡하면 그림으로 그린다(§12).
- 연산자 이름은 똑바로 선 글꼴로 쓴다(`\tr`, `\det`, `\rank`, `\operatorname{…}`). 이탤릭으로 쓰지 않는다.
- 수식도 문장의 일부다. 디스플레이 식 끝의 문장부호(`.` `,`)는 식 안에 넣는다.
- 수식을 이미지로 넣지 않는다.

### 11.5 번호, id, 상호참조

- **번호는 절 단위로 매긴다.**
  - 정의·정리·예 등은 한 카운터를 공유한다: Definition 6.2.1, Example 6.2.2, Theorem 6.2.3 …
  - 식 `(N.M.K)`, 그림 `Figure N.M.K`, 연습 `Exercise N.M.K`는 각각 따로 센다.
- **id 규칙**: 항목 `i-N-M-K`, 식 `eq-N-M-K`, 그림 `fig-N-M-K`, 연습 `ex-N-M-K`, 절 안의 소제목 `sec-slug`.
- **참조는 항상 링크로 건다.**
  - 같은 파일: `<a href="#i-6-2-3">Theorem 6.2.3</a>`
  - 같은 장의 다른 절: `<a href="6-1-definition.html#i-6-1-4">Definition 6.1.4</a>`
  - 다른 장: `<a href="../05-space-curves/5-3-frenet.html#eq-5-3-2">(5.3.2)</a>`
  - 식은 단어 없이 `(N.M.K)`로 참조한다. 문장 첫머리에서만 `Equation <a href="#eq-6-2-1">(6.2.1)</a>`처럼 쓴다. 장·절은 "Chapter N", "Section N.M"으로 쓴다("N장", "N.M절"이라 쓰지 않는다, §8.2).
  - `check_site.py`는 링크 없는 `Theorem N.M.K`·`Figure N.M.K`·`Exercise N.M.K`와 수식 밖의 `(N.M.K)`, 남은 한국어 참조(`정리 N.M.K`, `식 (N.M.K)`, `N장`, `N.M절`)를 경고한다.
- 번호를 바꿨다면 저장소 전체에서 옛 번호를 grep해 참조를 모두 고친다. check_site.py가 깨진 링크와 번호 불일치를 잡는다.
- 아직 없는 절을 참조해야 하면 §9 로드맵을 기준으로 링크를 걸고 `[확인 필요: 참조 대상 미작성]`을 붙인다.

### 11.6 메타데이터

`<head>` 안에 넣는다. 절 페이지의 예는 다음과 같다.

```html
<meta name="dg-id" content="6.2">
<meta name="dg-status" content="draft">        <!-- planned | draft | reviewed | final -->
<meta name="dg-prereq" content="6.1, 2.3">
<meta name="dg-refs" content="dC §2-4; Pr Ch. 4">
<meta name="dg-updated" content="2026-09-30">
```

페이지 종류별로 필요한 메타데이터는 다음과 같다.

| 페이지 | 경로 | `dg-id` | 그 밖의 필수 메타 |
|---|---|---|---|
| 절 | `chapters/NN-slug/N-M-slug.html` | `N.M` | `dg-status`, `dg-prereq`, `dg-refs`, `dg-updated` |
| 장 개요 | `chapters/NN-slug/index.html` | `N` | `dg-status`, `dg-updated` |
| 공용 페이지 | 루트의 `index.html`, `notation.html`, `glossary.html`, `references.html`, `concepts.html` | 파일 이름(`index`, `notation`, …) | `dg-updated` |

모든 페이지는 `lang="ko"`, charset, viewport, `<title>`을 갖춘다. 또 style.css → mathjax-config.js → MathJax CDN 순서로 불러온다. 수식이 없는 페이지도 예외가 아니다.

**상태(`dg-status`)의 뜻**

- `planned`: 뼈대나 계획만 있다. Phase 0에서 만든 장 개요가 이 상태다.
- `draft`: 작성 완료
- `reviewed`: 검토 완료
- `final`: 사용자 확인

장 개요의 `dg-status`는 **그 장의 모든 절이 `draft` 이상이 된 뒤에** `draft`로 올린다.

---

## 12. 그림

### 12.1 원칙

- 교재 그림은 **무엇을 어떻게 그릴지**(구도, 표시할 대상, 관례)를 정하는 참고로만 쓴다. 이미지를 복사·스캔·트레이싱하지 않고 새로 계산해서 그린다.
- **정량적 그림은 실제 수식으로 계산한다.** 곡선, 곡면, 벡터, 법선, 측지선 등이 여기에 속한다.
  - 예: 프레네 틀은 매개화에서 계산한 값으로 그리고, 접촉원의 반지름은 1/κ로 그린다.
  - 눈대중으로 그리지 않는다.
- 그림에는 **본문과 같은 기호, 같은 매개변수 값**을 쓴다. 그 값은 캡션에 적는다. 예: "a = 1, b = 0.3인 나선"
- 추상 다양체, 차트, 사상 도식 같은 **개념도**는 캡션 끝에 "(schematic)"이라고 표시한다. 정확한 수치 표현이 아님을 독자가 알 수 있게 하기 위해서다.
- **재생성할 수 있어야 한다.** 프로젝트 루트에서 `PYTHONPATH=tools python3 <스크립트>`를 한 번 실행하면 같은 파일이 나와야 한다. 난수를 쓰면 seed를 고정한다.
- 스크립트 안에 **자기검사 assert**를 넣는다. 예: |**t**| = 1, ⟨**t**, **n**⟩ = 0, 그린 점이 곡면 위에 있는지

### 12.2 도구

- Python 3, matplotlib 3.9, numpy를 쓴다. 공식은 sympy로 유도한 뒤 `lambdify`로 계산한다.
- **LaTeX가 설치되어 있지 않다.** `text.usetex=False`로 두고 mathtext를 쓴다. 교과서 느낌이 나도록 `mathtext.fontset="cm"`을 쓴다.
- 스타일은 반드시 `tools/dgfig.py`로 설정한다(부록 D). 스크립트 첫머리에서 `import dgfig; dgfig.setup()`을 호출한다.
- 출력은 스크립트와 같은 폴더에 같은 이름의 `.svg`로 저장한다(`dgfig.save`).
- 장 의존 그래프 같은 도식적 그래프는 graphviz(`dot -Tsvg`)로 그린다.
- 수치 ODE(측지선, 평행이동)는 `tools/dgnum.py`의 RK4를 쓴다. 이 환경에서는 scipy를 import할 수 없다(부록 E).
- 목록에 없는 도구(tikz, manim 등)가 필요하면 사용자에게 먼저 묻는다.

### 12.3 스타일 (교과서 선화 느낌)

- 흰 배경, 얇은 선, 절제된 색을 쓴다. 3D 좌표축은 의미가 있을 때만 그린다. 기본은 `set_axis_off()`다.
- **색은 의미에 고정한다.** Okabe–Ito 색맹 안전 팔레트를 쓰며, 값은 `dgfig.COLORS`에 있다.

| 역할 | 색 | hex |
|---|---|---|
| 주 대상(곡선, 곡면 윤곽, 점) | 검정 | `#000000` |
| 곡면의 면 | 연회색, alpha 0.35–0.6 | `#D0D0D0` |
| 접벡터 (**t**, **x**_u, **x**_v, X, T_pM의 v) | 파랑 | `#0072B2` |
| 법벡터 (**n**, **N**) | 주홍 | `#D55E00` |
| 세 번째 벡터 (**b**, 두 번째 벡터장 Y) | 청록 | `#009E73` |
| 강조 곡선·보조 벡터 (측지선, 평행이동 결과) | 주황 | `#E69F00` |
| 영역 채우기 (U, 차트의 치역) | 하늘색, alpha 0.25 | `#56B4E9` |
| 접평면 | 하늘색 alpha 0.2 + 파랑 테두리 | `#56B4E9` / `#0072B2` |
| 보조선, 숨은선 | 회색 점선 | `#777777` |
| 여벡터·1-형식 (등위선) | 자주 | `#CC79A7` |
| 곡면의 𝐍과 곡선의 𝐧이 한 그림에 함께 나올 때 | 𝐍은 주홍 실선, 𝐧은 주황 점선 | `#D55E00` / `#E69F00` |
| 곡률 부호 색칠 | 발산형 컬러맵 `RdBu_r`, 0이 중앙 (`TwoSlopeNorm`) | — |

- **선 굵기**: 주 곡선 1.8pt, 벡터 1.5pt, 곡면 격자 0.3pt, 보조선 0.8pt.
- **라벨**은 mathtext만 쓴다. 예: `r"$\mathbf{t}(s)$"`. 크기는 11–12pt이고, 라벨 기호는 본문 기호와 **글자 그대로** 같아야 한다.
- **그림 안에는 한글을 넣지 않는다.** 한글과 mathtext를 한 문자열에 섞으면 한글이 깨진다(확인함). 설명은 figcaption에 쓴다.
- **크기**: 단독 그림은 6×4.5 in, 두 개를 나란히 둘 때는 각각 폭 3.2 in로 한다.
- **패널 수**: 한 그림에 패널은 2개 이하가 좋다. 3패널 그림은 폰 폭에서 라벨이 너무 작아진다. 3개가 꼭 필요하면 라벨을 12pt 이상으로 하고, 폭 900px과 400px에서 모두 확인한다.
- **형식**: SVG(`svg.fonttype="path"`)로 저장한다. 1.5MB를 넘으면 격자 해상도를 낮추거나 PNG(200 dpi)로 저장한다.
- **숨은선**: 가려진 곡선 부분은 교과서처럼 점선으로 그리기를 권장한다(가능한 경우).

### 12.4 3D 그림 주의사항 (mplot3d)

- **깊이 정렬이 부정확하다.** 벡터가 곡면 뒤로 숨거나 반대로 비칠 수 있다. 다음 방법으로 대응한다.
  - 곡면의 alpha를 낮춘다.
  - 벡터와 곡선은 곡면보다 나중에 그린다.
  - `computed_zorder=False`로 두고 zorder를 직접 정한다.
  - 벡터가 잘 보이는 시점(`view_init(elev, azim)`)을 고른다.
- **비율을 유지한다.** `ax.set_box_aspect`로 실제 x:y:z 비율을 맞춘다. 구가 타원처럼 보이면 안 된다.
- **벡터 길이**: 실제 길이로 그리는 것이 기본이다. 배율을 쓰면 캡션에 적는다.
- **환경 문제**: `projection="3d"`는 부록 E의 문제를 먼저 해결해야 쓸 수 있다. `dgfig.setup()`이 해결해 준다.

### 12.5 교재의 대표적 그림 관례

- **매개화와 차트**
  - 왼쪽에 ℝ²의 U를 하늘색으로 채우고 좌표격자를 그린다.
  - 오른쪽에 곡면이나 다양체 위의 **x**(U)를 그린다.
  - 둘 사이의 곡선 화살표에 `$\mathbf{x}$` 또는 `$\varphi$` 라벨을 단다(dC 2장, Lee-ISM 1장 스타일).
- **좌표변환**: 겹치는 두 영역 U, V와 각 차트의 치역, 그리고 그 사이의 `$\psi\circ\varphi^{-1}$` 화살표를 그린다.
- **접평면**: 점 p, 반투명 평행사변형 T_pS, 파랑 **x**_u·**x**_v, 주홍 **N**을 그린다.
- **프레네 틀**: 곡선 위 여러 점에 (**t**, **n**, **b**) 삼각대를 그린다.
- **법곡률**: 한 점을 지나는 법평면 단면 곡선들을 그린다.
- **추상 다양체**: 푸리에 섭동으로 만든 매끈한 닫힌 곡선을 "얼룩" 모양의 M으로 쓰고, 그 위에 점 p와 곡선을 그린다.
- **리 괄호**: 두 흐름을 번갈아 따라간 경로가 닫히지 않고 남기는 틈을 그린다.
- **평행이동**: 구면의 위도원이나 측지삼각형을 따라 옮긴 벡터들을 그린다.

### 12.6 캡션

- `Figure N.M.K.` 다음에 무엇을 보여주는지 1–3문장, 사용한 매개변수, (개념도 표시)를 적는다. 문장은 한국어, 용어는 영어다(§8.2). 캡션 안의 수식은 MathJax로 쓴다.
- 모든 그림은 본문에서 **한 번 이상 참조**한다.
- `alt` 속성에 그림 내용을 한국어 문장(용어는 영어)으로 요약한다(스크린리더용).

### 12.7 인터랙티브 그림 (선택)

- 3D 회전이나 매개변수 슬라이더가 이해에 결정적일 때만 만든다. 정적 SVG는 **항상** 함께 둔다.
- 도구는 plotly 또는 JSXGraph(jsDelivr)를 쓴다. plotly는 Python에서 `include_plotlyjs="cdn"`으로 별도 HTML 파일을 만든다.
- 정적 그림과 같은 공식, 같은 매개변수를 쓴다. 본문에서는 링크나 `<iframe class="interactive">`로 연결한다.

---

## 13. 계산 검증

- **검증 대상**: 본문의 모든 비자명한 계산 결과
  - 기본형식 계수, 곡률, 크리스토펠 기호, 곡률텐서, 측지선 방정식
  - 적분값, 항등식(리 괄호의 좌표식, 카르탕 공식의 예 등)
  - 예제에서 주장하는 수치
- **방법**
  - 정확한 sympy 기호계산을 우선 쓴다. 불가능하면 numpy 수치 확인으로 대신하고 허용오차를 명시한다.
  - 검증은 증명을 대신하지 않는다. **증명과 계산의 오류를 잡는 용도**다.
- **파일**: `chapters/NN-slug/verify/v-N-M-slug.py`
  - 각 검사는 `tools/dgcheck.py`의 헬퍼로 쓴다. 예: `check("Example 7.1.2: sphere E,F,G", 조건)`, `sym_equal(이름, a, b)`, `close(이름, a, b, tol)`
  - 각 검사는 PASS/FAIL을 출력하고, 스크립트 끝에서 `summary()`를 호출한다.
  - 하나라도 실패하면 종료코드 1로 끝낸다.
  - 검사 이름에 본문 항목 번호를 넣는다.
- **규약 일원화**: `tools/dgsym.py`(부록 D)에 구현된 함수만 쓴다. 개별 스크립트에서 곡률 공식을 새로 짜지 않는다. 규약을 한 곳에만 구현해야 부호 실수를 막을 수 있다.
- **sympy 사용 요령**
  - 기호에 가정을 붙인다. 예: `sp.symbols('r', positive=True)`, θ ∈ (0, π)이면 `sin θ > 0`임을 반영한다. 그렇지 않으면 `sqrt(sin(θ)**2)` 같은 식이 남는다.
  - `simplify` 결과가 0이 아니라고 바로 오류로 단정하지 않는다. `.equals(0)`이나 임의의 수치 대입으로 다시 확인한다.
- 그림에 쓰는 수치 적분과 ODE(측지선, 평행이동)도 같은 코드로 검증한다. 예: §7의 평행이동 회전각

---

## 14. 연습문제

- 절마다 4–8개를 둔다.
- **난이도**
  - ★: 정의 확인, 직접 계산
  - ★★: 몇 단계의 논증, 표준적인 증명
  - ★★★: 도전 문제, 여러 절의 종합
  - ★ 문제가 절반 이상이 되게 한다.
- **유형을 섞는다**: 계산 / 증명 / 개념 확인(참·거짓 판별과 이유) / 그림 해석
- 모든 문제에 **완전한 풀이**(`details.solution`)를 단다. 필요하면 힌트(`details.hint`)도 단다.
  - 풀이도 본문과 같은 엄밀성 기준을 따른다.
  - 계산 풀이는 verify 스크립트로 확인한다.
- 본문 증명에서 "연습문제로 남긴다"는 ★ 수준의 짧은 확인에만 쓰고, 해당 연습 번호에 링크를 건다.
- **기준 예제를 연습문제에서 미리 계산했다면** "Chapter N에서 사용"이라고 표시한다. 뒤 장은 그 값을 가정하지 않고 §3.5의 자기 칸에서 본문으로 다시 계산한다. 이때 "Exercise 1.5.2에서 미리 해 보았다"처럼 링크를 건다.
- **이후 절에서 쓰는 결과는 연습문제로 두지 않는다.** 본문에서 증명한다. 부득이하게 연습문제로 둔다면 "Theorem x.y.z에서 사용"이라고 표시하고 풀이를 완전하게 쓴다.

---

## 15. 작업 절차

### 15.1 작업 단위

- 기본 단위는 **장 하나**다. agent 한 명이 한 장의 절들을 **순서대로** 쓴다(§3.7).
- 장이 너무 크면 절 단위로 나눠 맡길 수 있다. 이때도 앞 절이 `draft`가 된 뒤에 다음 절을 시작한다. 같은 장의 절을 동시에 쓰지 않는다.
- 동시에 진행하는 것은 서로 독립인 장끼리만 한다(§9의 의존 관계).

### 15.2 시작 전

1. 이 문서 전체를 읽는다.
2. 다음 파일을 읽는다.
   - 담당 장의 `index.html` (없으면 §9 로드맵). 특히 "Builds on"
   - **앞 장의 `index.html`**, 특히 "Leads to"
   - 선행 절: 적어도 바로 앞 절 전체와, 선수 지식으로 링크할 절 전부
   - `concepts.html`: 쓰려는 개념이 이미 정의되어 있는지 확인한다(§3.3)
   - `notation.html`, `glossary.html`
   - §3.1–3.5에서 담당 장에 해당하는 줄기, 호환성 명제, 기준 예제의 칸
3. `refs/`에 해당 교재가 있으면 대응 절을 읽는다. 구성과 관례를 파악하는 용도이며, 문장은 복사하지 않는다.
4. 계획을 세운다.
   - 정의·정리·예·그림·연습의 목록과 번호
   - 각 항목에 대해 **새로 정의하는 것 / 상기하는 것(링크) / 일반화하는 것**을 구분한다.
   - 로드맵 범위를 벗어나는 내용이 필요하면 최종 보고에 제안한다.

### 15.3 작성

- 부록 A 템플릿에서 시작한다. 문체와 설명 밀도는 부록 B를 기준으로 삼는다.
- 그림 스크립트와 verify 스크립트를 작성하고 실행한다.

### 15.4 마무리

1. `bash tools/check_all.sh --chapter NN`을 통과시킨다. 담당 장의 HTML 구조, 링크, 수식, 그림 재생성, 검증 스크립트를 검사하고, 중복 정의는 저장소 전체에서 검사한다. 인자 없는 전체 검사는 통합 담당이 한다.
2. 가능하면 브라우저로 열어 눈으로 확인한다(claude-in-chrome 도구를 쓸 수 있을 때).
   - 수식 오류(빨간 글씨)가 없는지
   - 그림이 표시되는지
   - 좁은 화면에서도 읽히는지
3. §16 체크리스트를 확인한다.
4. **최종 보고**에 다음을 적는다.
   - 만든 파일
   - 새 기호와 새 용어
   - `[확인 필요]` 목록
   - 규약에 관해 내린 결정
   - **다음 장이 알아야 할 것**: 넘겨주는 결과, 남긴 질문, 예고만 하고 정의하지 않은 개념
   - 로드맵·스타일·매크로 변경 제안

### 15.5 병렬 작업

- **공유 파일은 통합 담당(orchestrator)만 수정한다.** 공유 파일은 다음과 같다.
  - `index.html`, `notation.html`, `glossary.html`, `references.html`, `concepts.html`(자동 생성)
  - `assets/*`, `tools/*`, 이 문서
- 작성 agent는 **담당 장 폴더 안의 파일**만 만들고 고친다. 담당 장의 `index.html`, 절 파일, figures, verify가 여기에 속한다.
- 다른 장의 파일에서 고칠 점을 발견하면 직접 고치지 말고 최종 보고에 적는다.
- **선행 장과 동시에 쓰는 경우** (사용자가 속도를 위해 병렬을 지시했을 때)
  - 선행 장에서 이미 존재하는 절은 정확한 번호로 링크한다.
  - 아직 없는 절을 참조해야 하면 그 장의 `index.html`로 링크하고 `<span class="todo">[확인 필요: NN장 미작성 — 참조 번호 확정 필요]</span>`를 붙인다. 존재하지 않는 파일로 링크하지 않는다.
  - 선행 장이 정할 개념을 미리 정의하지 않는다. 기호는 §6을 그대로 따른다.
  - 작업을 끝내기 직전에 선행 장 폴더를 다시 확인해 링크를 확정한다.
  - 남은 todo는 장 검토 단계에서 모두 확정한다. 이것은 장 검토 agent의 필수 항목이다.
- 공유 파일에 반영할 내용은 최종 보고에 적는다.

### 15.6 검토

- 검토는 작성자와 **다른** agent가 한다. 검토는 세 단계로 나뉜다.
  1. **절 검토**: 절 하나가 `draft`가 되면 한다.
  2. **장 통합 검토**: 장의 모든 절이 `draft`가 되면, 장 전체를 **처음부터 끝까지 한 번에** 읽는다.
     - 절 사이의 흐름이 이어지는가(앞 절이 끝난 지점에서 다음 절이 시작하는가)
     - 빠진 논리 단계나 중복 서술은 없는가
     - 기호와 용어가 절마다 달라지지 않는가
     - 장 `index.html`의 "Builds on / Leads to"가 실제 내용과 맞는가
  3. **Part 통합 검토**: Part의 모든 장이 끝나면 한다.
     - §3.1 줄기가 실제로 이어지는가
     - §3.4 호환성 명제가 모두 증명되었는가
     - §3.5 기준 예제의 여정이 빠짐없이 진행되었는가
     - 앞 Part와의 연결 문장과 링크가 있는가
- **번호를 바꾸지 않는다.** 검토 중에도 다른 장이 이미 링크한 항목·식·그림 번호는 바꾸지 않는다. `grep -rn "#i-N-M-" chapters/`로 확인한다. 꼭 바꿔야 하면 바꾸지 말고 보고해서 통합 담당이 전체 링크와 함께 고치게 한다. 새 항목은 절 끝 번호로 추가해도 된다.
- **남은 todo 링크 확정**(§15.5): 선행 장이 쓰였으면 정확한 번호 링크로 바꾼다.
- 검토자는 §16을 기준으로 **수학 오류 → 규약 위반 → 연결·일관성 → 교육적 흐름 → 표현** 순서로 본다.
- 모든 계산을 다시 검산한다. verify 스크립트를 실행하고, 독립적인 검산도 1개 이상 한다.
- 고칠 수 있는 것은 직접 고치고 요약해서 보고한다. 판단이 필요한 것은 `<!-- REVIEW: … -->` 주석으로 남긴다. 검사 도구가 이 주석의 개수를 센다.
- **상태**
  - `planned`: 계획·뼈대만 있음
  - `draft`: 작성 완료
  - `reviewed`: 검토 완료, 검사 통과
  - `final`: 사용자 확인. `final`은 **사용자가 확인한 뒤에만** 붙인다.

---

## 16. 완료 체크리스트

**수학**
- [ ] 모든 정의와 정리에 가정을 명시했고, 선택에 무관함을 확인했다
- [ ] 모든 증명 단계에 근거가 있고, 생략한 증명은 §4.2 허용 목록 안에 있다
- [ ] 모든 비자명한 계산에 verify 스크립트가 있고 PASS한다
- [ ] 부호가 나오는 곳마다 법벡터와 방향을 명시했다

**교육**
- [ ] Goals, Prerequisites, Motivation, 직관 그림, Summary가 있다
- [ ] 정의마다 예와 비예가 있다
- [ ] 기준 예제를 썼고 값이 §7과 일치한다
- [ ] 처음 나오는 계산은 단계를 생략하지 않았다

**규약**
- [ ] 모든 기호가 §6 및 notation.html과 일치하고, 새 기호는 보고했다
- [ ] 교재와 다른 규약에 "Conventions" 상자(`box convention`)를 넣었다
- [ ] 대학 수학 용어를 `tools/terms.tsv`의 영어로 썼고(고등학교 수준의 말은 `terms_keep.txt`대로 한국어), 첫 정의는 `<dfn>English</dfn>(한국어)` 형식이다(§8.1)
- [ ] 라벨·제목(머리 단어, 상자 제목, 소제목, breadcrumb, pager, 상태 표시)이 §8.2의 영어이고, `check_site.py`에 남은 한국어 라벨·참조 경고가 없다

**출처**
- [ ] 대응 교재를 명시했고, 번호는 확인한 것만 적었다
- [ ] 교재 문장이나 그림을 복사하지 않았다

**HTML**
- [ ] 부록 A의 골격, 메타데이터, 네비게이션을 갖췄다
- [ ] 번호·id 규칙을 지켰고 모든 참조 링크가 동작한다
- [ ] 수식 구분자는 `\(` `\[`만 썼고, `<`는 `\lt`로 썼으며, 금지 매크로가 없다
- [ ] `check_all.sh`를 통과했다. 기존 페이지를 변환했다면 `python3 tools/term_guard.py <파일>`도 오류 없이 통과했다

**그림**
- [ ] 스크립트로 재생성되고, assert가 있고, 색 규칙을 지켰고, 그림 안에 한글이 없다
- [ ] 캡션(번호, 설명, 매개변수, 개념도 표시), alt, 본문 참조가 있다

**연결·일관성**
- [ ] 동기의 첫 문단이 앞 절(또는 앞 장)이 끝난 지점에서 출발하고, 요약 끝에 다음 절 예고가 있다
- [ ] 이미 정의된 개념을 다시 정의하지 않았다(`concepts.html` 확인). 일반화는 `data-generalizes`로 표시했다
- [ ] 이 장에 해당하는 §3.4 호환성 명제를 진술하고 증명했다
- [ ] 이 장에 해당하는 §3.5 기준 예제의 칸을 다뤘고, 앞 장의 결과는 인용으로 이어 갔다
- [ ] 예고(`box forward`)한 개념을 정의나 증명에 쓰지 않았다
- [ ] 앞에서 다룬 대상이 다시 나올 때마다 이전 위치를 링크했다

**연습문제**
- [ ] 4–8개이고 난이도를 표시했으며, 모든 풀이가 완전하다

---

## 17. 하지 말 것

- 정의하지 않은 기호나 용어를 쓰기. 뒤 절에서 정의할 것을 앞에서 쓰기
- 교재 번호, 쪽수, 인용문을 지어내기
- 교재 문장을 번역해 옮기기. 교재 그림을 캡처하기
- 이유 없이 "자명하다"를 쓰기. 증명 스케치를 완전한 증명처럼 쓰기
- 한 절 안에서 규약이나 기호를 바꾸기
- 기하적 의미 없이 좌표 계산만 늘어놓기. 반대로 정확한 정의 없이 그림과 직관만 쓰기
- 수식 이미지, 인라인 스타일, 페이지 전용 매크로, `$` 구분자
- 눈대중으로 그린 그림, 본문과 매개변수가 다른 그림, 그림 속 한글
- 병렬 작업 중에 공유 파일을 수정하기
- 처음부터 과하게 일반화하기. 예: 곡면론에서 벡터다발 일반론
- 이미 정의된 개념을 다른 곳에서 다시 정의하기. 같은 내용을 두 곳에 복사하기
- 앞 장의 결과를 인용하지 않고 처음부터 다시 유도하기 (새 관점이 목적이면 `box backward`로 밝힌다)
- 예고만 한 개념을 정의나 증명에 쓰기
- 일반화한 정의를 두고 이전 정의와 일치한다는 증명(호환성 명제)을 빼먹기
- 같은 장의 절을 동시에 나눠 쓰기

---

## 부록 A. 페이지 템플릿

### A-1. 절 페이지

```html
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>7.1 The First Fundamental Form — DG Study</title>
  <meta name="dg-id" content="7.1">
  <meta name="dg-status" content="draft">
  <meta name="dg-prereq" content="6.2, 6.3">
  <meta name="dg-refs" content="dC §2-5; Pr Ch. 6">
  <meta name="dg-updated" content="2026-10-04">
  <link rel="stylesheet" href="../../assets/style.css">
  <script src="../../assets/mathjax-config.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"></script>
</head>
<body>
  <nav class="breadcrumb">
    <a href="../../index.html">Contents</a> ›
    <a href="index.html">7. The First Fundamental Form</a> ›
    <span>7.1 The First Fundamental Form</span>
  </nav>

  <main>
    <header>
      <h1>7.1 The First Fundamental Form</h1>
      <p class="meta">References:
        <a class="cite" href="../../references.html#dC">[dC §2-5]</a>,
        <a class="cite" href="../../references.html#Pr">[Pr Ch. 6]</a></p>
    </header>

    <section class="goals">
      <h2>Goals</h2>
      <ul>
        <li>…</li>
      </ul>
    </section>

    <section class="prereq">
      <h2>Prerequisites</h2>
      <ul>
        <li><a href="../06-regular-surfaces/6-2-tangent-plane.html#i-6-2-1">Definition 6.2.1 (tangent plane)</a></li>
      </ul>
    </section>

    <section id="sec-motivation">
      <h2>Motivation</h2>
      <p>Chapter 6에서 regular surface와 그 tangent plane을 정의했다. 이제 surface <em>위에서</em> 길이와 각을 재고 싶다. …</p>
      <!-- 첫 문단은 앞 절(장의 첫 절이면 앞 장)이 끝난 지점에서 출발한다 (§3.2) -->
    </section>

    <!-- 본문: 직관 → 정의 → 예·비예 → 정리·증명 → 좌표 표현 → 계산 예제. 소제목도 영어로 쓴다 (§8.2) -->

    <section class="summary">
      <h2>Summary</h2>
      <ul>
        <li>…</li>
      </ul>
      <p class="next">다음 절에서는 first fundamental form을 보존하는 map, 즉 isometry를 다룬다. …</p>
    </section>

    <section class="exercises">
      <h2>Exercises</h2>
      <!-- div.exercise … -->
    </section>
  </main>

  <nav class="pager">
    <a rel="prev" href="../06-regular-surfaces/6-5-regular-values.html">← 6.5 Inverse Images of Regular Values</a>
    <a rel="next" href="7-2-isometries.html">7.2 Isometries →</a>
  </nav>
</body>
</html>
```

### A-2. 장 개요 페이지 (`chapters/NN-slug/index.html`)

`<head>`는 A-1과 같지만 `dg-id`는 장 번호(예: `7`)이고, `dg-prereq`와 `dg-refs`는 없다. `dg-status`는 그 장의 모든 절이 `draft` 이상이 될 때까지 `planned`로 둔다. `<main>` 부분은 다음과 같다. Phase 0에서 만든 26개 장 개요가 실제 예다.

```html
  <main>
    <header>
      <h1>7. The First Fundamental Form</h1>
      <p class="threads">Threads: <strong>T3</strong> Intrinsic vs. extrinsic (start) ·
        <strong>T5</strong> Integration and topology (area)</p>
    </header>

    <section class="questions">
      <h2>Questions</h2>
      <ul>
        <li>surface 위에서 길이, 각, 넓이를 어떻게 재는가?</li>
        <li>surface를 늘이지 않고 구부릴 때 변하지 않는 양은 무엇인가?</li>
      </ul>
    </section>

    <section class="inherits">
      <h2>Builds on</h2>
      <ul>
        <li><a href="../06-regular-surfaces/6-2-tangent-plane.html#i-6-2-1">tangent plane</a>과
            <a href="../06-regular-surfaces/6-4-differential-of-a-map.html">differential of a map</a> (Chapter 6)</li>
        <li><a href="../04-curves/4-2-arc-length.html">arc length</a> (Chapter 4): surface 위 곡선의 길이로 확장한다</li>
      </ul>
    </section>

    <h2>Sections</h2>
    <ol class="toc">
      <li id="toc-7-1" data-file="7-1-first-fundamental-form.html">
        <a class="toc-title" href="7-1-first-fundamental-form.html">7.1 The First Fundamental Form</a> <span class="status draft">Draft</span>
        <p class="toc-note">Topics: 정의와 coordinate representation · Running examples: sphere·torus of revolution·cylinder의 E, F, G</p>
      </li>
      <li id="toc-7-2" data-file="7-2-isometries.html">
        <span class="toc-title">7.2 Isometries</span> <span class="status planned">Planned</span>
        <p class="toc-note">…</p>
      </li>
    </ol>

    <section class="handoff">
      <h2>Leads to</h2>
      <ul>
        <li>E, F, G → Chapter 8 (curvature 공식), Chapter 9 (Christoffel symbols, Theorema Egregium), Chapter 16·21 (Riemannian metric과의 compatibility)</li>
        <li>isometry → Chapter 9 (Theorema Egregium의 진술), Chapter 21</li>
        <li>area element dA → Chapter 10 (Gauss–Bonnet theorem), Chapter 18 (volume form의 적분)</li>
        <li>Open question: first fundamental form만으로 surface의 "휘어짐"을 알 수 있는가? → Chapter 9</li>
      </ul>
    </section>
  </main>
```

---

## 부록 B. 모범 예시 (문체와 설명 밀도의 기준)

아래는 `7-1-first-fundamental-form.html` 본문의 일부다. **설명의 촘촘함, 계산 단계, 구성요소 사용법**의 기준으로 삼는다. 라벨·용어는 영어, 문장은 한국어다(§8).

```html
<section id="sec-definition">
  <h2>Definition</h2>
  <p>surface 위에서 길이와 각을 재려면 각 tangent plane 위에 inner product가 있어야 한다.
  가장 자연스러운 방법은 \(\R^3\)의 inner product를 tangent plane \(T_pS\)의 벡터에만 적용하는 것이다.</p>

  <div class="thm definition" id="i-7-1-1">
    <p><span class="thm-head">Definition 7.1.1</span>
    <span class="thm-name">(first fundamental form)</span>.
    \(S\)가 regular surface이고 \(p \in S\)일 때, quadratic form
    \[ \mathrm{I}_p\colon T_pS \to \R, \qquad \mathrm{I}_p(w) = \inner{w}{w} = \abs{w}^2 \]
    을 \(p\)에서 \(S\)의 <dfn>first fundamental form</dfn>(제1기본형식)이라 한다.</p>
  </div>

  <p>정의에는 parametrization이 전혀 쓰이지 않았다. 따라서 \(\mathrm{I}_p\)는 surface 자체에 속한 대상이다.
  계산할 때는 parametrization \(\mathbf{x}\colon U \to S\)로 coordinate representation을 얻는다.
  \(w \in T_pS\)는 \(w = a\,\mathbf{x}_u + b\,\mathbf{x}_v\) 꼴로 유일하게 쓸 수 있다
  (<a href="../06-regular-surfaces/6-2-tangent-plane.html#i-6-2-4">Proposition 6.2.4</a>).
  inner product는 bilinear이고 symmetric이므로</p>
  <div class="equation" id="eq-7-1-1">
  \[ \mathrm{I}_p(w) = E\,a^2 + 2F\,ab + G\,b^2, \qquad
     E = \inner{\mathbf{x}_u}{\mathbf{x}_u},\quad F = \inner{\mathbf{x}_u}{\mathbf{x}_v},\quad G = \inner{\mathbf{x}_v}{\mathbf{x}_v}. \tag{7.1.1} \]
  </div>
  <p>이다. 여기서 \(E, F, G\)는 \(q = \mathbf{x}^{-1}(p)\)에서 계산한 값이다.</p>
</section>

<div class="thm example" id="i-7-1-2">
  <p><span class="thm-head">Example 7.1.2</span> <span class="thm-name">(sphere)</span>.
  반지름 \(r\)인 sphere의 standard parametrization
  \[ \mathbf{x}(\theta,\varphi) = (r\sin\theta\cos\varphi,\ r\sin\theta\sin\varphi,\ r\cos\theta),
     \qquad 0 \lt \theta \lt \pi,\quad 0 \lt \varphi \lt 2\pi \]
  를 쓰자. partial derivative는
  \[ \begin{aligned}
     \mathbf{x}_\theta &= (r\cos\theta\cos\varphi,\ r\cos\theta\sin\varphi,\ -r\sin\theta),\\
     \mathbf{x}_\varphi &= (-r\sin\theta\sin\varphi,\ r\sin\theta\cos\varphi,\ 0)
  \end{aligned} \]
  이다. 따라서
  \[ \begin{aligned}
     E &= r^2\cos^2\theta\,(\cos^2\varphi + \sin^2\varphi) + r^2\sin^2\theta = r^2,\\
     F &= -r^2\sin\theta\cos\theta\sin\varphi\cos\varphi + r^2\sin\theta\cos\theta\sin\varphi\cos\varphi + 0 = 0,\\
     G &= r^2\sin^2\theta\,(\sin^2\varphi + \cos^2\varphi) = r^2\sin^2\theta.
  \end{aligned} \]
  \(E = r^2\)는 meridian(\(\varphi\) 고정)을 따라 \(\theta\)가 \(d\theta\)만큼 변할 때
  길이가 \(r\,d\theta\)만큼 변한다는 뜻이다.
  \(G = r^2\sin^2\theta\)는 circle of latitude의 반지름이 \(r\sin\theta\)라는 사실을 반영한다
  (<a href="#fig-7-1-1">Figure 7.1.1</a>).</p>
  <p class="verified">Verification: <code>verify/v-7-1-first-form.py</code></p>
</div>

<aside class="box warning">
  <p class="box-title">Warning</p>
  <p>\(F = 0\)은 두 coordinate curve가 서로 orthogonal이라는 뜻이다. 이것은 <em>parametrization</em>의 성질이지
  surface의 성질이 아니다. 같은 sphere라도 다른 parametrization을 쓰면 \(F \neq 0\)일 수 있다
  (<a href="#ex-7-1-3">Exercise 7.1.3</a>).</p>
</aside>

<aside class="box convention">
  <p class="box-title">Conventions</p>
  <p>do Carmo는 second fundamental form의 coefficient를 \(e, f, g\)로 쓴다
  <a class="cite" href="../../references.html#dC">[dC §3-3]</a>.
  이 study set은 \(g\)를 Riemannian metric 전용으로 남겨 두려고 \(L, M, N\)을 쓴다.
  즉 \(e = L,\ f = M,\ g = N\)이다. (Chapter 8에서 쓰는 형식의 예)</p>
</aside>
```

---

## 부록 C. `assets/mathjax-config.js` (정본)

이 파일은 브라우저와 `tools/check_math.mjs`가 **함께** 읽는다. 매크로를 추가하거나 바꾸면 이 부록과 파일을 **동시에** 고친다. 통합 담당만 할 수 있다.

```js
// DG-study 공용 MathJax 설정. 매크로 목록은 GUIDELINES.md 부록 C와 같아야 한다.
window.MathJax = {
  loader: { load: ['[tex]/amscd', '[tex]/boldsymbol'] },
  tex: {
    packages: { '[+]': ['amscd', 'boldsymbol'] },
    inlineMath: [['\\(', '\\)']],
    displayMath: [['\\[', '\\]']],
    tags: 'none',                       // 번호는 \tag{N.M.K}로 직접 붙인다
    macros: {
      R: '\\mathbb{R}',
      Z: '\\mathbb{Z}',
      Q: '\\mathbb{Q}',
      RP: '\\mathbb{RP}',
      inner: ['\\langle #1,\\, #2 \\rangle', 2],   // \inner{v}{w}_g
      abs: ['\\lvert #1 \\rvert', 1],
      norm: ['\\lVert #1 \\rVert', 1],
      pd: ['\\frac{\\partial #1}{\\partial #2}', 2],
      Lie: '\\mathcal{L}',
      id: '\\operatorname{id}',
      tr: '\\operatorname{tr}',
      rank: '\\operatorname{rank}',
      supp: '\\operatorname{supp}',
      Alt: '\\operatorname{Alt}',
      Sym: '\\operatorname{Sym}',
      Hom: '\\operatorname{Hom}',
      End: '\\operatorname{End}',
      GL: '\\operatorname{GL}',
      grad: '\\operatorname{grad}',
      divg: '\\operatorname{div}',
      Hess: '\\operatorname{Hess}',
      Rm: '\\operatorname{Rm}',
      Rc: '\\operatorname{Rc}',
      inj: '\\operatorname{inj}',
      vol: '\\operatorname{vol}',
      sgn: '\\operatorname{sgn}',
      im: '\\operatorname{im}',
      Span: '\\operatorname{span}',     // \\span은 TeX 원시 명령이라 쓰지 않는다
      diag: '\\operatorname{diag}'
    }
  }
};
```

- `\div`는 TeX에 이미 있는 기호(÷)이므로 발산은 `\divg`로 쓴다.
- `\pd{}{x^i}`처럼 첫 인자를 비우면 연산자 ∂/∂xⁱ가 된다.

---

## 부록 D. Phase 0: 기반 구축 작업 목록

콘텐츠 작성 전에 통합 담당(또는 지정된 agent 하나)이 한 번 수행한다. **2026-09-30에 완료했다.** 아래는 각 도구가 무엇을 하는지에 대한 명세로 남겨 둔다.

1. **`assets/style.css`**: §11.3의 클래스를 모두 지원한다.
   - 본문 글꼴 `"Noto Serif KR", "Nanum Myeongjo", serif`, 최대 폭 약 46rem, 줄간격 1.8
   - `.thm`의 종류별로 왼쪽 테두리 색을 구분한다.
   - `.proof`의 마지막 문단 끝에 ∎를 붙인다.
   - `details`의 접힘 스타일
   - `figure` 가운데 정렬, `max-width: 100%`
   - `.todo`는 눈에 띄는 배경색
   - 인쇄용 `@media print`: details를 펼치고 네비게이션을 숨긴다.
   - 좁은 화면 대응. 긴 디스플레이 수식은 가로 스크롤(`mjx-container[display] { overflow-x: auto; }`)
2. **`assets/mathjax-config.js`**: 부록 C 그대로 만든다.
3. **`tools/dgfig.py`**
   - `setup()`: 백엔드 Agg, 부록 E의 mpl_toolkits 우회, rcParams 설정
     - `mathtext.fontset="cm"`, `svg.fonttype="path"`, 글꼴 크기, 선 굵기
     - `font.family=["DejaVu Serif", "Noto Serif CJK JP"]`
   - `COLORS`: §12.3의 표. 키는 `main`, `surface`, `tangent`, `normal`, `third`, `accent`, `region`, `aux`
   - `save(fig, __file__)`: 스크립트 옆에 같은 이름의 `.svg`로 저장한다.
   - 헬퍼: `arrow3d(ax, base, vec, role, label=None)`, `tangent_plane(ax, p, e1, e2, size)`, `surface(ax, X, Y, Z)`, `blob(ax, center, radius, seed)`(개념도용 다양체), `map_arrow(fig, ax_from, ax_to, label)`(두 패널 사이의 사상 화살표)
4. **`tools/dgsym.py`**: sympy로 §6 규약을 구현한다.
   - 곡선: `curvature_torsion(gamma, t)`, `signed_curvature(gamma, t)`(평면곡선의 κ_s, n_s = J t 규약)
   - 곡면: `first_ff(X, u, v)`, `unit_normal(X, u, v)`, `second_ff(X, u, v)` → (L, M, N), `K_H(X, u, v)`
   - 리만: `christoffel(g, coords)` → `Γ[k][i][j]`, `riemann(g, coords)` → `R[i][j][k][l]` = R_{ijk}^l(Lee 규약), `riemann_lower`, `ricci`, `scalar`, `sectional(g, coords, v, w)`
   - `EXAMPLES`: §7 기준 예제의 **유일한 코드 정의**다.
     - 각 항목에 매개화 또는 계량(sympy 식), 좌표 기호와 가정(예: θ ∈ (0, π)), 법벡터 방향, §7의 기대값을 담는다.
     - verify 스크립트와 그림 스크립트는 모두 이것을 가져다 쓴다(§3.6).
   - 차트 예제: `stereographic(n)`(두 입체사영, 역사상, 좌표변환 u/|u|²), `rpn_charts(n)`(ℝℙⁿ의 φ_i, 역사상, 좌표변환). `EXAMPLES["stereographic"]`, `EXAMPLES["rpn"]`은 n = 2인 경우다.
   - 자체 테스트 `python3 tools/dgsym.py`
     - §7의 모든 값을 재현한다.
     - S²(r)의 K = 1/r²를 곡면 공식과 리만 곡률 공식, **두 경로로** 확인한다(부호 규약 교차검증).
     - S³에서 Rc = 2g/r², S = 6/r²를 확인한다.
5. **`tools/dgnum.py`**: `rk4(f, y0, ts)` 같은 수치 헬퍼. scipy를 쓸 수 없으므로 필요하다.
   - **`tools/dgcheck.py`**: verify 스크립트용 헬퍼
     - `check(name, cond)`
     - `sym_equal(name, a, b)`: 기호적 단순화에 더해 무작위 수치 대입으로 확인
     - `close(name, a, b, tol)`
     - `summary()`: 실패가 있으면 exit 1
6. **`tools/check_site.py`**: Python 표준 라이브러리 `html.parser`만 쓴다. 검사 항목은 다음과 같다.
   - 필수 meta와 골격(style.css, config, MathJax 로드 순서)
   - id 중복, 내부 링크와 앵커의 존재
   - `img`와 `iframe` 대상 파일의 존재, 그림에 대응하는 `.py` 소스의 존재
   - 번호 연속성, id와 표시 번호의 일치
   - `$` 구분자 사용, 원문 기준 수식 안의 `<`
   - 인라인 style, 페이지 `<style>`, 허용되지 않은 `<script>`
   - `todo`와 `REVIEW` 개수 보고
   - 라벨(§8.2): `thm-head`·연습·그림 머리는 영어 단어와 번호(`Definition 6.2.1`, `Exercise 6.2.1`, `Figure 6.2.1.`), toc 상태 라벨은 Planned·Draft·Reviewed·Final이어야 한다(오류)
   - 남은 한국어(경고): 머리·상자 제목·summary·소제목·breadcrumb·pager·상태·`p.meta`·`p.threads`·`p.toc-note`·`p.verified`가 한국어 라벨 단어(정의, 증명, 힌트, 직관, 이 절의 목표, 전체 목차, 대응 교재, 초안 …)로 시작함, 제목(h1–h3, toc-title, breadcrumb, pager, `<title>`)에 한글이 있음, 본문의 한국어 번호 참조(`정리 N.M.K`, `식 (N.M.K)`, `N장`, `N.M절`)
   - 링크 없는 상호참조(경고): `Theorem N.M.K` 같은 영어 라벨 + 번호, 수식 밖의 `(N.M.K)`
   - 연결 장치
     - 절 페이지마다 `p.next`가 있는지
     - 장 `index.html`에 `inherits`, `questions`, `handoff`, `threads`가 있는지
7. **`tools/build_index.py`**: 모든 장의 HTML을 읽어 `concepts.html`을 만든다(§3.6).
   - 수집하는 것
     - `<dfn>`: 용어, 정의 위치(가장 가까운 id), `data-generalizes`
     - 번호 항목(`div.thm`): 번호, 제목, 위치
     - 링크 그래프
   - 용어는 `<dfn>`의 영어, 번역어는 바로 뒤 괄호 안의 한국어다. 옛 형식 `<dfn>한국어</dfn>(English)`도 읽으며, 이때는 한글이 없는 쪽을 영어 용어로 본다.
   - `concepts.html`의 내용(영어): 영어 용어의 A–Z 묶음(글자가 아닌 것은 "#"), 대소문자 무시 알파벳순. 항목마다 용어, (한국어), 정의 위치, Generalizes / Generalized by, **Used in** = 역참조. 끝에 Compatibility propositions 목록
   - 오류로 처리하는 것
     - `data-generalizes` 없이 같은 영어 용어(대소문자 무시)의 `<dfn>`이 두 번 나옴. 한국어 번역어가 모두 다르면 동음이의어로 보고 경고만 한다
     - `data-generalizes`가 가리키는 대상이 없음
     - §3.4 표에 있는 장에 `compat` 항목이 없음
8. **`tools/check_math.mjs`** + `tools/package.json`(의존성 `mathjax-full@3.2.2`)
   - HTML에서 `\(…\)`와 `\[…\]`를 추출하고 HTML 엔티티를 해제한다.
   - `mathjax-config.js`를 `node:vm`에서 가짜 `window` 객체로 실행해 **같은 매크로**를 얻는다.
   - TeX 입력 패키지는 `['base','ams','amscd','boldsymbol','newcommand','configmacros']`만 쓰고, `formatError`에서 예외를 던지게 한다.
   - 정의되지 않은 매크로와 괄호 오류를 `파일:줄`과 함께 보고한다.
   - 이 방식은 2026-09-30에 이 환경에서 동작을 확인했다. `noundefined`와 `autoload` 패키지를 빼야 정의되지 않은 매크로가 오류로 잡힌다.
9. **`tools/check_all.sh`**: 다음을 모두 실행하고, 하나라도 실패하면 exit 1로 끝낸다.
   - check_site.py, build_index.py, check_math.mjs
   - 모든 `figures/*.py` (재생성)
   - 모든 `verify/*.py`
   - `python3 tools/dgsym.py`
10. **독자용 공용 페이지**
   - `index.html`: 로드맵 표와 절 상태
   - `notation.html`: §6을 독자용으로
   - `glossary.html`: §8
   - `references.html`: §5. 항목 id는 약어
11. **장 폴더와 장 `index.html` 뼈대**: 부록 A-2 형식으로 만든다.
    - 절 목록은 §9 기준으로 채운다.
    - "이어받는 것 / 이 장의 질문 / 넘겨주는 것 / 줄기"는 §3.1–3.5에서 초안을 채운다.
    - 각 장 담당 agent가 이 초안을 다듬는다.
12. (선택) 장 의존 그래프 `assets/roadmap.svg`(graphviz). §3.1 줄기별로 색을 구분한다.
13. **`tools/term_guard.py`** (2026-10-04 추가, 영어 용어 전환용): `python3 tools/term_guard.py [--rev HEAD] [--quiet] 파일…`. 각 파일을 `git show REV:파일`과 비교한다(`--against 옛파일`로 git 없이도 비교).
    - 오류: 요소 id 집합이 바뀜, 수식 `\(…\)`·`\[…\]`의 다중집합이 바뀜(공백 차이는 무시, `	ext{…}`·`\mbox{…}` 안의 한글만 바뀐 것은 NOTE), `div.thm`·`figure.dg-figure`·`div.exercise`·`div.equation`·`img` 개수가 바뀜, `href`·`src`·`data-generalizes`·`data-file` 대상이 바뀜(링크 글자는 바뀌어도 된다)
    - 경고: check_site와 같은 남은 한국어 라벨, `terms.tsv`의 한국어 용어가 본문·`<title>`·`alt`에 남음(`</dfn>` 바로 뒤 괄호, `terms_keep.txt`의 낱말, code/pre 제외. 한 글자 용어 상·핵·공·틀은 조사가 붙은 독립된 낱말만)

---

## 부록 E. 환경 메모 (2026-09-30 확인)

- **설치되어 있는 것**
  - Python 3.10, numpy, sympy 1.14, matplotlib 3.9.1(사용자 site-packages), plotly 6.0
  - graphviz `dot`
  - Node 22과 npm (레지스트리 접근 가능)
  - 한글이 포함된 Noto Sans/Serif CJK JP 글꼴
- **LaTeX는 없다.** matplotlib의 usetex, tikz, pdflatex를 쓸 수 없다.
- **scipy는 import할 수 없다**(권한 오류). 수치 ODE는 `tools/dgnum.py`를 쓴다.
- ⚠ **mplot3d 문제**: 시스템의 오래된 `mpl_toolkits`(matplotlib 3.5용)가 사용자 matplotlib 3.9의 것을 가린다. 그래서 `projection="3d"`가 `Unknown projection '3d'`로 실패한다. 3D를 쓰기 **전에** 아래 코드를 실행해야 하며, `dgfig.setup()`에 포함한다.
  ```python
  import os, matplotlib, mpl_toolkits
  mpl_toolkits.__path__ = [os.path.join(os.path.dirname(os.path.dirname(matplotlib.__file__)), "mpl_toolkits")]
  ```
- **mathtext와 한글**: mathtext(`$…$`)와 한글을 한 문자열에 섞으면 한글이 깨진다. 한글만 있는 문자열은 정상으로 나온다.
- **SVG를 눈으로 확인할 때**: ImageMagick `convert`는 matplotlib SVG의 채우기와 clip path를 잘못 그린다. headless Chrome을 쓴다.
  ```bash
  google-chrome --headless=new --no-sandbox --screenshot=out.png --window-size=900,700 file:///절대경로/그림.svg
  ```
  PNG를 자를 때는 `convert big.png -crop 900x2300+0+0 +repage part.png`를 써도 된다. PNG에는 문제가 없다.
  단, ImageMagick은 높이가 16000px을 넘는 PNG를 읽지 못한다(정책 제한). 긴 페이지는 Pillow로 자른다:
  `python3 -c "from PIL import Image; Image.MAX_IMAGE_PIXELS=None; im=Image.open('big.png'); im.crop((0,y0,im.size[0],y1)).save('part.png')"`
- **HTML 페이지를 눈으로 확인할 때**: `node tools/shot.mjs /절대경로/페이지.html out.png [폭]`
  - `google-chrome --screenshot`은 MathJax 조판을 기다리지 않아 수식이 원문 TeX로 찍힌다(01장 파일럿에서 확인).
  - `tools/shot.mjs`는 DevTools 프로토콜로 `MathJax.startup.promise`를 기다린 뒤 전체 페이지를 찍는다. MathJax 오류 개수와 **넘치는 디스플레이 수식의 id**도 출력한다.
  - 폰 폭은 `400`으로 확인한다. 네트워크가 필요하다(MathJax CDN).
- **sympy 양수 가정**: 구면의 sin θ, 원환면의 R + r cos u처럼 양수임을 알려 줘야 sqrt(sin²θ) 같은 식이 남지 않는다. `dgsym.EXAMPLES[…]["positive"]`에 들어 있다.
