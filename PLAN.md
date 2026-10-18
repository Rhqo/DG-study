# DG-study 진행 계획 (통합 담당용)

- 작성일: 2026-09-30
- 이 문서는 **통합 담당(orchestrator)**, 즉 사용자와 대화하는 메인 세션이 보는 문서다. 작성·검토 agent는 GUIDELINES.md를 따른다.
- 세션이 바뀌어도 이 문서의 "진행 기록"(§7)부터 읽으면 이어서 진행할 수 있다.

---

## 1. 현재 상태

**Phase 0(기반 구축) 완료, 2026-09-30.** `bash tools/check_all.sh`를 통과했다.

| 구분 | 파일 | 상태 |
|---|---|---|
| 공용 설정 | `assets/style.css`, `assets/mathjax-config.js` | 완료 |
| 계산·그림 | `tools/dgsym.py`(자체 테스트 63개), `tools/dgnum.py`(21개), `tools/dgcheck.py`, `tools/dgfig.py`(12개) | 완료 |
| 검사 | `tools/check_site.py`, `tools/build_index.py`, `tools/check_math.mjs`, `tools/check_all.sh`(전체 / `--chapter NN`), 검사기 테스트 57개 | 완료 |
| 독자용 페이지 | `index.html`, `notation.html`, `glossary.html`, `references.html`, `concepts.html`(자동 생성) | 완료 |
| 장 개요 | `chapters/NN-slug/index.html` × 26 (`planned`, 절 계획 총 141개) | 완료 |
| 모범 페이지 | `tools/tests/fixtures/site/` (모든 구성요소를 쓴 올바른 절 페이지) | 완료 |

---

## 2. 역할

| 역할 | 누가 | 하는 일 | 수정할 수 있는 파일 |
|---|---|---|---|
| 통합 담당 | 메인 세션 | 웨이브 시작, 보고 취합, 공유 파일 반영, 전체 검사, 사용자 보고, 이 문서 갱신 | 공유 파일 전부 (GUIDELINES.md의 §6·§7은 사용자 승인 후) |
| 장 작성 agent | 새 general-purpose agent, 장마다 1명 | 한 장의 절을 **순서대로** 작성: 본문, 그림, verify, 장 개요 갱신 | 담당 장 폴더 안 |
| 장 검토 agent | 작성자와 다른 새 agent, 장마다 1명 | 절 검토와 장 통합 검토(§15.6의 1–2단계), 직접 수정, `REVIEW` 주석 | 담당 장 폴더 안 |
| Part 검토 agent | 새 agent, Part마다 1명 | Part 통합 검토(§15.6의 3단계): 줄기, 호환성 명제, 예제 여정, Part 사이 연결 | 읽기 전용. 수정 목록만 보고 |
| 사용자 | — | 체크포인트에서 검토, `final` 승인, 규약 변경 승인 | — |

- agent는 fork가 아니라 **새 agent**로 띄운다. 이 대화의 맥락은 필요 없고, GUIDELINES.md와 저장소 파일이 곧 명세다.
- 동시에 도는 작성 agent는 **최대 3명**이다.

---

## 3. 장 하나의 파이프라인

```
작성 agent ──▶ 장 검토 agent ──▶ 통합 담당 ──▶ (체크포인트면) 사용자
 절 순서대로     절 검토 +          공유 파일 반영        검토 → final
 draft          장 통합 검토       전체 check_all
                reviewed          스크린샷 확인
```

1. **작성**
   - 절 계획(장 `index.html`) 순서대로 쓴다.
   - `bash tools/check_all.sh --chapter NN`을 통과시킨다.
   - 절의 상태를 `draft`로 올리고, 장 개요의 "이어받는 것 / 넘겨주는 것"을 실제 내용에 맞게 고친다.
2. **검토**
   - 모든 계산을 다시 검산한다. verify를 재실행하고, 절마다 독립 검산을 1개 이상 한다.
   - 고칠 수 있는 것은 직접 고친다. 판단이 필요한 것은 `<!-- REVIEW: … -->`로 남긴다.
   - 문제가 없는 절은 `reviewed`로 올린다.
3. **통합**
   - 보고에 적힌 새 기호·용어·참고문헌을 `notation.html`, `glossary.html`, `references.html`에 반영한다.
   - `index.html`의 상태 표시와 GUIDELINES.md §3.5 표를 갱신한다(규약이 아닌 계획 변경만).
   - 전체 `check_all.sh`를 돌리고, headless Chrome으로 절 몇 개를 스크린샷으로 확인한다.
   - §7 진행 기록을 갱신한다.
4. **선행 조건**: 다음 웨이브의 장은 선행 장이 모두 **`reviewed`가 된 뒤에** 시작한다. 검토에서 번호나 정의가 바뀔 수 있기 때문이다.

---

## 4. 웨이브 계획

의존 관계는 각 장 개요의 "이어받는 것" 링크에서 뽑았다. 대부분의 뒤쪽 의존은 "관계"(호환성) 절에만 걸린다. 그런 절은 **보류 절**로 두고 나중에 쓰면 Part I과 Part II를 나란히 진행할 수 있다.

| 웨이브 | 장 (동시 작업) | 보류 절 / 비고 | 체크포인트 |
|---|---|---|---|
| 1 | 01 선형대수 | **파일럿**: 문체, 밀도, 분량, 토큰 비용을 측정한다 | ✅ 사용자 검토 → 필요하면 가이드라인 조정 |
| 2 | 02 미적분, 03 위상 | | Part 0 검토 |
| 3 | 04 곡선, 06 정칙곡면 | 첫 미분기하 장. 그림과 설명 방식을 확인한다 | ✅ 사용자 검토 |
| 4 | 05 공간곡선, 07 제1기본형식, 11 매끄러운 다양체 | 11.6 "Part I과의 관계"는 09장 이후로 보류 | |
| 5 | 08 가우스 사상, 12 매끄러운 사상 | | |
| 6 | 09 내재기하, 13 접공간 | | |
| 7 | 10 가우스-보네, 14 부분다양체, 15 벡터장 (+11.6) | | ✅ Part I 검토, 사용자 검토 |
| 8 | 16 여벡터와 텐서 | | |
| 9 | 17 미분형식 | | |
| 10 | 18 적분·스토크스, 20 리 군(선택) | | |
| 11 | 19 드람, 21 리만 계량 | | ✅ Part II 검토, 사용자 검토 |
| 12 | 22 접속 | | |
| 13 | 23 측지선 | | |
| 14 | 24 곡률 | | |
| 15 | 25 리만 부분다양체 | | |
| 16 | 26 가우스-보네 재방문(선택) | | ✅ Part III 검토, 최종 검토 |

- Part III(21–26)는 장마다 앞 장을 모두 쓰므로 순차 진행밖에 할 수 없다.
- 선택 장 20과 26은 사용자 결정에 따라 뺄 수 있다.
- 웨이브를 돌리는 방법은 두 가지다.
  - **기본**: 통합 담당이 Agent 도구로 장마다 작성 agent를 띄우고, 끝나면 검토 agent를 띄운다.
  - **선택**: 사용자가 "workflow로 진행"을 명시적으로 요청하면 Workflow 도구로 웨이브 하나(작성 → 검토 파이프라인)를 한 번에 돌린다.

---

## 5. agent 프롬프트 템플릿

### 5.1 장 작성 agent

```
너는 /home/kbh2/DG-study 미분기하 study set의 {NN}장 「{제목}」 작성 담당이다.

1. GUIDELINES.md 전체와 CLAUDE.md를 읽어라. 이 문서가 모든 규칙의 정본이다.
2. 다음을 읽어라:
   - chapters/{폴더}/index.html (절 계획, 이어받는 것, 넘겨주는 것)
   - 이어받는 장들의 index.html과 이미 쓰인 절 전부
   - tools/tests/fixtures/site/ (모든 구성요소가 쓰인 올바른 페이지 예)
   - notation.html, glossary.html, concepts.html
   - refs/ 에 대응 교재가 있으면 해당 부분
3. 절 계획 순서대로 {절 목록}을 쓴다. 앞 절을 마친 뒤 다음 절로 간다. {보류 절이 있으면: "{절}은 쓰지 말고 계획 상태로 둔다."}
4. 절마다 figures/ 그림 스크립트와 verify/ 검증 스크립트를 만들고 실행한다.
   (PYTHONPATH=tools, dgfig/dgsym/dgnum/dgcheck, 기준 예제는 dgsym.EXAMPLES)
5. 장 index.html을 고친다:
   - 쓴 절을 링크로 바꾸고 상태를 draft로 올린다.
   - "이어받는 것 / 넘겨주는 것"을 실제 내용에 맞게 고친다.
   - 모든 절이 draft가 되면 장의 dg-status를 draft로 올린다.
6. bash tools/check_all.sh --chapter {NN}을 통과시킨다.
   가능하면 headless Chrome으로 각 절을 스크린샷해 렌더링을 확인한다 (GUIDELINES.md 부록 E).
7. 담당 장 폴더 밖의 파일은 수정하지 않는다.
   교재(특히 refs/의 dC) 연습문제를 쓸 때는 대상·조건·묻는 것을 바꾸어 실질적으로 변형한다. 숫자만 바꾸거나 번역만 하지 않는다(§4.4).
8. 최종 보고 (GUIDELINES.md §15.4). 특히 다음을 적는다:
   - 공유 파일 반영 요청 (새 기호, 새 용어, 새 참고문헌, §3.5 표 변경)
   - [확인 필요] 목록
   - 다음 장이 알아야 할 것
   - 사용한 참고자료
```

### 5.2 장 검토 agent

```
너는 /home/kbh2/DG-study {NN}장 「{제목}」의 검토 담당이다. 작성자와 독립적으로 판단한다.

1. GUIDELINES.md 전체를 읽어라. 특히 §15.6(검토), §16(체크리스트), §3(연결·일관성)을 본다.
2. 장의 모든 절을 처음부터 끝까지 한 번에 읽는다.
   검토 순서: 수학 오류 → 규약 위반 → 연결·일관성 → 교육적 흐름 → 표현
3. 검산:
   - 모든 verify 스크립트를 다시 실행한다.
   - 절마다 작성자 스크립트와 독립적인 검산을 1개 이상 새로 해 본다 (scratchpad에서, sympy 등).
4. 그림: 스크립트를 다시 실행하고 PNG로 렌더링해서 눈으로 확인한다 (부록 E).
   본문과 기호·매개변수가 같은지 본다.
5. 고칠 수 있는 것은 직접 고친다. 판단이 필요한 것은 <!-- REVIEW: … --> 주석으로 남긴다.
6. bash tools/check_all.sh --chapter {NN}을 통과시킨다. 문제가 없는 절은 dg-status를 reviewed로 올린다.
7. 담당 장 폴더 밖의 파일은 수정하지 않는다.
8. 최종 보고:
   - 발견한 문제와 수정 내용 (심각도 순)
   - 남은 REVIEW 주석
   - 작성자에게 되돌려야 할 것
   - 공유 파일 반영 요청
```

### 5.3 Part 검토 agent

```
너는 /home/kbh2/DG-study Part {P}({장 범위}) 통합 검토 담당이다. 파일을 수정하지 말고 보고만 한다.

GUIDELINES.md §15.6의 3단계를 따라 다음을 확인한다:
- §3.1 줄기가 실제로 이어지는가
- §3.4 호환성 명제가 모두 진술·증명되었는가
- §3.5 기준 예제 여정이 빠짐없이 진행되었는가
- 앞 Part와의 연결 문장과 링크가 있는가
- 장마다 기호·용어·문체·밀도가 일관되는가
- concepts.html에 중복되거나 빠진 정의가 없는가

보고: 문제 목록 (장·절·위치, 심각도, 수정 제안)
```

---

## 6. 비용과 일정 (추정 — 파일럿 후 보정)

- **01장 파일럿 실측**: 작성 agent가 6절에 약 71만 토큰, 78분, 도구 호출 195회를 썼다. 절당 약 12만 토큰, 13분(병렬 없이 단독 기준)이다.
- 이 실측으로 보정하면 장 하나(작성 + 검토)는 **약 100만 토큰**, 절 141개 전체는 **약 2,500만 토큰**이다.
- ⚠ 동시에 agent 4–5명을 돌리면 사용자 계정의 세션 사용 한도에 걸린다(2026-09-30에 한 번 걸려 02–04가 중단됐다가 재개됨). 동시 작성 agent는 3명 안팎으로 유지하고, 기계적인 작업(공유 파일 반영 등)은 가벼운 모델로 돌린다.
- Phase 0 agent 하나는 약 35만 토큰, 10–16분을 썼다.
- 장 하나(작성 + 검토)는 **대략 50만–150만 토큰**으로 추정한다. 절 5–7개, 그림과 verify를 포함한 추정이다. 26장이면 **대략 1,500만–3,500만 토큰**이다.
- 이 추정은 불확실하다. 파일럿(웨이브 1)에서 실제 값을 재고 이 표를 고친다.
- 웨이브 하나(작성 + 검토 + 통합)의 시간은 추정하지 않았다. 파일럿에서 잰다.

---

## 7. 진행 기록

| 장 | 상태 | 작성 | 검토 | 사용자 | 비고 |
|---|---|---|---|---|---|
| 01 | **reviewed** | 완료: 6절, 그림 7, 검증 205개 PASS | 완료: 독립 검산 47개, 엄밀성 보완 약 10곳, 번호 변경 없음 | 파일럿 확인 요청 | 웨이브 1 |
| 02 | **reviewed** | 완료: 7절, 그림 7, verify 7(PASS 255) | 완료: 독립 검산 52개, Part 0 예외 규칙 적용, 허용 밖 앞 참조 2곳 제거 | — | |
| 03 | **reviewed** | 완료: 7절(3.3을 부분공간·곱 / 몫으로 나눔), 그림 14, 검증 107 PASS | 완료: 독립 검산 34개, 링크 없는 참조 98개 연결, 번호 변경 없음 | — | |
| 04 | **reviewed** | 완료: 5절, 그림 10, verify 5(PASS 198) | 완료: 수학 오류 2개(곡선 모양 서술, 연습의 공역) 수정, 엄밀성·기호 충돌 정리, todo 확정 | 체크포인트(첫 미분기하 장) | |
| 05 | **reviewed** | 완료: 7절(5.7 선택), 그림 13, verify 7 | 완료: 연습 풀이 서술 오류 1, 캡션 수치 오류 1, I/I_n 규약, 정리 5.6.11 유일성 한 줄 보완. 명제 5.6.12(양의 방향 ⇒ rot = +1, 10장용) 추가 완료(verify 109 PASS), 그림 5.3.2 색 수정, 그림 5.6.4 신설 | — | |
| 06 | **reviewed** | 완료: 5절, 그림 11, verify 5(PASS 164) | 완료: 부호 오류 1개, 증명 부정확 2곳, 잘못된 링크, **dC 연습 거의 복사 7문항 변형**, 기호 충돌 정리 | 체크포인트(첫 곡면 장) | |
| 07 | **reviewed** | 완료: 5절, 그림 9, verify 5 | 완료: dC 근접 복사 1문항 변형, 04장과 중복된 연습 1문항 교체, 반례 교체, 기호 충돌 정리, 독립 검산 52개 | — | |
| 08 | draft | 완료: 5절, 그림 8, verify 5(PASS 222), todo 0 | **다음 세션에서 검토 시작** | — | |
| 09–10 | planned | — | — | — | |
| 11 | 11.1–11.5 **reviewed** (장은 planned) | 완료: 5절, 그림 9, verify 5(PASS 199+2) | 완료: 연습 11.3.5 풀이 오류(호환의 비추이성) 수정, 증명 공백 4곳 보완, 인용 약 90개 확인 | — | 11.6은 06·09 이후 |
| 12 | **reviewed** | 완료: 5절(12.5 응용 신설), 그림 11, verify 5(PASS 363), compat 12.1.11 | 완료: **명제 12.2.3 순환 증명 수정**, 빈틈 13곳 보완, 기호 충돌(k→λ 등), 우리손 보조정리 용어 | — | todo 2개(11.6 대기) |
| 13 | planned | 작성 agent를 시작했다가 사용자 지시로 곧바로 중지함. 산출물 없음, 처음부터 다시 시작 | — | — | 웨이브 6 |
| 14–26 | planned | — | — | — | |

**결정 로그**

- 2026-09-30: 결과물은 HTML. 규약은 GUIDELINES.md §6(비틀림률은 b′ = −τn, 제2기본형식 계수는 L, M, N). 로드맵은 26장, 절 계획 141개. Phase 0에서 fork 보고를 받아 가이드라인을 수정함(05장 회전지수를 필수로, 18/21장 호환성 명제 분리, 줄기에 03·12·20 추가 등).
- 2026-09-30: `refs/`에 dC(2판, Dover 2016)와 Lee-IRM(2판, 2018)을 확인했다. 둘 다 본문이 완전하다. 쪽 오프셋은 GUIDELINES.md §5.1에 적었다.
  - 원문과 대조해 가이드라인 규약을 확인했다: dC의 b′ = τn, e·f·g, H, Dw/dt, Lee-IRM의 R 부호, 식 (7.4), Rc, S, sec 식 (8.28), 가우스 방정식(정리 8.5), 야코비 방정식 (10.1), Δ = div grad.
  - 새로 반영한 것
    - **Lee-ISM의 라플라시안은 부호가 반대**라는 경고
    - 둘러싼 다양체 표기를 막대에서 물결표로 바꿈(Lee-IRM과 맞추고 ḡ와의 충돌을 없앰)
    - §6.6에 "원문 대조" 열 추가
- 2026-09-30: 사용자가 "계속 진행"을 지시해 웨이브 1(01장 파일럿)을 시작했다.
  - 선택 장 20·26의 포함 여부와 workflow 사용 여부는 아직 답을 받지 않았다. 기본값(Agent 도구로 진행, 선택 장은 나중에 결정)으로 둔다.
- 2026-09-30: 사용자가 01장 품질은 괜찮지만 속도가 느리다며 병렬화를 지시했다("part 1의 4장과 part 0의 나머지도 병렬로").
  - 01(진행 중)과 동시에 02, 03, 04 작성 agent를 시작했다. 동시 작성 agent가 4명으로, 원래 상한인 3명을 넘는다(사용자 지시).
  - 미작성 선행 절의 참조 규칙을 GUIDELINES.md §15.5에 추가했다: 장 index 링크 + todo, 검토에서 확정.
  - 앞으로도 속도를 위해 §4의 웨이브를 앞당길 수 있다. 이때 검토 단계에서 todo 링크 확정을 필수로 한다.
- 2026-09-30: 01장 파일럿 완료. 제안 10개를 처리했다.
  - GUIDELINES.md와 도구에 반영: §2.1(비예 대신 흔한 오해 허용), §2.2(Part 0 분량 완화, 링크된 절은 나누지 않음), §3.2(같은 장 안 예고는 일반 링크), §6.1(A^i_j는 1장부터), §8.1(첨자 읽기), §11.3(규약은 번호 붙은 비고로), §12.3(covector 색 #CC79A7, 패널 2개 이하), §14(연습에서 미리 계산한 기준 예제의 처리), §15.6(검토 중 번호 고정), 부록 E(`tools/shot.mjs`)
  - 매크로 `\im`, `\Span`, `\diag`를 추가했다.
  - check_site에 "링크 없는 번호 참조" 경고를 추가하는 작업은 통합 보조 agent(sonnet)에 맡겼다.
  - **사용자 결정 대기**: ① §7에 극좌표 기저를 기준 예제로 추가할지(규약 변경이라 승인 필요) ② 1.6절(행렬식·방향·외적)이 60분을 넘는데 1.6/1.7로 나눌지. 나누면 02–04장의 링크를 고쳐야 한다.
- 2026-09-30: 03장 완료. §3.5 표(원환면, ℝℙⁿ, 구면 여정의 03장 칸), 매크로 `\Q`, 북극 N 사용 범위(§6.4)를 갱신했다.
  - 03장 용어·기호의 glossary/notation 반영은 02·04장과 묶어 통합 보조 agent로 처리할 예정이다.
  - 03장이 정의하지 않은 개념(Int, 경계, 국소 컴팩트, 파라컴팩트, 완비성, 제1가산, 서로소 합집합 위상)은 뒤 장이 필요하면 스스로 정의한다. 12장(파라컴팩트), 23장(완비성) 작성 agent에게 알린다.
- 2026-09-30: 04장 완료. dgsym에 `signed_curvature`를 추가하고 자체 테스트 2개를 더했다(65 PASS). §3.5 표의 나선·원 칸을 갱신했다.
- 2026-09-30: 02장 완료.
  - 2.1절의 번호가 한 칸 당겨진 일(중복 보조정리 제거)이 있었다. 04장의 02장 링크 171개를 자동 대조해 모두 올바른 항목을 가리킴을 확인했다.
  - §6.3에 결정을 추가했다: 매개화의 미분은 D𝐱(q), 곡면 사이 사상의 미분은 dφ_p. T5 줄기에 02장을 넣었다.
  - 06장 작성을 시작했다(임계 경로).
  - 02–04장 공유 파일 반영은 통합 보조 agent(sonnet)가 맡았다.
  - 사용 한도를 피하려고 동시 opus agent를 5명 이하로 유지한다: 02장 검토와 05장 작성은 진행 중인 검토(01·03·04)가 끝나는 대로 시작한다.
- 2026-09-30: 01장과 03장 검토 완료(reviewed). 02장 검토와 05장 작성을 시작했다.
  - **Part 0 예외 규칙** 추가(GUIDELINES.md §3.2): 02장이 03장의 컴팩트성 사실 3개(정리 3.6.10, 3.6.7, 따름정리 3.6.19)를 "미적분학에서 알려진 사실"로 인용하는 것만 허용한다. 장 순서를 바꾸는 방안은 링크 수백 개를 깨므로 택하지 않았다.
  - 동기 문단에서 뒤 장을 한 문장으로 언급하는 것은 box forward 없이 허용한다(§3.2).
  - style.css: 폭 600px 이하에서 디스플레이 수식을 86%로 줄였다. 1.4절 기준으로 폰 폭에서 넘치는 식이 11개에서 2개가 됐다.
  - 공유 파일
    - index.html: 01·03 reviewed, 02·04 draft
    - notation.html: 매크로 행 문구, 쌍대기저 행의 𝓑*, π_S 추가
    - glossary 용어표: 준리만 계량 추가
    - 03장 개요와 01장 1.6절 사이에 명제 1.6.12 ↔ 정의 3.4.1 연결
- 2026-09-30: 04장 검토 완료. 11장(11.1–11.5)을 앞당겨 시작했다. 06장과 병렬이며, 06장 참조는 §15.5 병렬 규칙을 따른다.
- 2026-09-30: 02장 검토 완료로 **Part 0(01–03) 전체가 reviewed**. Part 0 통합 검토 agent(읽기 전용)를 시작했다.
- 2026-09-30: **Part 0 통합 검토**(읽기 전용) 완료.
  - 총평: "세 장이 한 사람이 쓴 것처럼 읽힌다"(뒤 장의 기준으로 충분). 1.6 분할은 하지 말 것을 권고했다.
  - 문제 27개를 셋으로 나눠 처리했다.
    - **규칙 결정** (GUIDELINES.md에 반영)
      - §3.2: 한 문장짜리 지시문 허용, Part 0 예외에 "받아들이는 사실"(ℝⁿ 완비성 등) 추가
      - §4.2: 다중적분 복습·그린 정리·흐름 정의역을 생략 허용 목록에 추가
      - §6.1: 저차원 예의 (x, y, z) 허용
      - §6.7: 북극 N의 사용 범위, ≅의 여러 뜻
      - §7: 토러스 T²(곱공간)와 원환면(ℝ³ 곡면)의 이름 구분
      - §8.2: 중간값 정리, 계수(rank/coefficient), 원환면/토러스
      - §9 의존 관계 문구, §3.5 Part 0 칸
    - **장 폴더 수정**: Part 0 수정 agent가 맡았다. 한도로 중단된 뒤 10-01에 재개. 2.6의 ∂ 거짓 문장, 폐포 이중 정의, 방향미분 오링크, 줄기 표기, 3.6·3.7 역링크와 새 따름정리 3.7.18, 그린 정리 2.6.13, 용어 통일, 대칭 목록 등을 고친다.
    - **공유 파일**: 통합 보조(sonnet)가 완료했다(notation·glossary·index). 입체사영·토러스·원·나선·구면의 첫 등장 장을 바로잡고, Part 0 표준 대상 표를 두고, 용어 9개를 추가했다.
- 2026-09-30 → 10-01: 사용 한도에 두 번째로 걸렸다. 05·06·11 작성 agent와 Part 0 수정 agent가 중단됐다가 재개됐다. 통합 보조는 한도 전에 끝났다.
- 2026-10-01: 11장 11.1–11.5 완료, 검토를 시작했다.
  - dgsym에 `stereographic(n)`, `rpn_charts(n)`을 추가했다(자체 테스트 80 PASS).
  - §3.5 표의 11장 칸을 갱신했다.
  - 06장 6.5절 상자의 "11장에서 토러스 ≅ 원환면" 문구를 12장으로 고치도록 06장 작성자에게 전달했다.

### 11.6 「Part I과의 관계」 작성 메모 (06·09장 완료 후)

- **06장에서 가져올 것**
  - 정의 6.1.1(조건 (2)가 차트의 위상동형성을 준다), 정리 6.3.1(매개변수 변환이 미분동형 → 호환성의 핵심)
  - 명제 6.1.4, 명제 6.1.8(그래프)
  - 예 6.1.5(여섯 반구 = n = 2의 반구 차트, 예 11.3.4)
  - 예 6.1.7(구면의 기준 매개화. 경도 φ와 겹치므로 차트는 ψ로 부른다, §6.7)
  - 예 6.5.8(원환면 ↔ 예 11.4.12)
- **호환성 명제의 증명 순서**
  1. 부분공간이므로 하우스도르프·제2가산이다(3.5.5, 3.5.10).
  2. (𝐱(U), 𝐱⁻¹)는 차트다.
  3. 좌표변환은 정리 6.3.1로 매끄럽다.
  4. 따라서 매끄러운 아틀라스이고, 구조는 명제 11.2.10으로 유일하다.
  - 이 명제에 `compat` 클래스를 붙인다.
- **11장이 11.6에 맡긴 약속**
  - 11.1의 비교 상자("매개화와 차트는 방향이 반대")
  - 11.3: 예 6.1.5와의 관계, S²가 정칙곡면이라는 사실과 표준 구조의 관계
  - "토러스 ≅ 원환면"은 12장의 미분동형사상으로 넘긴다.
- **09장에서 가져올 것**: 계량만 주어진 곡면 𝕌²(9.6절). 바탕 다양체는 ℝ²의 열린부분다양체(예 11.4.5(a))다. T3 줄기와 연결한다.
- **쓴 뒤 할 일**: 11.5 pager를 11.6으로 바꾸고, 장 dg-status를 draft로 올린다.
- 2026-10-01: 06장 완료, 검토를 시작했다. 07장 작성을 시작했다.
  - 부록 B의 "명제 6.2.3"을 실제 번호인 "명제 6.2.4"로 고쳤다. §3.5 표의 원기둥·원환면 06 칸(𝐱_u × 𝐱_v 방향)을 갱신했다.
  - 01–04장에서 06장 index로만 가던 예고 링크 6곳을 정확한 번호로 바꾸는 일을 Part 0 수정 agent에 추가했다.
  - 06장 공유 파일 요청(용어 11개 + 일반화 4개, 기호 Arg·P_xy·Π·φ̂·𝓗·T_{R,r} 등)은 05·07장과 묶어 통합 보조로 처리할 예정이다.
- 2026-10-01: **Part 0 통합 수정 완료**(목록 20개 + 06장 정확한 링크). 번호 변경은 없다.
  - 새 항목: 명제 3.1.27(폐포 = A ∪ 경계점), 따름정리 3.7.18(연결 열린집합에서 DF = 0이면 상수), 정리 2.6.13(그린 정리, 진술만)
  - §3.2 Part 0 예외에 중간값 정리(3.7.6)를 넣어 네 사실로 늘렸다. §6.7에 ∂의 두 뜻을 등록했다.
  - glossary 요청(선적분, 그린 정리, 조르당 곡선 정리, 내점·내부, 경계점, 코시 수열, 완비성, 최대 노름)은 다음 통합 보조 묶음에 넣는다.
- 2026-10-01: 11장 11.1–11.5 검토 완료.
  - 기호 결정(GUIDELINES.md): 원환면의 점집합은 T_{R,r}, 그래프는 Γ(f), 단위행렬은 I_n, 남극 S는 Part II에서만 쓴다.
  - **후속 정리 작업(대기)**: 03장(원환면 S)과 11장(원환면 𝐱(ℝ²), 11.3의 단위행렬 E)을 새 기호로 맞춘다. 06장(Γ_f)은 검토가 끝난 뒤 Γ(f)로 맞춘다. 05·07장이 끝나면 glossary·notation 통합 보조와 묶어 처리한다.
- 2026-10-01: 05장 완료(검토 시작), 06장 검토 완료.
  - 06장 검토에서 dC 연습문제를 거의 그대로 옮긴 7문항이 발견되어 모두 실질적으로 변형했다. §4.4의 "실질적 변형" 기준을 강화했다. **앞으로 작성 agent 프롬프트에 이 점을 명시한다.**
  - §6.7 φ 규칙을 "한 항목 안에서만 ψ"로 좁혔다. §11.2에 pager 규칙(첫 절 이전 → 자기 장 개요, 마지막 절 다음 → 다음 장 개요)을 추가했다.
- 2026-10-01: 통합 보조(05·06·11 + Part 0 추가분) 완료.
  - glossary에 약 55행을 추가했다. 새 그룹은 "매끄러운 다양체"이고, 꼭짓점의 두 뜻을 구분했다.
  - notation에 25행과 del_j 행을 추가했다. 기호 충돌 목록에 ∂, I_n, Γ(f), 남극 S, φ 규칙을 갱신했다.
  - **후속 기호 정리 완료**: 03·11장의 원환면을 T_{R,r}로, 03·06장의 그래프를 Γ(f)로, 11.3의 단위행렬 E를 I_n으로 바꿨다.
  - 전체 check_all에서 오류는 작성 중인 07·12장 페이지에만 있다(작성 중에 예상되는 상태).
- 2026-10-01: 05장 검토 완료.
  - 10장을 위한 틈(그린 정리의 양의 방향 ⇒ rot = +1)을 발견해 05장 보완 agent가 명제 5.6.12를 추가하고 있다.
  - §12.3에 𝐍/𝐧 동시 표기 색 규칙을 추가했다(𝐍 주홍 실선, 𝐧 주황 점선). §3.5 원 행에 05 칸을 넣었다.
  - 사용 한도에 세 번째로 걸렸다(07·12 작성 중단 → 재개).
- 2026-10-01: 07장 완료, 검토를 시작했다. 08장 작성을 시작했다.
  - 방향의 공식 정의는 "점마다의 방향 + 연속성"(정의 7.5.4, Lee식)이다. det Dh 판정과 𝐍장은 따름정리 7.5.6과 정리 7.5.5로 증명했다.
  - **후속 문구 정리(대기)**: 01장(1.6 예고 "N으로 정의")과 06장(6.3 예고 "det Dh로 정의")의 예고 문구를 7장의 실제 정의에 맞춘다. 07장 공유 파일 요청과 묶어 처리한다.
- 2026-10-01: 12장 완료(12.5 응용 절 신설). 검토를 시작했다.
  - 11.6 메모에 추가할 것
    - 12.1이 "(𝐱(U), 𝐱⁻¹) 전체가 매끄러운 아틀라스 𝒜_S"를 11.6의 결과로 인용한다. 이것을 번호 항목으로 진술한 뒤 12.1과 12장 index의 todo 2곳을 정확한 번호로 바꾼다.
    - S²의 표준 구조(11.3.2) = 𝒜_S의 구조는 11.6에서 증명한다. 예 12.2.8을 인용한다.
  - 표기: "좌표표현"(붙여 씀)으로 통일한다(06장의 "좌표 표현"을 고칠 것). 06장 연습 6.3.7에 "(따름정리 12.5.5에서 사용)" 표시가 필요하다. 둘 다 다음 정리 작업에서 처리한다.

## 8. 작업 중지 (2026-10-01, 사용자 지시)

사용자가 "하던 것까지만 하고 작업 중지, 그리고 push"를 지시했다.
- **새 작업은 시작하지 않는다.** 방금 시작한 13장 작성 agent는 산출물 없이 중지했다.
- **진행 중이던 것은 끝까지 마친다**: 07장 검토, 08장 작성, 05장 보완(명제 5.6.12), 12장 검토.
- 현재 상태를 push하고, 위 네 작업이 끝나면 다시 commit·push한다.
- **다시 시작할 때 할 일** (우선순위 순)
  1. 위 네 작업의 결과를 반영한다: 공유 파일, §3.5, index 상태.
  2. 08장 검토
  3. 13장 작성 (처음부터)
  4. 정리 작업: 05·07·12장 glossary·notation, 01·06장 방향 예고 문구, 06장 "좌표표현"·연습 6.3.7 표시
  5. 이후 웨이브: 09, 14–15, 11.6 …
  6. 사용자에게 받을 답: 극좌표 기준 예제 추가 여부, 선택 장 20·26 포함 여부, 체크포인트 장(01·04·06) 피드백
- 2026-10-01 (중지 중): 07장 검토 완료. 나선면을 H로 부르는 문제(07장; 평균곡률 H와 충돌)를 08장 작성자에게 알렸다. 다시 시작할 때 glossary에 등각나선·양의 매개화·국소 등각사상·방향이 주어진 곡면·비가향을 추가한다.
- 2026-10-01 (중지 중): 12장 검토 완료.
  - §6.7에 "곡면 S와 한 절에 나오면 남극은 −N" 규칙을 추가했다. §3.5 표의 12 칸을 넣었다.
  - 다시 시작할 때 정리 작업에 추가할 것
    - 02장 정리 2.6.9(b)의 "받침"에 `<dfn>`을 붙인다(12.3.2의 data-generalizes 대상).
    - 06장 연습 6.3.7에 "따름정리 12.5.5에서 사용" 표시를 단다.
    - glossary에 12장 용어(국소 유한, 국소 컴팩트, 컴팩트 소진, 파라컴팩트, 소진 함수, 매끄러운 계단 함수, 우리손 보조정리, 확장 보조정리, 좌표표현)를 추가한다.
    - notation에 λ(12장)를 추가한다.
- 2026-10-01 (중지 중): 05장 보완 완료. 명제 5.6.12는 조르당 곡선 정리를 받아들인 사실로 쓴다. 그림 5.6.4(꼭짓점 경우)는 유지한다.
  - 다시 시작할 때 할 것
    - **10장 작성자에게**: "단순 닫힌 곡선(곡선다각형)의 두 방향 가운데 하나는 양의 방향"은 05장에서 증명하지 않았다(5.7도 받아들인 사실로 씀). 10장도 받아들인 사실로 명시해야 한다.
    - 공유 도구 개선: `dgfig.arrow3d`가 선 모양(linestyle)을 받게 한다. 지금은 `ls=`가 라벨에만 전달되는 버그가 있다.
    - notation에 𝐞(φ) = (cos φ, sin φ)(5.6, 국소 기호)를 추가할지 검토한다.
- 2026-10-01 (중지 중): 08장 작성 완료(draft). 진행 중이던 작업이 모두 끝났다. **작업을 중지한다.**
  - 다시 시작할 때 추가로 할 것
    - 08장 검토 (§8의 우선순위 2보다 먼저)
    - 25장 index.html toc-note의 II(X,Y) = (∇̄_X Y)^⊥를 물결표 ∇̃로 고친다(§6.5).
    - 08장 공유 파일 요청
      - glossary: 자기수반, 고윳값/고유벡터, 헤세 행렬, 법단면, 점의 분류 4종, 극소곡면, 뫼니에 정리, 오일러 공식, 로드리게스 판정, 바인가르텐 방정식
      - notation: 08장 국소 기호
      - §3.5 표: 나선·원환면·안장면·원기둥·회전면·그래프의 08 칸
    - 21장 작성자에게: Hess를 정의할 때 08장 8.2의 "헤세 행렬"에 data-generalizes를 붙인다.

## 9. 영어 용어 전환 (2026-10-04, 사용자 지시)

사용자가 "용어는 영어로 다 바꿔 달라"(예: 쌍대공간, 핵, 상)고 지시했다.
- **결정**(AskUserQuestion 답)
  - 범위: 대학 수학 용어 전부. 고등학교 수준의 말은 한국어로 둔다.
  - 처음 정의할 때만 괄호로 한국어를 병기한다.
  - 라벨까지 영어로 쓴다(Definition, Theorem, Proof 등). 장·절 제목도 전부 영어다.
- **정본**: GUIDELINES.md §8(새로 씀), §9 장 제목(영어), tools/terms.tsv(404개), tools/terms_keep.txt
- **진행 순서**
  1. 도구 agent(opus): check_site·build_index 영어 라벨, style.css, fixture, 템플릿·모범 예시, 새 도구 tools/term_guard.py — 진행 중
  2. 장 변환 agent(sonnet), 동시 4명씩
     - 묶음 1: 01, 02, 03, 04
     - 묶음 2: 05, 06, 07, 08
     - 묶음 3: 11, 12, 계획 상태 장 개요(09, 10, 13–26), 공용 페이지(index, notation, glossary 뒤집기, references)
  3. 각 agent는 `python3 tools/term_guard.py <파일>`로 수식·id·링크 대상이 그대로인지 확인한다. 그다음 `check_all --chapter N`을 통과시킨다.
  4. 통합 담당이 전체 check_all을 돌리고 스크린샷을 확인한 뒤 commit한다. push는 사용자 확인 후에 한다(push = 공개 사이트 갱신).
- 2026-10-04: 묶음 1(01–04) 변환 완료, commit dda56e9.
  - 묶음 2(05–08)를 시작했다.
  - 이름 붙은 정리는 고등학교 것이라도 영어로 쓴다(§8.1).
  - **마지막 정리 작업**: 01–04의 남은 한국어 용어 약 45곳을 고친다. 대부분 terms.tsv에 나중에 추가한 말로, 최댓값·최솟값 정리, 상한·하한, 상미분방정식, 8자 곡선, 기준 예제, 평행이동(→ translation) 등이다. 장 사이에 서로 다른 절 제목·용어를 맞춘다. 02장 thm-name의 수식 중복 등 어색한 곳을 다듬는다.
- 2026-10-04: **영어 용어 전환 완료.**
  - 대상: 01–08, 11, 12장(절 전부), 계획 상태 장 개요 16개, 공용 페이지(Contents, Notation and Conventions, Glossary를 English → 한국어로 뒤집고 560행, References, Index)
  - 검사
    - term_guard: 86개 파일에서 0 errors. 남은 경고 8개는 모두 근거가 있다(의도한 한국어 병기, 적분 치환의 '치환', '이상한' 오탐).
    - 전체 check_all OK(그림·verify 157개 PASS, 수식 34,318개 오류 0)
    - pager·breadcrumb 제목은 대상 페이지 h1과 모두 일치한다.
  - 링크 괄호 속 항목 이름은 짧은 형태가 많아 대상 thm-name과 135곳에서 다르다. 원래 한국어판도 짧은 이름을 썼으므로 맞추지 않았다.
  - 다음 세션: §8의 "다시 시작할 때 할 일"(08장 검토, 13장 작성 …)로 돌아간다. 새로 쓰는 장은 처음부터 GUIDELINES §8의 영어 용어 규칙을 따른다.


## 10. Chapter 0: A Guided Tour (2026-10-06, 사용자 지시)

사용자는 컴퓨터비전 전공 CS 학생이다. 깊은 장이 직관을 얻기에 너무 어려워졌다고 했다. 깊은 장은 그대로 두고, Preliminaries 앞에 전체 개념을 가볍게 훑는 입구를 만들기로 했다.
- **결정**
  - 이름은 「A Guided Tour of Differential Geometry」, 폴더는 `chapters/00-tour/`, 장 번호는 0이다.
  - 규칙은 GUIDELINES §2.3이다. 독자는 CV·그래픽스 CS 학부생이다.
    - 증명은 쓰지 않는다. dfn 대신 strong으로 표시한다.
    - 절마다 In computer vision 상자(`aside.box.application`)와 Go deeper 상자(`aside.box.deeper`)를 둔다.
    - Exercises 제목은 "Quick check"다.
  - 절은 8개다(0.1–0.8). 절 계획은 `chapters/00-tour/index.html`에 있다.
- **기반 작업**(완료)
  - GUIDELINES §2.3, §9 표, §11.3 상자 종류
  - style.css에 application·deeper 상자 추가
  - 루트 index.html에 Tour 표와 "어디서 시작할까" 상자 추가
  - 01 index의 pager prev
  - references.html `#sec-refs-applications`: CV·그래픽스 문헌 23개
  - term_guard가 `</strong>(한국어)`도 첫 병기로 본다
- **진행**
  - 작성 agent 2명(opus)을 동시에 띄웠다. A는 0.1–0.5, B는 0.6–0.8을 쓴다.
  - 서로의 파일이 아직 없으면 `index.html#toc-0-M`으로 임시 링크하고 보고한다.
  - 2026-10-06: 작성 완료(A·B 모두 check_all --chapter 0 OK).
    - 통합: \(\mathrm{SO}(3)\)로 통일, references에 Harris88 추가, terms.tsv에 38개 추가, glossary `#sec-glossary-tour` 신설, toc-note 수정(manifold learning은 0.8로 옮김)
    - 검토 agent 2명(opus)을 동시에 띄웠다. R1은 0.1–0.4, R2는 0.5–0.8을 맡는다. 둘 다 장 전체를 읽고 일관성을 본다.
  - 사용자 지시(2026-10-06): 끝나면 알아서 push한다.
  - 2026-10-06: **검토 완료, Chapter 0 전체 reviewed.**
    - R1(0.1–0.4)이 고친 것
      - 수학 오류 1건: saddle surface의 H = 0인 점은 원점만이 아니라 두 직선 x = ±y다.
      - seam의 이유, conformal·넓이 보존 문장, LSCM의 null space 설명을 바로잡았다.
      - "정점"을 vertex로 통일했다.
      - 첫 병기가 빠진 용어 13개를 보충했다.
      - 그림 2개의 시점을 바꿨다.
    - R2(0.5–0.8)가 고친 것
      - gimbal lock의 β = −π/2 경우
      - "SO(3)를 덮는 chart 하나는 없다"의 조건(정의역이 ℝ³의 open set)
      - heat method의 과장된 진술
      - curvature operator와 curvature tensor의 구별(§6.5)
      - "카메라 방향 = SO(3)"를 orientation으로 고쳤다
      - Where to Go Next의 선수 장 누락
      - 기호 충돌(θ, R)
      - 첫 병기가 빠진 용어 약 35개를 보충했다
    - 분량: 0.6이 약 25분으로 가장 길다. 줄이려면 구조를 바꿔야 한다(Vector Fields and Flows를 한 문단으로). 지금은 그대로 둔다.
    - 통합 담당이 반영한 것
      - references: Harris88, Mokhtarian92
      - terms.tsv·glossary: 곡률 연산자, 심사도법, 아이코날 방정식, 다양체 가설, retraction, 호의 길이 매개화, 프레네-세레 공식, mesh의 vertex
      - 상태: 00 index의 toc와 dg-status, 루트 Tour 표를 Reviewed로 바꿨다.
    - 전체 check_all OK
    - 남은 것
      - verify 3곳이 본문에서 지운 주장을 아직 검사한다(Varadhan, DLT 수렴, Figure 0.6.2의 σ 값). 해는 없다.
      - θ_j(angle defect)와 θ₀(colatitude)가 가까이 쓰인다. 사소하다.
  - 다음: §8의 "다시 시작할 때 할 일"(08장 검토, 13장 작성 …)로 돌아간다. 깊은 장을 쓸 때 해당 Tour 절을 링크할 수 있다(§2.3).

## 11. Part IV: 3D Vision Through Differential Geometry (2026-10-07, 사용자 지시)

사용자는 3D vision 전문가(NeRF, 3DGS, SfM. 지금 관심은 3DGS)다. 3D vision에 쓰이는 기하를 differential geometry로 해석하는 내용을 요청했다. 목차를 먼저 승인받았다.
- **결정**
  - 31장(Deformation and Dynamic Scenes)은 뺀다.
  - Part IV는 27–30장, 18절이다. 목차와 읽는 순서는 Tour 다음, Part 0 앞이다.
  - 29장(3DGS)부터 쓴다.
  - 그림 설명을 많고 자세하게 쓴다. 사용자: "예시 그림이 이해되지 않는 경우들이 있었어."
- **기반 작업**(완료)
  - GUIDELINES: §2.4(Part IV 원칙), §9 표(27–30), §11.3 `insight` 상자, §12.6 그림 설명 규칙(캡션 네 부분, 본문에서 그림 따라 읽기, 그림 안의 단서)
  - style.css에 `aside.box.insight`(What geometry explains) 추가
  - references.html `#sec-refs-3d-vision`: 문헌 24개
  - 장 계획 4개(27–30)
  - 루트 index에 Part IV 표 추가
  - pager: 00 → 27 → … → 30 → 01
  - Tour 0.8의 Where to Go Next에 Part IV 경로 추가
- **진행**
  - 29장 작성 agent 1명(opus)을 띄웠다. 29.1–29.6을 차례로 쓴다.
  - 다음 할 일
    1. 보고를 받는다.
    2. 검토 agent를 띄운다. 그림 설명 규칙을 특히 본다.
    3. 통합하고 commit한다.
    4. push한다. 사용자 지시(2026-10-07): "29장까지 작성하고, 검토가 끝나면 알아서 commit하고 push하고 나한테 알려줘"
    5. 이어서 27, 28, 30장을 쓴다(27·28은 서로 독립).
  - 제안할 것: Tour 그림의 캡션을 §12.6 규칙으로 감사하는 일. 사용자는 2026-10-07에 "일단 29장까지"라고 했으므로 보류한다.
  - 2026-10-07: **29장 작성·검토 완료(reviewed).**
    - 6절, 그림 23개(interactive 3개), verify 6개(약 500개 검사)
    - 검토자 R1(29.1–29.3), R2(29.4–29.6)가 공식 코드와 대조했다: gaussian-splatting 54c035f, diff-gaussian-rasterization main과 9c5c202, simple-knn, mip-splatting.
    - 고친 사실
      - clone은 같은 위치에 복사한다(논문의 설명과 다르다).
      - densification의 NDC norm은 해상도에 거의 무관하다. 처음 원고의 해상도 주장은 틀렸다.
      - 초기 scale은 3-NN 거리의 RMS다.
      - `--antialiasing` 옵션, Mip-Splatting filter 값
    - 고친 수학
      - EWA 오차 계수는 c₄ = 0.839다(처음 0.918). 오차 표를 quadrature로 다시 계산했다.
      - 2D 예의 rank 주석
      - affine-invariant curvature의 정규화
      - BW 보간 진술에 commute 조건을 붙였다.
    - 통합
      - references에 Ye24(AbsGS) 추가
      - terms.tsv 3개
      - glossary `#sec-glossary-3d-vision`
      - notation `#sec-3d-vision`과 Symbol Conflicts(W, J, ℓ)
    - 남은 것
      - 27.1, 27.4, 28.1, 28.2, 30.3으로 가는 링크는 장 index의 toc 앵커를 가리킨다. 그 절을 쓸 때 실제 파일로 바꾼다.
      - 27장의 투영 기호를 29.3처럼 φ로 맞춘다(27 index는 π).
      - ω_k(아래첨자, 29장)와 Tour의 ω^k가 다르다.
      - 400px에서 그림 글자가 작다. 사이트 전체에 click-to-enlarge CSS를 검토한다.
  - 2026-10-07: 사용자 지시 "30장부터 써줘".
    - 30장 작성 agent(opus)를 띄웠다. 30.1–30.4를 차례로 쓴다.
    - references에 8개를 추가했다: Li23, Barron22, Goldman05, Osher88, Crandall83, Curless96, Newcombe11, Kazhdan06.
    - 30 index toc-note의 mean curvature 부호를 고쳤다. \(\operatorname{div}\mathbf N = -2H\)(규약 \(W_p = -dN_p\)).
    - 구현 사실은 작성 단계에서부터 공식 코드와 대조하라고 지시했다.
  - 2026-10-07: 사용자 "ㅇㅇ"(30장 검토 후 push 허락). 이어서 mesh 장을 승인했다("31장도 작성해줘").
    - Chapter 31 「Meshes: Discrete Differential Geometry」 6절(31.6 parametrization/texture 포함)
    - 폴더 `31-meshes`. Part IV의 마지막 장이다. 순서는 30 → 31 → 01이다.
    - 앞서 뺀 dynamic 장의 번호를 이 장에 다시 썼다.
    - 반영: GUIDELINES §2.4·§9, 루트 index, Tour 0.8과 00 index, 01 index의 pager prev, references에 14개 추가
    - **통합 때 할 일**: 30 index의 pager next를 31로 바꾼다. 지금은 30 작성자가 그 파일을 고치는 중이라 미뤘다.
    - 31장 작성 agent(opus)를 30장과 동시에 띄웠다. 두 장은 서로 독립이다.
    - push는 30장과 함께 검토 후에 한다.
  - 2026-10-08: **30장 작성·검토 완료(reviewed).** 사용자: "검토 끝나면 알아서 commit push 해"
    - 4절, 그림 17개(interactive 3개), verify 4개
    - 검토 R1(30.1–30.2)이 고친 것
      - eikonal loss에 "topology가 정하는 하한"이 있다는 오류(infimum은 0)
      - curvature loss에서 minimal surface의 비용이 0이라는 오류(offset sample에서는 \(H_t \ne 0\))
      - 안쪽에서 \(\grad f\)의 방향
      - VolSDF의 eikonal sample 영역(구가 아니라 cube)
    - 검토 R2(30.3–30.4)가 고친 것
      - grazing angle에서의 FD normal 오차를 2D에서 3D 분석으로 바로잡았다. depth를 픽셀 중심에서 어긋나게 읽으면 grazing에서 오차가 커진다.
      - SuGaR loss를 실제 density로 쓴 \(f\)로 고쳤다.
      - smoothness prior 실험의 λ 단위를 맞췄다.
      - 2DGS의 normal loss 구현은 \(1 - A\langle\cdot\rangle\)이다.
    - 통합
      - 29장의 30.3 링크를 실제 파일로 바꿨다.
      - terms.tsv 4개, glossary 3개, notation(𝖪, \(d\), \(\rho\), \(K\)/𝖪 충돌)
      - 부록 E에 Chrome 스크린샷이 잘리는 문제를 적었다.
    - 31장은 작성 중이다. 공개 사이트에 깨진 링크가 없도록 31 index는 planned 판으로만 commit했다(작업 트리는 그대로 둔다).
  - 2026-10-08: **31장 작성·검토 완료(reviewed).**
    - 6절, 그림 26개(interactive 9개), verify 6개, 공용 helper `chapters/31-meshes/lib/dgmesh.py`(2026-10-18 `tools/dgmesh.py`로 옮김)
    - 검토 R1(31.1–31.3)이 고친 것
      - Hildebrandt06 예의 수치(consistent mass)
      - Figure 31.3.3의 원인 설명(둔각삼각형의 mixed area)
      - uniform loss 결과 서술
      - \(\iint H\)의 상수 \(-\tfrac12\theta_e\ell_e\)
      - SVG gouraud shading이 만든 가짜 색
    - 검토 R2(31.4–31.6)가 고친 것
      - Prop 31.5.4(평균이 face 평면 밖이면 두 binding이 다름)
      - Prop 31.6.4의 가정(국소 one-to-one)
      - similarity binding이 "가장 가까운 conformal"이라는 오류
      - 직사각형 eigenvalue 중복
      - PyTorch3D의 `dists`는 거리의 제곱이다.
      - LSCM flip 반례를 독립 구현으로 재확인했다(극단적 mesh에서만 생기며 드물다는 것을 함께 밝힘).
    - 통합
      - 용어 10개, references 6개
      - notation(mesh 기호, \(L\)·\(M\)·\(\chi\) 충돌)
      - glossary 9개
      - check_all에 BLAS 쓰레드 제한 추가. v-29-3이 가끔 segfault를 냈다.
    - 남은 것
      - dgmesh.py를 tools/로 옮길지 검토한다.
      - plotly Mesh3d의 facenormalsepsilon 문제가 다른 장 interactive 그림에도 있는지 확인한다.
      - 27·28장은 아직 planned다. 29–31장의 27·28 링크는 toc 앵커를 가리킨다.
  - 2026-10-08: 사용자 "계속 진행해줘". 27장과 28장 작성 agent(opus)를 동시에 띄웠다. 두 장은 서로 독립이다.
    - references에 21개를 추가했다(Ma04, Nister04, Hartley97, Kannala06, Plücker, Lie group spline, BARF 등).
    - 27 index의 투영 기호를 π에서 φ로 바꿨다(29.3과 맞춤).
    - 다음: 장마다 검토 agent 2명 → 통합(29–31장의 27·28 앵커 링크를 실제 파일로) → commit, push. 사용자가 검토 후 자동 commit·push를 계속 허락하고 있다.
  - 2026-10-08: **27장·28장 작성·검토 완료(reviewed). Part IV(27–31장) 전체가 reviewed다.**
    - 27장: 4절, 그림 18개, verify 4개
      - R1(27.1–27.2)이 고친 것
        - Figure 27.2.1의 거울상 오류(vanishing point 좌우)
        - Prop 27.2.1의 전역 논증
        - \(\Sigma\)가 dual conic의 블록이라는 서술을 \(\mu=0\)으로 한정
        - NDC 평행성
        - 장 내부 기호 충돌 다수
      - R2(27.3–27.4)가 고친 것
        - Tron17의 quotient는 \(\mathcal E/\{\pm1\}\)이다.
        - (σ,σ,0) 유도의 일반화
        - Pless03 부호 설명
        - light field가 4D인 조건(장면을 떠난 뒤의 구간)
        - Ma04 용어("essential space")
        - COLMAP 초기화 기준(median angle, forward motion)
    - 28장: 4절, 그림 15개, verify 4개, lib/dglie.py·dgba.py
      - R1(28.1–28.2)이 고친 것
        - Exercise 28.2.1 풀이
        - Ceres 축의 frame과 공분산 4배
        - log 문턱값 산술
        - retraction의 회전각 닫힌 꼴
        - COLMAP `CamFromWorld`
        - nerfstudio world 축 변경
        - Strasdat 벡터 순서
      - R2(28.3–28.4)가 고친 것
        - B-spline order k는 \(C^{k-2}\)다(Sommer20의 \(C^{k-1}\)은 오류).
        - LM step이 orbit에 수직인 것은 D = I일 때뿐이다(Ceres는 diag(JᵀJ)).
        - BARF Jacobian의 점
        - bi-invariant pseudo-metric 언급
        - COLMAP gauge 적용 범위(global BA)
    - 통합
      - 29–31장의 27·28 앵커 링크 23개를 실제 절로 바꿨다.
      - references 28개를 추가했다.
      - terms.tsv에 13개를 추가했다.
      - notation: 27·28장 행과 T/t, J, W/s 충돌
      - glossary: 11개를 추가했다.
    - 남은 것
      - 400px에서 2–3패널 그림 글자가 작다(검토자 여럿이 지적). 사이트 전체에 확대 CSS를 검토한다.
      - lib/*.py를 tools/로 옮길지 검토한다. (2026-10-18: dgmesh는 tools/로 옮겼다. 28장 lib/는 그대로다)

## 12. 애니메이션: manim (2026-10-11, 사용자 지시)

- 사용자 요청: "Chapter 0과 Chapter 31에 대해서 manim으로 잘 표현할 수 있는 애니메이션이나 중요한 부분, 직관을 잘 표현하는 부분 전부 애니메이션 만들어줘. commit push해서 공개 사이트에 올라가도록. 애니메이션 규칙도 작성하고." 이어서 "애니메이션 작업이 끝나면 알아서 commit push 해줘"라고 했다.
- **기반**(완료)
  - `tools/setup_manim.sh`: `.manim/`(git 제외)에 manim 0.19.1을 설치한다.
    - 우회한 것: python3 -S, pyzed 권한 문제, manimpango·srt 직접 빌드, 시스템 pycairo
    - 실행은 `python3 -S`로 한다. `-S`가 없으면 scipy가 /usr/local의 cython.py에서 PermissionError를 낸다.
  - `tools/dganim.py`: 흰 배경, dgfig 색, `label`, `mtex`(matplotlib mathtext, 크기 보정), `mesh_group`(Lambert 음영), `main`(faststart mp4, poster, 4 MB 상한)
  - `tools/render_anims.sh`: taskset -c 0-7, `--check`
  - check_site: `figure.dg-figure.animation`, id `anim-N-M-K`, "Animation N.M.K."(그림과 따로 센다), video poster·controls·aria-label·source, .py 존재 확인
  - style.css: video 스타일
  - GUIDELINES §12.8(애니메이션 규칙), §10, §11.5, 부록 E
- **진행**: 작성 agent 4명(opus)을 동시에 띄웠다. A는 Tour 0.1–0.4, B는 Tour 0.5–0.8, C는 31.1–31.3, D는 31.4–31.6을 맡는다.
  - B 완료: 21개(0.5.1–0.5.5, 0.6.1–0.6.6, 0.7.1–0.7.4, 0.8.1–0.8.6). 검토 agent 2명(0.5–0.6, 0.7–0.8)을 띄웠다.
  - B의 요청 처리
    - dganim.mtex: `$…$` 밖의 공백 글리프(d 없는 `<path>`) 때문에 svgelements가 TypeError를 냈다. `_fix_empty_paths`로 고쳤다.
    - docstring: 바뀌는 숫자는 label, Unicode minus, Text의 `_`는 첨자가 아님을 적었다.
    - cross-section은 terms.tsv에 넣지 않았다. 수학 용어가 아닌 일반 낱말이다.
  - C 완료: 12개(31.1.1–31.1.4, 31.2.1–31.2.4, 31.3.1–31.3.4). 검토 agent 2명(31.1–31.2, 31.3)을 띄웠다.
  - C의 요청 처리
    - mesh_group: face마다 shade_in_3d를 켜서 Scene3D가 깊이 정렬한다. `_Camera3D`가 이 face에 음영을 다시 넣지 않는다. torus로 시험했다.
    - dganim docstring: frozen-frame updater
    - §12.8: 가장자리 여백, 멈춘 상태의 control bar(아래 약 12%), face 중심 정렬의 한계, 용량
  - D 완료: 15개(31.4.1–31.4.5, 31.5.1–31.5.6, 31.6.1–31.6.4). 검토 agent 2명(31.4, 31.5–31.6)을 띄웠다. mtex docstring에 `\lt`/`\gt` 없음, label 폭 어림을 적었다.
  - 검토 0.5–0.6 완료: 7개를 고쳤다. 수치 오류는 없었고 진술 3건을 고쳤다(0.6.5 건너뛴 뒤의 회전각 2π − |ω|, 0.5.5 땅콩 모양, 0.5.1 Mercator 자르는 위도).
    - 이어서 poster 아래 띠(90 px)와 가장자리 여백을 점검했다. 0.5.1, 0.5.4, 0.6.1, 0.6.3, 0.6.5의 숫자 줄을 옮겨 다시 렌더했다. 0.5–0.6 검토 완료.
  - 사용량 한도로 멈춤(2026-10-11). 이후 7명 모두 SendMessage로 재개했다.
  - 검토 0.7–0.8 완료: 수학 오류 없음. 9개를 고치고 10개 모두 다시 렌더했다(0.7.4 본문의 γ₂ 누적값 +0.28 → −0.28 → 0, 0.8.1 "straight" → "horizontal", 0.7.3 edge당 1.5초 등). poster 띠는 모두 비었다.
  - 검토 31.1–31.2 완료: 수학 오류 없음. 7개를 고치고 다시 렌더했다. 가장 큰 결함은 31.1.3 tube가 U자 구간에서 접혀 보이는 면의 약 25%가 안쪽이던 것이다. rotation-minimizing frame으로 다시 만들고 바깥 normal assert를 넣었다. 나머지는 poster 띠, 31.2.4 "A(ε) = …", Fig 31.1.4 alt 색.
  - 검토 31.4 완료: 5개 모두 고쳐 다시 렌더했다. 수치 오류 1건: HKS 교차 시각 0.58 → 0.60(격자점 대신 보간, 본문 1곳·캡션 2곳). 나머지는 poster 띠, 겹침, 캡션 시간 구간, |H − 1| 범위 문구.
  - A 완료(재개 후): 14개(0.1.1–0.1.2, 0.2.1–0.2.4, 0.3.1–0.3.4, 0.4.1–0.4.4, 0.4는 문서 순서로 다시 매김). 검토 agent 2명(0.1–0.2, 0.3–0.4)을 띄웠다.
    - A의 요청: terms.tsv와 glossary에 "평행원 (회전면의) → parallel (of a surface of revolution)"을 넣었다. §12.8에 좁은 화면에서는 control bar 음영이 약 20%라는 점과 "되도록 아래 15% 위"를 적었다.
  - 검토 31.5–31.6 완료: 수치 오류 없음. 10개 모두 고쳐 다시 렌더했다.
    - 31.6.1: SVD 부호 때문에 W와 U가 거의 반 바퀴씩 돌던 것을 회전각이 작은 (−U, −W)로 바꿨다.
    - 31.5.6: √2는 normal 방향으로 밀 때의 값이라고 밝혔다.
    - 31.5.2: "충분조건일 뿐"
    - 31.6.4: 시점을 바꿔 seam이 보이게 했고, 1 chart는 cap과 "거의 같다"로 고쳤다.
    - 그 밖에 poster 띠
  - 검토 0.3–0.4 완료: 수치 오류 없음. 5개를 다시 렌더했다.
    - 0.4.2: torus가 안팎이 뒤집혀 그려졌다(cull이 x_u × x_v 쪽을 앞면으로 본다). 격자 방향을 뒤집고, 바깥 normal assert를 넣었다.
    - 0.4.3: 위아래 축척 차이를 점선으로 보였다.
    - 0.3.4: "ellipse가 납작해지며"가 틀렸다(비는 0.008 → 0.016).
    - 그 밖에 0.3.2 경로, poster 띠
    - GUIDELINES §12.8 예시의 파일 이름을 anim-0-4-3-gauss-map으로 고쳤다.
  - 검토 31.3 완료: 수학 오류 없음. 4개 모두 아래 띠의 글자를 올려 다시 렌더했다. 캡션 3곳을 고쳤다(31.3.4의 Laplacian은 거칠기 0.62를 남기며 줄어든다; d/dt Σ|x−c|² = −2k𝓛의 근거). 그때 아직 돌고 있던 것: A(Tour 0.1–0.4 작성, 0-4 페이지의 anim 번호를 바꾸는 중이라 check_all --chapter 0이 FAIL), 검토 0.7–0.8, 31.1–31.2, 31.3, 31.4, 31.5–31.6
  - 검토 0.1–0.2 완료: 수학 오류 없음. 6개 모두 다시 렌더했다(poster 띠, 겹침, 0.2.3 γ(u) → γ(t), 0.1.1 "정확히 같아진다" → "가까워진다").
- **통합**(2026-10-11)
  - mtime이 낡은 3개(0.2.3, 0.3.1, 0.4.2)를 다시 렌더했다.
  - `_helpers_a.py` docstring에 cull 방향 경고를 넣었다. §12.8의 마지막 상태 유지를 1–3초로 고쳤다.
  - `render_anims --check` 62/62, `check_all` OK, test_checkers 통과
- **결과**: 애니메이션 62개, 약 81 MB
  - Tour 35개: 0.1 2, 0.2 4, 0.3 4, 0.4 4, 0.5 5, 0.6 6, 0.7 4, 0.8 6
  - Chapter 31 27개: 31.1 4, 31.2 4, 31.3 4, 31.4 5, 31.5 6, 31.6 4
  - 모두 작성, 독립 검토, 통합을 거쳐 commit하고 push했다.
- **남은 것(선택)**
  - `_helpers_a.Painter`, `_helpers_c/d.MeshView`, `_helpers_b.View`를 tools/dganim으로 옮긴다.
  - Sun09 원문에서 HKS의 a₁ 서술(“scalar curvature … Gaussian curvature”)을 확인한다(31.4 본문, 이번 작업 이전부터 있던 문장).

## 13. Track 2: Discrete Differential Geometry (CMU 15-458/858) + 세 track 재구성 (2026-10-18, 사용자 지시)

- 사용자 요청: "discrete differential geometry만 모아보려고 해. cmu의 강의 기준으로 할거야(재생목록 PL9_jI1bdZmz0hIrNCMQW1YmZysAiIYSSS, 교재 Crane DDG notes). 하면서 전체 교재를 다시 정리해보자. 심화 버전(part 123), 기초와 쉬운 버전(tour, part 0·4). part 순서도 다시 정해보자. 이 강의는 독립적인 섹션으로 두자."
- 이어서: "5시간 한도에 도달해도 초기화되면 알아서 계속 진행하고, 작업이 끝나면 알아서 commit push 해줘."
  - 그래서 cron(매시 17분, 세션 안에서만 유지, 7일 뒤 만료)이 자동 재개를 점검한다.
  - push는 검토와 check_all을 통과한 단위마다 origin/main으로 한다(push = 공개).
- **결정** (AskUserQuestion, 모두 추천안)
  1. 순서는 Track 1 기초·입문(Tour → Part IV → Part 0) → Track 2 DDG(32–45) → Track 3 심화(Part I–III)다.
  2. 장 번호는 그대로 둔다. URL, 항목 번호, 애니메이션 id가 바뀌지 않는다.
  3. 31장은 Part IV에 남기고 DDG와 서로 링크한다.
  4. DDG 목차는 14장 73절로 승인되었다. 재생목록에 없는 43 Conformal(L19–20)과 45 Vector Fields(노트 8장)도 넣는다.
- **작업 공간**
  - worktree `.claude/worktrees/ddg-cmu`, branch `worktree-ddg-cmu`. 원래 세션(dg-study-main)이 main checkout을 쓰고 있어서 분리했다.
  - `refs`와 `.manim`은 main checkout으로의 symlink이고, `.git/info/exclude`에 등록해 두었다.
- **자료**
  - `refs/ddg-crane/crane-ddg-notes.pdf`: 2025-01-29판. PDF 쪽 = 인쇄 쪽 + 1
  - `refs/ddg-crane/slides/`: 강의 슬라이드와 과제
  - 강의와 영상, 슬라이드의 대응표는 references.html `#sec-refs-ddg`에 있다.
- **기반 작업**
  - 완료(통합 담당)
    - GUIDELINES: §1 세 track, §2.4 31장 처리, §2.5(Track 2 원칙: 출처, 엄밀성, dfn 규칙, 절 구성, 기호, 기준 예제), §5 표(Cr, CMU-DDG), §5.1 Track 2 자료, §9(Track 열, 32–45 행, DDG 상세표, 의존 관계), §10 도구 목록, §11.3 상자 `discrete`·`code`
    - style.css: `aside.box.discrete`(olive), `aside.box.code`(slate, pre)
    - references.html `#sec-refs-ddg`: CMU-DDG, 강의 행 `CMU-DDG-L1` … `CMU-DDG-L21-22`, `CMU-DDG-VVF`, `CMU-DDG-HM1/HM2`, 논문 16개
    - 루트 index.html: About의 세 track, "어디서 시작할까"의 세 경로, Table of Contents를 `sec-track-1/2/3`로 묶음(Part는 h4), Track 2 표(32–45, Planned), Dependencies
    - pager: 03 index와 3.7의 next → 32, 04 index의 prev → 45
  - tools agent(진행 중)
    - `chapters/31-meshes/lib/dgmesh.py` → `tools/dgmesh.py`
    - 새 `tools/dgdec.py`: DEC, 단체복합체, halfedge, discrete curve, 기하량의 미분. self-test를 check_all에 등록
  - planning agent(진행 중): 장 개요 14개(`planned`). toc-note가 작성자에게 주는 계약이다.
- **웨이브 계획**
  - 원칙
    - 선행 장이 **draft**가 되면 다음 장을 시작한다. 이전 원칙(§3의 4번)은 reviewed 뒤에 시작했다. 검토자가 번호를 바꾸지 않으므로(§15.6) 이렇게 풀었다.
    - 아직 쓰이지 않은 선행 장은 §15.5대로 장 개요를 링크하고 todo를 단다. 검토 단계에서 확정한다.
  - 두 갈래로 나눈다.
    - A갈래(forms/DEC): 34 → 35 → 36 → 42 → 43
    - B갈래(curves/surfaces/curvature): 32 → 37 → 38 → 39 → 40 → 41
    - 그 뒤 44(← 39, 42) → 45(← 36, 42, 44)
  - 웨이브
    - W1: 32, 33, 34
    - W2: 35, 37
    - W3: 36, 38
    - W4: 39, 42
    - W5: 40, 43
    - W6: 41, 44
    - W7: 45
  - 장 하나가 draft가 되면 검토 agent를 띄운다. 6절인 장은 2명이 절을 나누고, 4–5절인 장은 1명이다.
  - 동시에 도는 agent는 작성과 검토를 합쳐 5명 이하로 둔다(이 컴퓨터의 부하).
  - 통합: 웨이브마다 공유 파일(terms.tsv, glossary `#sec-glossary-ddg`, notation `#sec-ddg`, references)을 반영한다. 그다음 check_all을 돌리고 commit한다. 검토가 끝난 단위는 push한다.
  - 끝난 뒤
    - Track 2 통합 검토(§15.6의 3단계)
    - 31장·Tour에서 DDG로 가는 링크를 단다.
    - 애니메이션 단계(§12.8): 사용자가 Tour·31장에서 원했던 수준으로 한다.
    - 마지막에 push한다.

### 13.1 DDG 장 작성 agent 프롬프트 (템플릿)

```
너는 DG-study의 Track 2(Discrete Differential Geometry) {NN}장 「{제목}」 작성 담당이다.
작업 공간: /home/kbh2/DG-study/.claude/worktrees/ddg-cmu (git worktree). 여기서만 명령을 실행한다. commit하지 않는다.
1. CLAUDE.md와 GUIDELINES.md 전체를 읽는다. 특히 §2.5(Track 2 원칙)를 지킨다.
2. 다음을 읽는다.
   - chapters/{폴더}/index.html: 절 계획과 toc-note. toc-note가 계약이다.
   - 이어받는 장들의 index와 이미 쓰인 절
   - 31장과 Tour의 해당 절. 겹치는 서술은 링크로 비교하고 복사하지 않는다.
   - tools/tests/fixtures/site/, notation.html, glossary.html, concepts.html
   - tools/dgmesh.py와 tools/dgdec.py(공용 구현, 부호의 정본)
3. 출처: refs/ddg-crane/crane-ddg-notes.pdf(PDF 쪽 = 인쇄 + 1)와 refs/ddg-crane/slides/의 해당 강의.
   - 구성만 참고한다. 문장, 그림, 연습문제를 옮기지 않는다(§2.5 저작권).
   - 영상은 볼 수 없으므로 영상 내용을 지어내지 않는다.
4. 절을 순서대로 쓴다. 절마다 figures(§12, 그림 설명 규칙 §12.6 필수)와 verify(§13)를 만들고 실행한다.
   - Implementation 상자의 코드와 coding 연습의 풀이는 verify에서 그대로 실행한다.
   - 3D mesh에는 interactive 그림을 권장한다. 애니메이션은 이번 단계에서 만들지 않는다.
5. 장 index.html을 고친다. 쓴 절을 링크로 바꾸고 draft로 올린다. Builds on과 Leads to를 실제 내용에 맞게 고친다.
6. bash tools/check_all.sh --chapter {NN}을 통과시킨다. node tools/shot.mjs로 절마다 900px과 400px에서 렌더링을 확인한다.
7. 담당 장 폴더 밖의 파일은 수정하지 않는다. 공유 파일에 반영할 것은 보고한다.
   - 공용 helper가 더 필요하면 장 안의 lib/에 두고 보고한다(통합 담당이 tools/로 옮긴다).
8. 최종 보고(§15.4): 만든 파일, 새 용어(terms.tsv 형식), 새 기호, 새 참고문헌, [확인 필요], 규약 결정, 다음 장이 알아야 할 것
```

### 13.2 DDG 장 검토 agent 프롬프트 (템플릿)

```
너는 DG-study Track 2 {NN}장 「{제목}」({절 범위})의 검토 담당이다. 작성자와 독립적으로 판단한다.
작업 공간: /home/kbh2/DG-study/.claude/worktrees/ddg-cmu. 여기서만 명령을 실행한다. commit하지 않는다.
1. GUIDELINES.md 전체를 읽는다. 특히 §2.5, §15.6, §16, §12.6.
2. 장 전체를 처음부터 끝까지 읽는다(맡은 절은 자세히). 순서는 수학 오류 → 규약(§6, §2.5 기호와 부호, dfn 규칙) → 연결 → 흐름 → 표현이다.
3. 검산
   - verify를 다시 실행한다.
   - 절마다 독립 검산을 1개 이상 한다. scratch는 /home/kbh/.claude/jobs/6f56e074/tmp/에 둔다.
   - Implementation 상자의 코드를 직접 돌려 본다.
   - Crane 노트·슬라이드와 대조해 내용을 확인하되, 문장을 옮겨 오지 않았는지도 확인한다.
4. 그림은 PNG로 렌더링해 눈으로 확인하고, 캡션의 네 부분(§12.6)을 확인한다. 400px 폭도 본다.
5. 고칠 수 있는 것은 직접 고친다. 판단이 필요한 것은 <!-- REVIEW: … -->로 남긴다. 번호는 바꾸지 않는다.
6. 선행 장이 쓰였으면 todo 링크를 확정한다. check_all --chapter {NN}을 통과시키고, 문제없는 절을 reviewed로 올린다.
7. 최종 보고: 문제와 수정(심각도 순), 남은 REVIEW, 공유 파일 요청
```

### 13.3 진행 기록
- 2026-10-18: 결정과 기반 작업(위). tools agent와 planning agent를 띄웠다.
- 2026-10-18: 기반 작업 완료, commit b5a4f98(branch push, main은 아직).
  - tools agent
    - dgmesh를 tools/로 옮겼다. 31장 그림은 바이트 단위로 같고, anim --check 27/27이다.
    - dgdec self-test 149 PASS. API와 규약은 dgdec docstring에 있다.
    - 확인된 부호: ∇Area_i = −(Lx)_i, (Lx)_i ≈ 2HN A_i, 단위구에서 H = −1.
    - Cr Exercise 5.4는 area gradient의 부호가 틀렸다. 41장에 Conventions가 필요하다.
    - mean_curvature의 기본값은 mixed area다. barycentric은 valence 5 꼭짓점에서 점별 수렴하지 않는다.
  - planning agent: 장 개요 14개
    - 개념의 정의 위치표와 Conventions 목록은 각 장 toc-note에 있다.
    - 다듬은 절 제목: 35.4 Stokes's Theorem, 38.1 Parametrized, 43.2 Equations, 43.5 Parametrization, 44.4 Logarithm Map
  - 통합
    - terms.tsv에 Track 2 용어 약 85개, references에 7개(Chern18, Alexandrov05, Borrelli03, Martinez05, Sharp19, Wang12, Max99)
    - index의 37·39 thread에 T5를 더했다.
    - §2.5에 "Track 2가 정의하는 smooth 개념" 목록을 넣었다(Track 3이 다시 정의하지 않게).
  - W1 작성 agent 3명을 띄웠다(32, 33, 34).
  - 통합 때 할 일: Tour 0.8 Where to Go Next와 31장 index·절에서 Track 2로 가는 링크, glossary `#sec-glossary-ddg`
- 2026-10-18: 32장 작성 완료(draft). 그림 11개, verify 4개(172 checks).
  - 결정
    - edge i는 γ_i→γ_{i+1}이다.
    - tangent angle은 φ로 쓴다(θ_i는 turning angle).
    - D flow는 pointwise이고 외심 방향이다. bisector normal을 쓰면 원이 둥글게 남지 않는다.
    - DRIFT를 정확히 진술했다. smooth flow에서는 무게중심이 움직인다(limaçon 예).
  - 통합
    - `tools/dgcode.py`(code_block, run_block)를 공용으로 올리고 §2.5에 코드 id 규칙을 적었다.
    - references에 GageHamilton86, Grayson87, terms.tsv에 17개를 추가했다.
  - 32장 검토 agent와 37장 작성 agent를 띄웠다. 지금 도는 것: 33·34 작성, 32 검토, 37 작성
  - 나중에: dgsym.EXAMPLES에 ellipse와 flower를 넣는다(지금은 ddg32.py에 있다). D curvature vector를 dgdec에 넣는다.
- 2026-10-18: 사용량 한도로 네 agent(33·34 작성, 32 검토, 37 작성)가 멈췄다(429, 07:30 초기화). cron 점검에서 SendMessage로 모두 재개했다.
- 2026-10-18: 33장 작성 완료(draft). 6절, 그림 14개(interactive 3개), verify 253 checks.
  - 계약과 다른 점
    - Prop 33.5.14(halfedge mesh는 닫힌 가향 곡면)는 Proof sketch로 썼다.
    - Prop 33.6.5(c)(dual = (H, η, ρ∘η))를 더했다.
  - 기호
    - 단체복합체 𝒦, St/Cl/Lk/bd/int, 부호 접속 행렬 A_k^±
    - halfedge H, η, ρ(33.5–33.6에서만 쓴다)
    - vertex orbit은 next∘twin(Crane)이다. dgdec는 반대 방향이다.
  - 통합: terms 20개, Guibas85
  - 검토 R1(33.1–33.3)과 R2(33.4–33.6)를 띄웠다. 코드 블록을 §2.5의 id 방식과 dgcode로 바꾸게 했다.
- 2026-10-18: 34장 작성 완료(draft). 5절, 그림 10개(interactive 3개), verify 388 checks.
  - 계약과 다른 점
    - basis 정리를 34.1로 옮겼다(Theorem 34.1.11).
    - k-vector를 규칙 (W1)–(W4)로 정의하고, 34.5에서 존재와 유일성을 보였다.
    - k-form은 strong이다(17.1 링크). verify는 결정식 규약의 쐐기곱이 ((k+l)!/(k!l!))Alt(ω⊗η)와 같음을 확인한다.
  - orthogonal complement는 Chapter 1에 정의가 없다. 34.2가 정의하기로 하고 §2.5 목록에 넣었다. 검토자가 dfn으로 바꾼다.
  - 35장에 넘길 것(극좌표·구면의 Hodge star, 단체 위 상수 형식의 적분 (1/k!)ω(…))은 35장 작성 프롬프트에 넣었다.
  - terms 15개. 35장 작성 agent를 띄웠다. 34장 검토는 동시 agent 수 때문에 다음 빈자리에 띄운다.
- 2026-10-18: **32장 reviewed.** 검토가 10개 항목을 고쳤다.
  - 32.4의 B 흐름 설명이 거꾸로였다(긴 edge 옆 vertex가 가장 많이 움직인다). 고치고 verify를 더했다.
  - 로드맵 그림 32.1.2의 edge가 상자 뒤로 지나가 없는 의존을 보였다. 다시 그리고 assert를 넣었다.
  - Prop 32.3.10의 가정(arc-length sampling)과 그림의 t-uniform sampling 차이를 문단으로 설명했다(jitter를 주면 1차).
  - "강의는"을 "강의 슬라이드는"으로 고쳤고, 슬라이드에 없는 인용을 바로잡았다.
  - Prop 5.6.12 인용, "counterclockwise" 가정 제거, 400px 수식 9개
  - 남은 제안: 32.3은 길다(분할은 작성 단계의 결정이라 그대로 둔다). 37.4는 Σθ = 2πk를 다시 증명하지 말고 32.3.4를 인용한다.
  - 통합: §2.5 목록에 curve shortening flow, terms 5개, `tools/ddg_glossary.py`(glossary `#sec-glossary-ddg` 자동 생성. 통합 때마다 실행한다), root index 상태
  - commit efc1635(branch push)
  - 34장 검토 agent를 띄웠다. 지금 도는 것: 33 검토 2명, 34 검토, 35·37 작성
- 2026-10-18: 33장 R1(33.1–33.3) 완료. 수학 오류는 없었다.
  - 저작권 문제: Crane의 예제와 문장에 너무 가까운 곳이 4곳 있었다(Example 33.3.9, 33.3 Motivation, Non-example 33.2.3(a), Def 33.3.11). 모두 새로 썼다. §2.5에 "예제도 그대로 쓰지 않는다"를 추가했다. 앞으로 작성 프롬프트에 강조한다.
  - orientation 코드 box의 잘못된 진술, Warning 2곳, 400px 수식을 고쳤다. 코드 블록은 id 방식으로 바꿨다(`out-N-M-K` 추가).
- 2026-10-18: **33장 reviewed**(R2 33.4–33.6 완료). 수학 오류는 없었다.
  - vertex orbit 규약: L2B 슬라이드는 η∘ρ, 노트는 ρ∘η다. Conventions box를 보강했다.
  - Prop 33.6.5(c)의 증명을 보완했다.
  - 근거 없는 진술 5곳을 약하게 고치거나 출처를 달았다.
  - 33.4 discrete box의 틀린 진술("edge 위의 점은 문제없다")을 고쳤다.
  - lib/ch33의 옛 코드 실행기를 지웠다.
  - 남은 것: 33.4와 33.5는 길다(분할하려면 번호가 바뀐다). Example 33.5.2·33.5.15는 규약상 Crane과 같을 수밖에 없어서 출처를 달았다.
  - 통합
    - Tour 0.8 Where to Go Next와 31장 Leads to에서 Track 2로 가는 링크를 달았다.
    - terms: rotation system
  - commit f733507(branch push). 지금 도는 것: 34 검토, 35·37 작성
- 2026-10-18: **34장 reviewed.** 검토가 고친 것
  - 연습 34.1.3 풀이의 계산 오류(16−3 → 16−12)
  - (34.2.5)의 "동치"를 "특히"로 낮췄다.
  - Non-example 34.1.2에 (c)·(d)를 더했다(규칙 넷이 모두 필요함을 보임).
  - orthogonal complement를 dfn으로 바꿨다.
  - KN 귀속을 "일부 교재"로 바꿨다.
  - 34.2 Motivation이 Cr §4.1.3을 너무 가깝게 옮긴 문장이었다. 새로 썼다.
  - interactive 34.1.3의 카메라
  - 통합: notation `#sec-ddg`에 32–34장 기호 9행, Alt 정규화 문헌 표기를 "일부 교재"로 바꿨다.
  - **공개 절차**: 작성 중인 장(35, 37)이 생성 페이지에 섞이지 않게 한다. `git add`한 뒤 `/home/kbh/.claude/jobs/6f56e074/tmp/clean_build.sh`가 HEAD와 staged 파일로 만든 깨끗한 트리에서 glossary와 concepts를 만들고 검사한다. 그다음 commit하고 `git push origin worktree-ddg-cmu:main`
- 2026-10-18: **공개 1차**: origin/main = b8c9dce(fast-forward). Pages "built b8c9dce", 34.2 페이지 200.
  - 세 track 목차, Track 2 32–34장(reviewed), 35–45장 개요(planned)
  - dg-study-main 세션에 "main이 앞서 갔으니 pull --ff-only 하라"고 알렸다.
  - Pages 확인은 `gh api repos/Rhqo/DG-study/pages/builds/latest`로 한다. 인증 없는 API는 404다.
- 2026-10-18: 37장 작성 완료(draft). 6절, 그림 15개(interactive 1개), verify 267 checks.
  - 결정
    - Crane의 Frenet–Serret 부호를 정리한 표(τ_Crane = −τ)
    - discrete torsion은 dgdec의 부호를 따른다.
    - D(차분 행렬)는 Chapter 36이 생기면 d_0으로 바꿀 수 있다.
    - 새 예제: figure-eight, limaçon, trefoil(holonomy 3.0270)
  - 통합: references 7개(Whitney37, Bishop75, MehlhornYap91, Crane13f, Yu21, Hasimoto72, Padilla19), terms 13개
  - 검토 R1(37.1–37.3)과 R2(37.4–37.6)를 띄웠다. 35장 작성은 계속 돌고 있다.
  - todo 표기: "NN장 미작성"은 check_site가 한국어 참조로 경고한다. "Chapter NN 미작성"으로 쓴다(§15.5 문구를 고칠 것).
- 2026-10-18: 두 번째로 사용량 한도에 걸렸다(12:30 초기화). 35 작성과 37 R1·R2 검토를 SendMessage로 재개했다. 사용자도 "다시 시작"이라고 했다.
- 2026-10-18: 35장 작성 완료(draft). 5절, 그림 11개, verify 430 checks. dfn은 L2 inner product, codifferential, vector-valued form뿐이다.
  - **42.1 작성자에게 넘길 것**
    - d*는 L2 adjoint다: d* = (−1)^{n(k+1)+1}⋆d⋆.
    - n = 2에서 Crane의 δ = −d*이다.
    - n = 3의 1-form에서 Crane의 δd + dδ는 (curl curl + grad div)♭이다. 우리 dd* + d*d는 −(ΔX)♭다(Ex 35.5.6).
  - **36장에 넘길 것**
    - 단체 적분 (35.3.6–7)과 Prop 35.3.12(상수 ω면 (1/k!)ω(…), 순서를 바꾸면 sgn π)
    - 점 적분(Remark 35.3.14)
    - ∂σ = Σ(−1)^m(…)(Prop 35.4.4 = 33.3.7)
    - Stokes(edge, triangle: Cor 35.4.11)
    - ∂∂ = 0의 증명은 36.2로 남겼다.
  - 38.3은 N dA = ½dx∧dx와 vector area를 맡는다. 35.5는 sphere 값만 주었다.
  - terms 반영. 35 검토, 36 작성, 38 작성을 띄웠다.
- 2026-10-18: **37장 reviewed.**
  - R1(37.1–37.3): 수학 오류는 없었다.
    - holonomy 진술과 Prop 37.3.9의 Lemma 적용(열린 구간)을 고쳤다.
    - Crane의 Frenet box를 원문과 대조해 보강했다.
    - Whitney37·Bishop75를 cite로 바꿨다.
  - R2(37.4–37.6)가 고친 수학 오류
    - flower에서 가정이 성립하는 n(n ≥ 24가 아니라 10, 20, n ≥ 23이다)
    - κᴮ·κᶜ로도 곡선이 결정된다(닫힘 조건만 비선형이 된다). κᴰ로는 θ를 되찾을 수 없다.
  - R2가 고친 그 밖의 것
    - Figure 37.6.1의 설명이 그림과 맞지 않았다. 측정한 사실로 다시 썼다.
    - smoke-ring 식의 부호
    - ψ 기호 충돌
  - 남은 REVIEW 2개
    - Fig 37.6.1(a)를 edge-length plot으로 바꾸자는 제안
    - 37.1의 Looking ahead를 36장 뒤에 Looking back으로 바꾸는 것
    - Track 2 통합 검토에서 처리한다.
  - 통합
    - style.css: 좁은 화면에서 `.thm td`가 숫자를 자르지 않게 했다.
    - notation: 35·37장 기호 9행
- 2026-10-18: 37장 commit(branch만). 37장이 35장 절을 링크하므로 main 공개는 35장 검토가 끝난 뒤 함께 한다.
- 2026-10-18: **35장 reviewed.** 정리, 값, 부호 오류는 없었다.
  - 검토가 고친 것
    - 35.5의 Crane 규약 box를 원문과 맞췄다. L7 slide 29는 L2 inner product를 ∫⋆α∧β로 정의하는데, 이것은 우리 정의와 부호 (−1)^{k(n−k)}만큼 다르다. Cr Ex 8.4는 2D에서 부호가 틀렸다.
    - box에서 ∂∂ = 0의 증명을 36.2로 미뤘던 것을 35.4에서 직접 증명했다. 36.2는 simplicial chain만 맡는다.
    - Fig 35.3.1의 이유를 바로잡았다. 0인 값은 출발점에서 x = 0이기 때문이다.
    - ∫fΔf ≤ 0이 성립하는 조건(compact support)
  - **42.1 작성자에게**: Crane 노트는 Laplacian 부호가 일관되지 않다(Ex 7.6은 −⋆d⋆d, §4.6.6과 L6 slide 36은 ⋆d⋆d).
  - **45 작성자에게**: Cr Ex 8.4–8.6은 부호 없는 δ를 쓴다. Ex 8.6의 "δ(⋆φ)는 ∇φ를 반시계로 돌린 것"은 우리 d*로는 시계 방향이다(2D에서 δ = −d*).
  - 통합: 34.4의 discrete box가 Prop 35.3.12로 링크한다.
- 2026-10-18: **공개 2차**: origin/main = 573457e(35·37장 reviewed), Pages built, 37.6 페이지 200. 지금 도는 것: 36·38 작성
- 2026-10-18: **사용자 지시: "일단 지금 하는 부분까지만 작성하고 나머지는 다음에 하자."**
  - 진행 중인 36·38장 작성만 끝낸다. 새 웨이브와 검토는 시작하지 않는다.
  - 36·38장은 draft 상태로 통합, commit, push한다.
  - cron a30f1fce를 지우고, 마무리 전용 cron(매시 23분)으로 바꿨다. 마무리가 끝나면 이것도 지운다.
  - **다음에 할 일** (다시 시작할 때, 우선순위 순)
    1. 36장 검토(2명), 38장 검토(2명). 36장이 쓰이면 37·38장의 Chapter 36 todo 링크를 확정한다.
    2. W4: 39, 42 → W5: 40, 43 → W6: 41, 44 → W7: 45 (§13 웨이브 계획, 작성 프롬프트는 §13.1, 검토는 §13.2)
    3. Track 2 통합 검토: 남은 REVIEW(37.6 Fig 37.6.1(a), 37.1 Looking ahead → back), 31장·Tour 링크 점검
    4. 애니메이션 단계(§12.8). 각 장 작성자의 제안이 이 기록과 보고에 있다.
    5. 각 장 작성 프롬프트에 넣을 인계 사항은 위 기록의 "~ 작성자에게" 항목에 있다(42.1 부호, 45 δ 부호 등).
- 2026-10-18: **36장 작성 완료(draft).** 6절, 그림 16개(interactive 2개), verify 247 checks.
  - 다음 장이 쓸 항목
    - d0·d1(Def 36.4.1, Prop 36.4.4)
    - discrete Stokes(Thm 36.4.5), R∘d = d∘R(Thm 36.4.7), d1d0 = 0(Thm 36.4.8)
    - 쌍대 방향(Remark 36.5.1)
    - cotan formula(Thm 36.5.10), ⋆_k(Def 36.5.14)
    - d*(Def 36.6.2), L(Def 36.6.6, 유일한 정의), Delaunay PSD(Prop 36.6.9)
    - 일반 PSD는 42.3이 맡는다.
  - Non-example 36.5.18: Crane의 부호 없는 ⋆1은 non-Delaunay edge에서 Thm 36.5.16을 깨뜨린다.
  - 통합
    - 37장의 36장 todo 3개를 확정했다.
    - 38장 작성자에게 36장 API를 보냈다.
    - terms 10개, notation 4행
  - 다음에: lib/ch36.py의 dual_d0/dual_d1/left_right_faces를 dgdec로 옮기는 것을 검토한다. 36장 검토(2명).
- 2026-10-18: **38장 작성 완료(draft).** 6절, 그림 18개(interactive 3개), verify 630 checks, todo 0.
  - 주요 결과
    - vertex normal 4종(Def 38.5.6)
    - 정규화 없이 subdivision에 불변인 것은 angle-weighted뿐이다(Prop 38.5.8).
    - 구면에서 Δ_h z ≈ −2z다. barycentric ⋆0로는 수렴하지 않는다(Ex 38.6.6).
    - shape from normals(Thm 38.6.8, 증명), dimpled icosahedron(edge 길이가 같아도 부피가 다르다)
  - 규약
    - ⋆dx = −N×dx(Crane: ⋆df = N×df)
    - e^{2μ}
  - **검토 때 볼 것**
    - 38장은 mesh vertex를 굵은 𝐱_i로 썼다. §2.5와 31장은 x_i다. 통일 여부를 정한다.
    - 36장 번호를 바꾸지 않았는지 다시 확인한다.
  - 통합
    - 36장에서 38장으로 가는 링크 7개를 실제 절로 바꿨다.
    - terms 14개, Smale59, notation 3행
    - root index Draft
- 2026-10-18: **마무리 완료, 작업을 멈춘다(사용자 지시).**
  - origin/main = ae2e707, Pages built, 38.6 페이지 200. 전체 check_all OK.
  - Track 2 현황
    - reviewed: 32, 33, 34, 35, 37
    - draft: 36, 38
    - planned: 39–45
  - 마무리 cron d4de8017을 지웠다. 자동 재개 cron은 이제 없다.
  - 다시 시작할 때는 위의 "다음에 할 일" 목록부터 한다.
- 2026-10-18: **사용자 지시: "36, 38은 검토하고 commit push 까지만 하자. 나머지 39–45는 나중에 하자."**
  - 검토 agent 4명을 띄웠다: 36 R1(36.1–36.3), 36 R2(36.4–36.6), 38 R1(38.1–38.3), 38 R2(38.4–38.6).
  - 굵은 𝐱_i 표기: 38장처럼 discrete immersion으로 볼 때는 허용한다. 한 장 안에서는 하나로 쓴다. §2.5에 적었다.
  - 이번 범위는 이 검토와 commit, push까지다. 39–45장은 하지 않는다.
- 2026-10-18: 36장 R1(36.1–36.3) 완료. 수학 오류는 없었다.
  - 35장이 "그림에서 ∂∂ = 0을 봤다"고 한 진술을 바로잡았다.
  - L8 slide 36의 Whitney k-form에는 k!이 없다는 출처 주석을 넣었다. 그대로 쓰면 R∘W = id가 깨진다.
  - Thm 36.3.10에서 R_k를 적용하는 방법과 Example 36.2.15의 각을 보완했다.
  - "유일한 최저 차수 보간"이라는 과장을 고쳤다.
  - 통합: 35.4의 36.2 링크 3개를 Thm 36.2.8로 바로 잇게 했다.
- 2026-10-18: 38장 R1(38.1–38.3) 완료. 수학 오류는 없었다. ⋆dx = −N×dx를 Def 34.2.5로 직접 확인했다.
  - Crane §7.3의 ⋆α = α∘J는 1-form에서 우리와 부호가 반대다. Crane §4의 ⋆는 우리와 같다. Conventions box를 §7에만 해당하게 고쳤다.
  - **43장 작성자에게**: Crane의 ⋆dz = i dz는 우리 규약으로 ⋆dz = −i dz다.
  - 저작권: 비유와 번역에 가까운 문장 3곳을 새로 썼다. Ex 38.3.5(b)에 [Cr §5.1 변형] 표시를 달았다.
  - 인용 정확성 5곳을 고쳤다. parametrized surface를 dfn으로 바꿨다. 독자가 볼 수 없는 "§7" 참조를 notation 링크로 바꿨다.
  - 통합: notation 38장 행(굵은 𝐍_f, 𝐍_i^w, Crane ⋆의 §4/§7 구분, e^u/e^{2u})
  - 제안(사용자 승인 필요, 보류): GUIDELINES §6.6에 표면 위 1-form Hodge star의 Crane 비교 행을 넣는 것
- 2026-10-18: **36장 reviewed**(R2 36.4–36.6 완료).
  - 수학 오류: Ex 36.6.4(b)의 부호(β = −J grad ψ)
  - 그 밖에 고친 것
    - 36.5가 d0ᵀ⋆1d0를 "cotan Laplacian"이라고 부른 것을 −L로 바로잡았다.
    - 경계가 있는 mesh에서 증명 없이 d̃1 = −d0ᵀ를 쓰고 있었다. 내부 행에서는 성립함을 밝히고, 경계 행은 정의로 채택했다(42.5로 넘김).
    - Lemma와 Proposition 레이블, (36.6.1)의 근거, L9 slide 32의 부호 차이
    - Prop 36.6.9(c)의 표현, w ≥ 0인 경우의 PSD
    - 그림 4개
  - 다음에 할 것
    - dgmesh.plotly_save: 400px에서 3D interactive가 좌우로 잘린다(카메라·autosize).
    - check_site: 링크 글자와 대상 제목이 맞는지 검사하는 기능(검토자의 linklabels.py)
  - 통합: 독자가 볼 수 없는 GUIDELINES 절 번호(§2.5, §6.x, §7) 36곳을 notation.html 링크로 바꿨다(32–37장, 조사 보정). 38.4는 R2에게 맡겼다.
- 2026-10-18: **38장 reviewed**(R2 38.4–38.6 완료).
  - 수학 오류
    - Prop 38.4.11: 0 < θ_k < π가 필요하다(θ = π면 face가 퇴화한다).
    - Prop 38.5.8(d): 증명의 조건을 고쳤다. mixed-Voronoi 법벡터도 subdivision에서 22.00° 움직인다.
    - Prop 38.5.8(c): triangulation 독립성으로 고쳐 진술하고 (38.5.6)으로 새로 증명했다.
    - Prop 38.5.8(b): 인용을 고쳤다.
  - 그 밖에 고친 것
    - 과장된 진술 6곳
    - 번호와 각도 2곳
    - \phi를 \varphi로 바꿨다. L과 겹쳐서 x̃로 바꿨다.
    - flat edge를 dfn으로 바꿨다.
    - 38.6 Motivation이 슬라이드 순서를 따라가던 것을 새로 썼다.
    - Fig 38.6.1의 글자 크기
  - 남은 판단
    - C_i가 38.4에서는 closed star, 38.5에서는 dual cell을 뜻한다. 절마다 밝혀 두었다. 40·42장 계획도 C_i를 dual cell로 쓴다.
  - 통합: terms(flat edge), notation(x̃, C_i), root index 36·38 Reviewed
