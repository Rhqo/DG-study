"""verify 스크립트용 검사 헬퍼 (GUIDELINES.md §13).

사용법::

    from dgcheck import check, sym_equal, close, summary
    check("예 7.1.2: 구면 F = 0", F == 0)
    sym_equal("예 7.1.2: 구면 G", G, r**2 * sp.sin(th)**2, {th: (0, sp.pi)})
    close("예 9.4.1: 회전각", angle, 2*np.pi*np.cos(th0), tol=1e-8)
    summary()          # 실패가 하나라도 있으면 exit 1

출력 형식은 한 줄에 하나씩 ``PASS  <이름>`` 또는 ``FAIL  <이름> — <설명>``이다.
"""

import random
import sys

_passed = 0
_failed = 0
_failed_names = []


def _report(name, ok, detail=""):
    global _passed, _failed
    if ok:
        _passed += 1
        print(f"PASS  {name}")
    else:
        _failed += 1
        _failed_names.append(name)
        print(f"FAIL  {name}" + (f" — {detail}" if detail else ""))
    return ok


def check(name, cond, detail=""):
    """조건 ``cond``가 참인지 검사한다. sympy 관계식도 받는다."""
    try:
        ok = bool(cond)
    except TypeError as exc:  # sympy가 참·거짓을 정하지 못한 관계식
        return _report(name, False, f"조건을 판정할 수 없음: {exc}")
    return _report(name, ok, detail)


def _as_list(x):
    import sympy as sp
    if isinstance(x, (sp.MatrixBase, sp.NDimArray)):
        return list(sp.flatten(x.tolist())), x.shape
    if isinstance(x, (list, tuple)):
        flat = list(sp.flatten(x))
        return flat, (len(flat),)
    return [sp.sympify(x)], ()


def sym_equal(name, a, b, symbols_domain=None, samples=6, tol=1e-9, seed=0):
    """두 sympy 식(스칼라, 행렬, 중첩 리스트)이 같은지 검사한다.

    1. ``simplify(a - b) == 0``이 되면 통과.
    2. 아니면 ``symbols_domain``({기호: (하한, 상한)})의 열린 구간 안에서
       무작위 수치 대입을 ``samples``번 해서 차이가 ``tol`` 이하인지 본다.
       구간을 주지 않은 자유 기호는 (0.3, 1.7)에서 뽑는다.
    """
    import sympy as sp

    la, sa = _as_list(a)
    lb, sb = _as_list(b)
    if len(la) != len(lb):
        return _report(name, False, f"모양이 다름: {sa} vs {sb}")

    diffs = [sp.sympify(x) - sp.sympify(y) for x, y in zip(la, lb)]
    unresolved = []
    for d in diffs:
        try:
            s = sp.simplify(d)
        except Exception:  # pragma: no cover - simplify 실패는 수치 검사로 넘김
            s = d
        if s != 0:
            unresolved.append(s)
    if not unresolved:
        return _report(name, True)

    domain = dict(symbols_domain or {})
    free = set()
    for d in unresolved:
        free |= d.free_symbols
    rng = random.Random(seed)
    worst = 0.0
    for _ in range(samples):
        point = {}
        for s in free:
            lo, hi = domain.get(s, (0.3, 1.7))
            lo, hi = float(lo), float(hi)
            # 열린 구간 안쪽에서만 뽑는다 (경계의 특이점 회피).
            margin = 0.05 * (hi - lo)
            point[s] = rng.uniform(lo + margin, hi - margin)
        for d in unresolved:
            try:
                val = complex(sp.N(d.subs(point)))
            except (TypeError, ValueError) as exc:
                return _report(name, False, f"수치 대입 실패 ({exc}); 남은 식: {d}")
            worst = max(worst, abs(val))
    ok = worst <= tol
    detail = "" if ok else f"최대 차이 {worst:.3e}; 단순화 후 남은 식 예: {unresolved[0]}"
    return _report(name, ok, detail)


def close(name, a, b, tol=1e-9):
    """수치(스칼라 또는 numpy 배열) ``a``, ``b``가 ``tol`` 이내로 같은지 검사한다."""
    import numpy as np

    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)
    if a.shape != b.shape:
        return _report(name, False, f"모양이 다름: {a.shape} vs {b.shape}")
    err = float(np.max(np.abs(a - b))) if a.size else 0.0
    ok = err <= tol
    return _report(name, ok, "" if ok else f"최대 차이 {err:.3e} > tol {tol:g}")


def summary():
    """결과를 요약하고, 실패가 있으면 종료코드 1로 끝낸다."""
    print(f"{_passed} passed, {_failed} failed")
    if _failed:
        for n in _failed_names:
            print(f"  failed: {n}")
        sys.exit(1)
