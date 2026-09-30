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
      sgn: '\\operatorname{sgn}'
    }
  }
};
