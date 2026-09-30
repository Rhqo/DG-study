#!/usr/bin/env node
// DG-study 수식(TeX) 검사기 (GUIDELINES.md §11.4, 부록 D).
//
// 사용법:
//   node tools/check_math.mjs [--root DIR] [파일 ...]
//
// 페이지에서 \( … \)와 \[ … \]를 뽑아 MathJax(mathjax-full 3.2.2)의 TeX 입력기로 변환해 본다.
// 브라우저와 같은 매크로를 쓰도록 assets/mathjax-config.js를 node:vm에서 실행해 매크로를 얻는다.
// TeX 패키지는 base, ams, amscd, boldsymbol, newcommand, configmacros만 쓴다.
// noundefined·autoload를 빼서 정의되지 않은 매크로가 오류가 되게 한다.
// 출력: `ERROR 경로:줄: 메시지 in: <tex>`. 오류가 있으면 종료코드 1.
// 준비: cd tools && npm install

import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

let mathjax, TeX, liteAdaptor, RegisterHTMLHandler;
try {
  ({ mathjax } = await import('mathjax-full/js/mathjax.js'));
  ({ TeX } = await import('mathjax-full/js/input/tex.js'));
  ({ liteAdaptor } = await import('mathjax-full/js/adaptors/liteAdaptor.js'));
  ({ RegisterHTMLHandler } = await import('mathjax-full/js/handlers/html.js'));
  await import('mathjax-full/js/input/tex/base/BaseConfiguration.js');
  await import('mathjax-full/js/input/tex/ams/AmsConfiguration.js');
  await import('mathjax-full/js/input/tex/amscd/AmsCdConfiguration.js');
  await import('mathjax-full/js/input/tex/boldsymbol/BoldsymbolConfiguration.js');
  await import('mathjax-full/js/input/tex/newcommand/NewcommandConfiguration.js');
  await import('mathjax-full/js/input/tex/configmacros/ConfigMacrosConfiguration.js');
} catch (e) {
  console.log('ERROR tools: mathjax-full is not installed — run: cd tools && npm install');
  process.exit(1);
}

const PACKAGES = ['base', 'ams', 'amscd', 'boldsymbol', 'newcommand', 'configmacros'];
const SKIP_DIRS = new Set(['tools', 'refs', 'assets', 'node_modules', '.git']);

// ---------------------------------------------------------------- 인자

const here = path.dirname(fileURLToPath(import.meta.url));
let root = path.dirname(here);
const files = [];
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i++) {
  if (argv[i] === '--root') root = argv[++i];
  else if (argv[i] === '-h' || argv[i] === '--help') {
    console.log('usage: node tools/check_math.mjs [--root DIR] [files...]');
    process.exit(0);
  } else files.push(path.resolve(argv[i]));
}
root = path.resolve(root);

function sitePages(dir, top = true) {
  const out = [];
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    if (ent.name.startsWith('.')) continue;
    const p = path.join(dir, ent.name);
    if (ent.isDirectory()) {
      if (top && SKIP_DIRS.has(ent.name)) continue;
      if (ent.name === 'node_modules') continue;
      out.push(...sitePages(p, false));
    } else if (ent.name.endsWith('.html') && !ent.name.endsWith('-interactive.html')) {
      out.push(p);
    }
  }
  return out.sort();
}

// ---------------------------------------------------------------- MathJax 준비

function loadMacros() {
  const cfgPath = path.join(root, 'assets', 'mathjax-config.js');
  const ctx = { window: {} };
  vm.runInNewContext(fs.readFileSync(cfgPath, 'utf8'), ctx, { filename: cfgPath });
  const mj = ctx.window.MathJax;
  if (!mj || !mj.tex) throw new Error('assets/mathjax-config.js did not set window.MathJax.tex');
  // vm 컨텍스트 객체를 일반 객체로 복사한다
  return JSON.parse(JSON.stringify(mj.tex.macros || {}));
}

let macros;
try {
  macros = loadMacros();
} catch (e) {
  console.log(`ERROR assets/mathjax-config.js:0: cannot load macros: ${e.message}`);
  process.exit(1);
}

RegisterHTMLHandler(liteAdaptor());
function newDocument() {
  const tex = new TeX({
    packages: PACKAGES,
    macros,
    formatError: (jax, err) => { throw err; },
  });
  return mathjax.document('', { InputJax: tex });
}
let doc = newDocument();

// ---------------------------------------------------------------- 추출

const BLANK_RE = /<!--[\s\S]*?-->|<head\b[^>]*>[\s\S]*?<\/head\s*>|<(script|style|code|pre|textarea|noscript)\b[^>]*>[\s\S]*?<\/\1\s*>/gi;

function blankRegions(raw) {
  return raw.replace(BLANK_RE, (m) => m.replace(/[^\n]/g, ' '));
}

const NAMED = { lt: '<', gt: '>', amp: '&', quot: '"', apos: "'", nbsp: ' ' };
function decodeEntities(s) {
  return s.replace(/&(#x[0-9a-fA-F]+|#\d+|[a-zA-Z]+);/g, (m, e) => {
    if (e[0] === '#') {
      const code = e[1] === 'x' || e[1] === 'X' ? parseInt(e.slice(2), 16) : parseInt(e.slice(1), 10);
      return String.fromCodePoint(code);
    }
    return Object.prototype.hasOwnProperty.call(NAMED, e) ? NAMED[e] : m;
  });
}

const unescaped = (t, i) => i === 0 || t[i - 1] !== '\\';

// check_site.find_math_spans와 같은 규칙: 백슬래시가 앞에 붙은 \( \[ 는 여는 기호가 아니다 (예: \\[4pt])
function findMathSpans(text) {
  const spans = [];
  const opener = /\\[([]/g;
  let m;
  while ((m = opener.exec(text)) !== null) {
    const p = m.index;
    if (!unescaped(text, p)) continue;
    const display = text[p + 1] === '[';
    const closeTok = display ? '\\]' : '\\)';
    const openTok = display ? '\\[' : '\\(';
    let j = p + 2;
    let end = -1;
    let nested = false;
    for (;;) {
      const c = text.indexOf(closeTok, j);
      if (c === -1) break;
      if (!unescaped(text, c)) { j = c + 2; continue; }
      const o = text.indexOf(openTok, p + 2);
      if (o !== -1 && o < c && unescaped(text, o)) nested = true;
      end = c;
      break;
    }
    if (end === -1) {
      spans.push({ start: p, display, tex: null, error: `unclosed ${openTok} (no matching ${closeTok})` });
      continue;
    }
    spans.push({
      start: p, display, tex: text.slice(p + 2, end),
      error: nested ? `${openTok} opened again before the previous one was closed` : null,
    });
    opener.lastIndex = end + 2;
  }
  return spans;
}

const lineOf = (text, pos) => text.slice(0, pos).split('\n').length;
const short = (s) => { const t = s.replace(/\s+/g, ' ').trim(); return t.length > 90 ? t.slice(0, 87) + '...' : t; };

// ---------------------------------------------------------------- 검사

const targets = files.length ? files : sitePages(root);
let nFormulas = 0;
const errors = [];

for (const file of targets) {
  const rel = path.relative(root, file).split(path.sep).join('/');
  let raw;
  try { raw = fs.readFileSync(file, 'utf8'); } catch (e) { errors.push(`ERROR ${rel}:0: cannot read file`); continue; }
  const text = blankRegions(raw);
  for (const s of findMathSpans(text)) {
    const line = lineOf(text, s.start);
    if (s.tex === null) { errors.push(`ERROR ${rel}:${line}: ${s.error}`); continue; }
    if (s.error) errors.push(`ERROR ${rel}:${line}: ${s.error}`);
    if (s.tex.includes('<')) {
      // check_site가 원문 '<'를 따로 보고한다. 태그가 섞인 수식은 변환하지 않는다.
      errors.push(`ERROR ${rel}:${line}: raw '<' inside math (write \\lt) in: ${short(s.tex)}`);
      continue;
    }
    const tex = decodeEntities(s.tex);
    const def = tex.match(/\\(newcommand|renewcommand|def|let)\b/);
    if (def) {
      errors.push(`ERROR ${rel}:${line}: \\${def[1]} is not allowed in pages (request a macro for 부록 C) in: ${short(tex)}`);
      continue;
    }
    nFormulas++;
    try {
      doc.convert(tex, { display: s.display });
    } catch (e) {
      errors.push(`ERROR ${rel}:${line}: ${e.message || e} in: ${short(tex)}`);
      doc = newDocument(); // 오류 뒤 상태를 초기화
    }
  }
}

for (const e of errors) console.log(e);
console.log(`check_math: ${targets.length} pages, ${nFormulas} formulas, ${errors.length} errors`);
process.exit(errors.length ? 1 : 0);
