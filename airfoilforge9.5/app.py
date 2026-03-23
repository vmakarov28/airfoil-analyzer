"""
AirfoilForge Pro — Main launcher / Home page
"""

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="AirfoilForge Pro",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from components.shared import init_session_state
init_session_state()

# ══════════════════════════════════════════════════════════════════════════════
# ANIMATED AIRFOIL HERO
# Streamlines are TRUE GEOMETRIC OFFSET CURVES from the airfoil polygon.
# Each line is the airfoil contour expanded outward by a fixed normal distance,
# then smoothly blended into horizontal far-field lines. Zero physics hacks.
# ══════════════════════════════════════════════════════════════════════════════
components.html("""
<!DOCTYPE html>
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@700&family=Inter:wght@400&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{background:#070b12;overflow:hidden;}
#hero{
  position:relative;width:100%;height:420px;
  background:radial-gradient(ellipse 80% 55% at 50% 50%,
    rgba(231,76,60,0.05) 0%,transparent 65%), #070b12;
}
canvas{display:block;position:absolute;top:0;left:0;}
#tagline{
  position:absolute;top:18px;left:0;right:0;text-align:center;
  pointer-events:none;z-index:2;
}
#tagline h1{
  font-family:'Rajdhani',sans-serif;font-size:2.25rem;font-weight:700;
  letter-spacing:0.07em;
  background:linear-gradient(135deg,#fff 0%,#e74c3c 45%,#f39c12 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
#tagline p{font-family:'Inter',sans-serif;font-size:0.83rem;color:#46587a;
           margin-top:5px;letter-spacing:0.06em;}
.bdg{display:inline-block;
  background:linear-gradient(90deg,#e74c3c,#f39c12);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  font-size:0.58rem;font-weight:700;letter-spacing:0.1em;
  border:1px solid rgba(231,76,60,0.38);border-radius:10px;
  padding:1px 8px;vertical-align:middle;margin-left:8px;
}
#lbl{
  position:absolute;bottom:10px;left:0;right:0;text-align:center;z-index:3;
  pointer-events:none;font-family:'Rajdhani',sans-serif;font-size:1.28rem;
  font-weight:700;letter-spacing:0.20em;color:#ff7060;
  text-shadow:0 0 26px rgba(231,76,60,0.85),0 0 52px rgba(231,76,60,0.55);
  text-transform:uppercase;transition:opacity 0.5s ease;
}
</style>
</head>
<body>
<div id="hero">
  <div id="tagline">
    <h1>AirfoilForge Pro <span class="bdg">FREE · UNLIMITED</span></h1>
    <p>1,638+ UIUC airfoils &nbsp;·&nbsp; NeuralFoil-powered analysis &nbsp;·&nbsp; Interactive Plotly charts</p>
  </div>
  <canvas id="c"></canvas>
  <div id="lbl">NACA 0012</div>
</div>
<!-- REPLACE THE ENTIRE <script> ... </script> BLOCK inside the first components.html with this updated version -->
<script>
// ─── canvas setup ─────────────────────────────────────────────────────────────
const CV = document.getElementById('c'), ctx = CV.getContext('2d');
const HR = document.getElementById('hero'), LBL = document.getElementById('lbl');
function resize(){CV.width = HR.offsetWidth; CV.height = HR.offsetHeight;}
resize(); window.addEventListener('resize', resize);

// ─── NACA 4-digit generator ──────────────────────────────────────────────────
function naca4(m, p, t, n=180){
  const upper = [], lower = [];
  for(let i = 0; i <= n; i++){
    const b = Math.PI * i / n;
    const x = 0.5 * (1 - Math.cos(b));
    const yt = 5 * t * (0.2969 * Math.sqrt(x) - 0.126 * x - 0.3516 * x*x + 0.2843 * x*x*x - 0.1015 * x*x*x*x);
    let yc = 0, dyc = 0;
    if(m > 0 && p > 0){
      if(x < p){ yc = m / p / p * (2 * p * x - x * x); dyc = 2 * m / p / p * (p - x); }
      else     { yc = m / (1 - p) / (1 - p) * ((1 - 2 * p) + 2 * p * x - x * x); dyc = 2 * m / (1 - p) / (1 - p) * (p - x); }
    }
    const th = Math.atan2(dyc, 1);
    upper.push([x - yt * Math.sin(th), yc + yt * Math.cos(th)]);
    lower.push([x + yt * Math.sin(th), yc - yt * Math.cos(th)]);
  }
  return {upper, lower};
}

const PROFILES = [
  {m:0,   p:0,  t:.12, n:"NACA 0012"},
  {m:.02, p:.4, t:.12, n:"NACA 2412"},
  {m:.04, p:.4, t:.14, n:"NACA 4414"},
  {m:.06, p:.4, t:.09, n:"NACA 6409"},
  {m:0,   p:0,  t:.18, n:"NACA 0018"},
  {m:.02, p:.3, t:.08, n:"NACA 2308"},
  {m:.05, p:.5, t:.12, n:"NACA 5512"},
  {m:.03, p:.4, t:.15, n:"NACA 3415"},
];
const PD = PROFILES.map(p => ({d: naca4(p.m, p.p, p.t), n: p.n}));

function lF(a,b,t){return a+(b-a)*t;}
function ease(t){return t<.5?2*t*t:-1+(4-2*t)*t;}
function smoothStep(t){t=Math.max(0,Math.min(1,t));return t*t*(3-2*t);}

function lerpAirfoil(A,B,t){
  const n=Math.min(A.upper.length,B.upper.length);
  const upper=[],lower=[];
  for(let i=0;i<n;i++){
    upper.push([lF(A.upper[i][0],B.upper[i][0],t),lF(A.upper[i][1],B.upper[i][1],t)]);
    lower.push([lF(A.lower[i][0],B.lower[i][0],t),lF(A.lower[i][1],B.lower[i][1],t)]);
  }
  return {upper,lower};
}

// ─── GEOMETRIC OFFSET STREAMLINES — FINAL BULLETPROOF VERSION ────────────────
function buildStreamlinesForAirfoil(af){
  const lines = [];
  const N_SAMPLES = 420;
  const X_START   = -2.8;
  const X_END     =  3.8;
  const N_LEVELS  = 3;           // ← ONLY 2 TOP + 2 BOTTOM
  const H_MAX     = 0.2;        // slightly larger spacing for clean look

  const contour = [];
  for(let i=0;i<af.upper.length;i++) contour.push(af.upper[i]);
  for(let i=af.lower.length-2;i>=0;i--) contour.push(af.lower[i]);

  const n = contour.length;
  let cx=0, cy=0;
  for(const p of contour){ cx += p[0]; cy += p[1]; }
  cx /= n; cy /= n;

  const normals = [];
  for(let i=0;i<n;i++){
    const a = contour[(i+n-1)%n];
    const b = contour[(i+1)%n];
    let nx = -(b[1]-a[1]), ny = b[0]-a[0];
    let len = Math.sqrt(nx*nx + ny*ny) + 1e-12;
    nx /= len; ny /= len;
    if(nx*(contour[i][0]-cx) + ny*(contour[i][1]-cy) < 0){ nx = -nx; ny = -ny; }
    normals.push([nx, ny]);
  }

function ramp(x){
    const entryS = -1.6, entryE = -0.05;   // wider ramp = more curve on outer lines
    const exitS  =  1.05, exitE  =  2.7;
    if(x < entryS || x > exitE) return 0;
    if(x > entryE && x < exitS) return 1;
    if(x < entryE)  return smoothStep((x-entryS)/(entryE-entryS));
    return smoothStep(1-(x-exitS)/(exitE-exitS));
  }

  for(let side=0; side<2; side++){
    const isUpper = side === 0;
    for(let k=1; k<=N_LEVELS; k++){
      const dist = H_MAX * k / N_LEVELS;
      const path = [];
      for(let i=0; i<N_SAMPLES; i++){
        const x = X_START + (X_END - X_START) * i / (N_SAMPLES - 1);
        let y = isUpper ? dist : -dist;

        const r = ramp(x);
        if(r > 0){
          let minDist = Infinity;
          let closestIdx = 0;
          for(let j=0; j<contour.length; j++){
            const dx = x - contour[j][0];
            const dy = y - contour[j][1];
            const d = Math.sqrt(dx*dx + dy*dy);
            if(d < minDist){ minDist = d; closestIdx = j; }
          }
          const targetY = contour[closestIdx][1] + (isUpper ? dist : -dist) * (isUpper ? 1.032 : 1.028);
          y = lF(y, targetY, r);
        }

        if(x < -1.6 || x > 2.6) y = isUpper ? dist : -dist;
        if(!isUpper) y = Math.max(y, -H_MAX * 0.96);
        if(isUpper)  y = Math.min(y,  H_MAX * 1.02);

        path.push([x, y]);
      }
      lines.push({path, d: Math.abs(dist)});
    }
  }
  return lines;
}

function lerpStreamlines(lA,lB,t){
  const out=[];
  const n=Math.min(lA.length,lB.length);
  for(let i=0;i<n;i++){
    const pa=lA[i].path, pb=lB[i].path;
    const m=Math.min(pa.length,pb.length);
    const path=[];
    for(let j=0;j<m;j++) path.push([lF(pa[j][0],pb[j][0],t),lF(pa[j][1],pb[j][1],t)]);
    out.push({path,d:lF(lA[i].d,lB[i].d,t)});
  }
  return out;
}

const slCache={};
function cachedStreamlines(idx){
  if(!slCache[idx]) slCache[idx]=buildStreamlinesForAirfoil(PD[idx].d);
  return slCache[idx];
}

// ─── Animation control & drawing (unchanged) ─────────────────────────────────
const HOLD_DUR = 140;
const MORPH_DUR = 100;
let phase = 'hold';
let fromI = 0;
let toI = 1;
let morphT = 0;
let holdF = 0;

function tick(){
  if(phase === 'hold'){
    holdF++;
    if(holdF >= HOLD_DUR){
      phase = 'morph';
      holdF = 0;
      morphT = 0;
    }
  } else {
    morphT += 1 / MORPH_DUR;
    if(morphT >= 1){
      fromI = toI;
      toI = (toI + 1) % PROFILES.length;
      phase = 'hold';
      holdF = 0;
      morphT = 0;
    }
  }
}

function getAirfoil(){
  if(phase === 'hold') return PD[fromI].d;
  return lerpAirfoil(PD[fromI].d, PD[toI].d, ease(morphT));
}

function getLines(){
  if(phase === 'hold') return cachedStreamlines(fromI);
  return lerpStreamlines(cachedStreamlines(fromI), cachedStreamlines(toI), ease(morphT));
}

const PARTS = Array.from({length:50}, () => ({
  x: Math.random(), y: Math.random(),
  vx: 0.00018 + Math.random()*0.0004,
  r: 0.4 + Math.random()*1.2,
  al: 0.03 + Math.random()*0.07
}));

function drawParticles(W, H){
  ctx.save();
  for(const p of PARTS){
    p.x += p.vx;
    if(p.x > 1) p.x = 0;
    ctx.globalAlpha = p.al;
    ctx.fillStyle = '#e74c3c';
    ctx.beginPath();
    ctx.arc(p.x*W, p.y*H, p.r, 0, Math.PI*2);
    ctx.fill();
  }
  ctx.restore();
}

function toCV(nx, ny, cx, cy, ch){ return [cx + (nx-0.5)*ch, cy - ny*ch]; }

function drawDotGrid(W, H){
  ctx.save();
  ctx.fillStyle = 'rgba(255,255,255,0.022)';
  const g = 40;
  for(let x = g/2; x < W; x += g)
    for(let y = g/2; y < H; y += g){
      ctx.beginPath();
      ctx.arc(x, y, 0.8, 0, Math.PI*2);
      ctx.fill();
    }
  ctx.restore();
}

function drawStreamlines(lines, cx, cy, ch){
  ctx.save();
  ctx.lineWidth = 0.95;
  for(const {path, d} of lines){
    if(!path || path.length < 3) continue;
    const alpha = 0.16 + Math.min(0.42, d * 3.5);
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = 'rgba(130, 210, 255, 0.96)';
    ctx.beginPath();
    let [sx, sy] = toCV(path[0][0], path[0][1], cx, cy, ch);
    ctx.moveTo(sx, sy);
    for(let i=1; i<path.length; i++){
      let [px, py] = toCV(path[i][0], path[i][1], cx, cy, ch);
      ctx.lineTo(px, py);
    }
    ctx.stroke();
  }
  ctx.restore();
}

function drawAirfoil(af, cx, cy, ch){
  const upper=af.upper, lower=af.lower;
  function buildPath(){
    ctx.beginPath();
    const [fx,fy] = toCV(upper[0][0], upper[0][1], cx, cy, ch);
    ctx.moveTo(fx, fy);
    for(let i=1; i<upper.length; i++){
      const [px,py] = toCV(upper[i][0], upper[i][1], cx, cy, ch);
      ctx.lineTo(px, py);
    }
    for(let i=lower.length-1; i>=0; i--){
      const [px,py] = toCV(lower[i][0], lower[i][1], cx, cy, ch);
      ctx.lineTo(px, py);
    }
    ctx.closePath();
  }

  const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, ch*0.55);
  g.addColorStop(0, 'rgba(231,76,60,0.10)');
  g.addColorStop(1, 'transparent');
  buildPath(); ctx.fillStyle = g; ctx.fill();

  ctx.save();
  ctx.shadowColor = '#e74c3c';
  ctx.shadowBlur = 34;
  ctx.strokeStyle = 'rgba(231,76,60,0.62)';
  ctx.lineWidth = 3.3;
  buildPath(); ctx.stroke();
  ctx.restore();

  ctx.save();
  ctx.strokeStyle = 'rgba(255,190,180,0.94)';
  ctx.lineWidth = 1.35;
  buildPath(); ctx.stroke();
  ctx.restore();
}

function updateLabel(){
  if(phase === 'morph'){
    const fadeOut = Math.min(1, morphT * 2.8);
    const fadeIn  = Math.max(0, (morphT - 0.35) * 2.8);
    LBL.style.opacity = (1 - fadeOut + fadeIn).toString();
    if(morphT > 0.35) LBL.textContent = PD[toI].n;
  } else {
    LBL.style.opacity = '1';
    LBL.textContent = PD[fromI].n;
  }
}

// ─── Main frame ──────────────────────────────────────────────────────────────
function frame(){
  const W = CV.width, H = CV.height;
  const cx = W * 0.50;
  const cy = H * 0.54;
  const ch = W * 0.41;

  tick();
  const af = getAirfoil();
  const lines = getLines();
  updateLabel();

  ctx.clearRect(0, 0, W, H);
  drawDotGrid(W, H);
  drawParticles(W, H);
  drawStreamlines(lines, cx, cy, ch);
  drawAirfoil(af, cx, cy, ch);

  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
</script>
</body>
</html>
""", height=432)

# ── Section divider ───────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin:2px 0 16px;">
  <span style="font-size:0.66rem;text-transform:uppercase;letter-spacing:0.14em;
               color:#2a3650;font-weight:600;">Select a Module</span>
  <div style="width:38px;height:1px;margin:5px auto 0;
              background:linear-gradient(90deg,transparent,rgba(231,76,60,0.42),transparent);"></div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# NAV CARDS
# Bottom padding increased to 40px so the last row's hover lift + shadow
# is never clipped by the iframe boundary.
# ══════════════════════════════════════════════════════════════════════════════
components.html("""
<!DOCTYPE html>
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Inter:wght@400;500&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{
  background:transparent;
  padding:18px 12px 40px 12px;
  overflow:visible;
}
.grid{
  display:grid;
  grid-template-columns:repeat(3,1fr);
  gap:14px;
  overflow:visible;
}
.card{
  position:relative;
  background:linear-gradient(148deg,#0d1525 0%,#090e1c 100%);
  border:1px solid rgba(38,52,88,0.78);
  border-radius:14px;
  padding:22px 20px 20px;
  cursor:pointer;
  overflow:visible;
  min-height:152px;
  display:flex;flex-direction:column;gap:10px;
  transition:transform .28s cubic-bezier(.34,1.38,.64,1),border-color .25s ease,box-shadow .28s ease;
  isolation:isolate;
}
.card::before{
  content:'';position:absolute;top:0;left:18%;right:18%;height:1px;
  background:linear-gradient(90deg,transparent,rgba(231,76,60,0),transparent);
  transition:all .28s ease;border-radius:1px;
}
.card::after{
  content:'';position:absolute;top:-55px;right:-55px;
  width:140px;height:140px;border-radius:50%;
  background:radial-gradient(circle,rgba(231,76,60,0) 0%,transparent 70%);
  transition:all .32s ease;pointer-events:none;
}
.card:hover{
  transform:translateY(-7px) scale(1.018);
  border-color:rgba(231,76,60,.52);
  box-shadow:0 0 0 1px rgba(231,76,60,.15),0 14px 40px rgba(0,0,0,.58),
             0 0 55px rgba(231,76,60,.10),inset 0 0 28px rgba(231,76,60,.032);
}
.card:hover::before{
  background:linear-gradient(90deg,transparent,rgba(231,76,60,.92),transparent);
  left:3%;right:3%;
}
.card:hover::after{background:radial-gradient(circle,rgba(231,76,60,.13) 0%,transparent 70%);}
.card:active{transform:translateY(-3px) scale(1.008);}
.icon{
  font-size:2.05rem;line-height:1;
  transition:transform .28s cubic-bezier(.34,1.56,.64,1),filter .28s ease;display:block;
}
.card:hover .icon{transform:scale(1.14) translateY(-2px);filter:drop-shadow(0 0 10px rgba(231,76,60,.48));}
.title{
  font-family:'Rajdhani',sans-serif;font-size:1.28rem;font-weight:700;
  letter-spacing:.045em;color:#dde2f0;margin-top:2px;transition:color .22s;
}
.card:hover .title{color:#ff7060;}
.desc{
  font-family:'Inter',sans-serif;font-size:0.79rem;color:#46577a;
  line-height:1.52;font-weight:400;transition:color .22s;
}
.card:hover .desc{color:#6a7e9e;}
.arrow{
  position:absolute;bottom:13px;right:14px;
  font-family:'Inter',sans-serif;font-size:.67rem;
  color:rgba(231,76,60,0);letter-spacing:.09em;font-weight:500;transition:all .22s ease;
}
.card:hover .arrow{color:rgba(231,76,60,.65);right:12px;}
.tri{
  position:absolute;top:0;right:0;width:0;height:0;
  border-top:32px solid rgba(231,76,60,0);border-left:32px solid transparent;
  transition:border-top-color .25s;border-radius:0 14px 0 0;
}
.card:hover .tri{border-top-color:rgba(231,76,60,.22);}
</style>
</head>
<body>
<div class="grid" id="g"></div>
<script>
const CARDS=[
  {icon:"🔍",title:"Reference Library",  desc:"Browse & download from 1,638+ UIUC airfoils",  href:"/Reference_Library"},
  {icon:"⚗️",title:"Design Studio",      desc:"Generate NACA 4/5/6-series, modify & blend",   href:"/Design_Studio"},
  {icon:"📊",title:"Analysis Lab",       desc:"Multi-airfoil, multi-Re batch + Plotly charts", href:"/Analysis_Lab"},
  {icon:"🎯",title:"Optimizer",          desc:"Maximize L/D ratio with geometric constraints", href:"/Optimizer"},
  {icon:"🛫",title:"Wing Designer",      desc:"Lifting-line full-wing performance estimates",  href:"/Wing_Designer"},
  {icon:"📐",title:"Reynolds Calculator",desc:"Compute Re with live flow regime gauge",        href:"/Reynolds_Calculator"},
  {icon:"📚",title:"Tutorials",          desc:"Step-by-step guides and workflow tips",         href:"/Tutorials"},
  {icon:"ℹ️",title:"About",              desc:"Credits, limitations, and open-source info",    href:"/About"},
];
const g=document.getElementById('g');
CARDS.forEach(c=>{
  const d=document.createElement('div');
  d.className='card';
  d.innerHTML=`
    <span class="icon">${c.icon}</span>
    <div><div class="title">${c.title}</div><div class="desc">${c.desc}</div></div>
    <div class="arrow">OPEN →</div>
    <div class="tri"></div>`;
  d.onclick=()=>{window.parent.location.href=c.href;};
  g.appendChild(d);
});
</script>
</body>
</html>
""", height=500, scrolling=False)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#1a2535;font-size:0.74rem;padding:10px 0 4px;">
  <a href="https://m-selig.ae.illinois.edu/ads/coord_database.html"
     style="color:#6a2020;text-decoration:none;opacity:0.6;">UIUC Database</a>
  &nbsp;·&nbsp;
  <a href="https://github.com/peterdsharpe/NeuralFoil"
     style="color:#6a2020;text-decoration:none;opacity:0.6;">NeuralFoil</a>
  &nbsp;·&nbsp;
  <a href="https://github.com/peterdsharpe/AeroSandbox"
     style="color:#6a2020;text-decoration:none;opacity:0.6;">AeroSandbox</a>
</div>
""", unsafe_allow_html=True)
