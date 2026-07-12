// PHYSARUM — a living interface.
// Raw WebGPU, no dependencies. Three passes per step:
//   agents (sense/turn/move/deposit) -> diffuse+decay -> render.
// The DOM labels are rasterized into a "pheromone" texture the agents can smell.

import { AudioEngine } from './audio.js';

const AGENTS = 150_000;
const SUBSTEPS = 2;
const SIM_MAX_W = 1400;

const WGSL_COMMON = /* wgsl */`
struct P {
  size: vec2f, canvas: vec2f,
  mouse: vec2f, dt: f32, time: f32,
  mouseStr: f32, agentN: f32, sensAng: f32, sensDist: f32,
  turnSpd: f32, speed: f32, attractW: f32, decay: f32,
  depositAmt: f32, hue: f32, pad1: f32, pad2: f32,
};
`;

const WGSL_AGENTS = WGSL_COMMON + /* wgsl */`
@group(0) @binding(0) var<uniform> u: P;
@group(0) @binding(1) var<storage, read_write> agents: array<vec4f>;
@group(0) @binding(2) var<storage, read> trailIn: array<f32>;
@group(0) @binding(3) var<storage, read_write> deposit: array<atomic<u32>>;
@group(0) @binding(4) var attract: texture_2d<f32>;

fn wrapc(v: vec2f) -> vec2f { return (v % u.size + u.size) % u.size; }

fn cellIdx(v: vec2f) -> u32 {
  let c = min(vec2u(wrapc(v)), vec2u(u.size) - 1u);
  return c.y * u32(u.size.x) + c.x;
}

fn pcg(v: u32) -> u32 {
  var s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn rand01(s: u32) -> f32 { return f32(pcg(s)) / 4294967295.0; }

fn sense(pos: vec2f, ang: f32) -> f32 {
  let sp = wrapc(pos + vec2f(cos(ang), sin(ang)) * u.sensDist);
  let a = textureLoad(attract, vec2i(min(sp, u.size - 1.0)), 0).r;
  let md = sp - u.mouse;
  let m = u.mouseStr * exp(-dot(md, md) / 7200.0);
  return trailIn[cellIdx(sp)] + u.attractW * a + m;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= u32(u.agentN)) { return; }
  var a = agents[i];
  var pos = a.xy;
  var ang = a.z;

  let c = sense(pos, ang);
  let l = sense(pos, ang - u.sensAng);
  let r = sense(pos, ang + u.sensAng);
  let rnd = rand01(i * 9781u + u32(u.time * 1000.0));

  if (c > l && c > r) {
    // hold course
  } else if (l > r) {
    ang -= u.turnSpd * u.dt;
  } else if (r > l) {
    ang += u.turnSpd * u.dt;
  } else {
    ang += (rnd - 0.5) * 2.0 * u.turnSpd * u.dt;
  }

  pos = wrapc(pos + vec2f(cos(ang), sin(ang)) * u.speed * u.dt);
  agents[i] = vec4f(pos, ang, a.w);
  atomicAdd(&deposit[cellIdx(pos)], u32(u.depositAmt * 256.0));
}
`;

const WGSL_DIFFUSE = WGSL_COMMON + /* wgsl */`
@group(0) @binding(0) var<uniform> u: P;
@group(0) @binding(1) var<storage, read> trailIn: array<f32>;
@group(0) @binding(2) var<storage, read_write> trailOut: array<f32>;
@group(0) @binding(3) var<storage, read_write> deposit: array<atomic<u32>>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let W = u32(u.size.x);
  let H = u32(u.size.y);
  if (gid.x >= W || gid.y >= H) { return; }
  var sum = 0.0;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let x = (i32(gid.x) + dx + i32(W)) % i32(W);
      let y = (i32(gid.y) + dy + i32(H)) % i32(H);
      sum += trailIn[u32(y) * W + u32(x)];
    }
  }
  let i = gid.y * W + gid.x;
  let d = f32(atomicExchange(&deposit[i], 0u)) / 256.0;
  trailOut[i] = clamp((sum / 9.0 + d) * u.decay, 0.0, 6.0);
}
`;

const WGSL_RENDER = WGSL_COMMON + /* wgsl */`
@group(0) @binding(0) var<uniform> u: P;
@group(0) @binding(1) var<storage, read> trail: array<f32>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  var p = array(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(p[vi], 0.0, 1.0);
}

// rotate color about the grey axis (Rodrigues): cheap full-spectrum hue shift
fn hueRotate(c: vec3f, a: f32) -> vec3f {
  let k = vec3f(0.57735026);
  return c * cos(a) + cross(k, c) * sin(a) + k * dot(k, c) * (1.0 - cos(a));
}

fn at(x: i32, y: i32) -> f32 {
  let W = i32(u.size.x);
  let H = i32(u.size.y);
  let cx = (x + W) % W;
  let cy = (y + H) % H;
  return trail[u32(cy) * u32(W) + u32(cx)];
}

@fragment
fn fs(@builtin(position) fc: vec4f) -> @location(0) vec4f {
  let sp = fc.xy / u.canvas * u.size - 0.5;
  let b = floor(sp);
  let f = fract(sp);
  let x = i32(b.x); let y = i32(b.y);
  let t = mix(
    mix(at(x, y), at(x + 1, y), f.x),
    mix(at(x, y + 1), at(x + 1, y + 1), f.x),
    f.y);

  var col = vec3f(0.012, 0.024, 0.019);
  col = mix(col, vec3f(0.04, 0.28, 0.21), smoothstep(0.0, 0.7, t));
  col = mix(col, vec3f(0.22, 0.82, 0.52), smoothstep(0.55, 1.9, t));
  col = mix(col, vec3f(0.88, 1.0, 0.86), smoothstep(1.7, 4.2, t));
  col = clamp(hueRotate(col, u.hue), vec3f(0.0), vec3f(1.0));

  let v = fc.xy / u.canvas - 0.5;
  col *= 1.0 - dot(v, v) * 0.6;
  return vec4f(col, 1.0);
}
`;

const canvas = document.getElementById('gpu');
const audio = new AudioEngine();
let device, context, format;
let agentPipe, diffusePipe, renderPipe;
let paramsBuf, sim = null; // sim: size-dependent resources
let hovered = null;
const mouse = { x: -9999, y: -9999, speed: 0, lastX: 0, lastY: 0, lastT: 0 };
let startTime = performance.now();

// interaction state: dropped food, photophobia, color
const food = []; // {x, y, s} in CSS px, s = strength 0..1
let lightOn = false;
let hueScroll = 0; // user scroll offset, radians
let huePulse = 0;  // decaying action pulse: warm on feed, cold on light

const params = new Float32Array(20);

function fail() {
  document.getElementById('nogpu').style.display = 'flex';
  document.getElementById('splash').classList.add('gone');
}

async function setup() {
  if (!navigator.gpu) return fail();
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return fail();
  device = await adapter.requestDevice();
  context = canvas.getContext('webgpu');
  format = navigator.gpu.getPreferredCanvasFormat();

  paramsBuf = device.createBuffer({ size: params.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  agentPipe = device.createComputePipeline({
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: WGSL_AGENTS }), entryPoint: 'main' },
  });
  diffusePipe = device.createComputePipeline({
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: WGSL_DIFFUSE }), entryPoint: 'main' },
  });
  const renderModule = device.createShaderModule({ code: WGSL_RENDER });
  renderPipe = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: renderModule, entryPoint: 'vs' },
    fragment: { module: renderModule, entryPoint: 'fs', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });

  initSim();
  requestAnimationFrame(frame);
}

function initSim() {
  const dpr = Math.min(devicePixelRatio || 1, 2);
  canvas.width = Math.round(innerWidth * dpr);
  canvas.height = Math.round(innerHeight * dpr);
  context.configure({ device, format, alphaMode: 'opaque' });

  const scale = Math.min(1, SIM_MAX_W / innerWidth);
  const W = Math.max(64, Math.round(innerWidth * scale));
  const H = Math.max(64, Math.round(innerHeight * scale));

  if (sim) for (const b of [sim.trailA, sim.trailB, sim.deposit, sim.agents]) b.destroy();
  if (sim) sim.attractTex.destroy();

  const cells = W * H;
  const trailA = device.createBuffer({ size: cells * 4, usage: GPUBufferUsage.STORAGE });
  const trailB = device.createBuffer({ size: cells * 4, usage: GPUBufferUsage.STORAGE });
  const deposit = device.createBuffer({ size: cells * 4, usage: GPUBufferUsage.STORAGE });

  // spawn scattered: the network condenses out of noise
  const data = new Float32Array(AGENTS * 4);
  for (let i = 0; i < AGENTS; i++) {
    data[i * 4] = Math.random() * W;
    data[i * 4 + 1] = Math.random() * H;
    data[i * 4 + 2] = Math.random() * Math.PI * 2;
    data[i * 4 + 3] = Math.random();
  }
  const agents = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(agents, 0, data);

  const attractTex = device.createTexture({
    size: [W, H], format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  const attractCanvas = Object.assign(document.createElement('canvas'), { width: W, height: H });

  const agentBG = [trailA, trailB].map(tin => device.createBindGroup({
    layout: agentPipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: agents } },
      { binding: 2, resource: { buffer: tin } },
      { binding: 3, resource: { buffer: deposit } },
      { binding: 4, resource: attractTex.createView() },
    ],
  }));
  const diffuseBG = [[trailA, trailB], [trailB, trailA]].map(([tin, tout]) => device.createBindGroup({
    layout: diffusePipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: tin } },
      { binding: 2, resource: { buffer: tout } },
      { binding: 3, resource: { buffer: deposit } },
    ],
  }));
  const renderBG = [trailA, trailB].map(t => device.createBindGroup({
    layout: renderPipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: t } },
    ],
  }));

  sim = { W, H, trailA, trailB, deposit, agents, attractTex, attractCanvas, agentBG, diffuseBG, renderBG, flip: 0 };
  drawAttractants();
}

// Rasterize the DOM labels into the pheromone field the agents can smell.
function drawAttractants() {
  const { attractCanvas, attractTex, W, H } = sim;
  const ctx = attractCanvas.getContext('2d');
  const sx = W / innerWidth, sy = H / innerHeight;
  ctx.filter = 'none';
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, W, H);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  for (const el of document.querySelectorAll('.food')) {
    const r = el.getBoundingClientRect();
    const cx = (r.left + r.width / 2) * sx;
    const cy = (r.top + r.height / 2) * sy;
    const isHover = el === hovered;

    // wide, weak radial basin so agents smell the label from far away
    // (the title's glyphs are basin enough; a disc there saturates the center)
    if (el.id !== 'title') {
      const R = (isHover ? 300 : 190) * sx;
      const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, R);
      grad.addColorStop(0, `rgba(255,255,255,${isHover ? 0.3 : 0.09})`);
      grad.addColorStop(1, 'rgba(255,255,255,0)');
      ctx.fillStyle = grad;
      ctx.fillRect(cx - R, cy - R, R * 2, R * 2);
    }

    // sharp, strong glyphs on top so veins trace the letterforms
    const fontPx = Math.max(8, parseFloat(getComputedStyle(el).fontSize) * sx * (isHover ? 1.4 : 1));
    ctx.font = `${fontPx}px monospace`;
    ctx.filter = `blur(${isHover ? 2 : 4}px)`;
    ctx.fillStyle = isHover ? 'rgba(255,255,255,1)' : 'rgba(255,255,255,0.6)';
    ctx.fillText(el.dataset.label || el.textContent.trim(), cx, cy);
    ctx.filter = 'none';
  }

  // dropped food: bright blobs that fade as the colony consumes them
  for (const f of food) {
    const fx = f.x * sx, fy = f.y * sy;
    const R = 130 * sx * (0.5 + f.s * 0.5);
    const g = ctx.createRadialGradient(fx, fy, 0, fx, fy, R);
    g.addColorStop(0, `rgba(255,255,255,${0.75 * f.s})`);
    g.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = g;
    ctx.fillRect(fx - R, fy - R, R * 2, R * 2);
  }

  device.queue.copyExternalImageToTexture({ source: attractCanvas }, { texture: attractTex }, [W, H]);
}

function writeParams(time) {
  const { W, H } = sim;
  const sx = W / innerWidth, sy = H / innerHeight;
  // hue: slow autonomous drift + gentle breathing + user scroll + action pulses
  const hue = time * 0.03 + Math.sin(time * 0.1) * 0.15 + hueScroll + huePulse;
  params.set([
    W, H, canvas.width, canvas.height,
    mouse.x * sx, mouse.y * sy, 1.0, time,
    lightOn ? -7.0 : 1.2, AGENTS, 0.45, 12.0,
    0.5, 1.7, 2.2, 0.91,
    0.35, hue, 0, 0,
  ]);
  device.queue.writeBuffer(paramsBuf, 0, params);
}

function frame(now) {
  const time = (now - startTime) / 1000;
  writeParams(time);

  const enc = device.createCommandEncoder();
  for (let s = 0; s < SUBSTEPS; s++) {
    const f = sim.flip;
    let pass = enc.beginComputePass();
    pass.setPipeline(agentPipe);
    pass.setBindGroup(0, sim.agentBG[f]);
    pass.dispatchWorkgroups(Math.ceil(AGENTS / 256));
    pass.setPipeline(diffusePipe);
    pass.setBindGroup(0, sim.diffuseBG[f]);
    pass.dispatchWorkgroups(Math.ceil(sim.W / 8), Math.ceil(sim.H / 8));
    pass.end();
    sim.flip = 1 - f;
  }
  const rp = enc.beginRenderPass({
    colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: 'clear', clearValue: [0, 0, 0, 1], storeOp: 'store' }],
  });
  rp.setPipeline(renderPipe);
  rp.setBindGroup(0, sim.renderBG[sim.flip]);
  rp.draw(3);
  rp.end();
  device.queue.submit([enc.finish()]);

  // decay mouse speed and feed it to the audio filter
  mouse.speed *= 0.92;
  audio.setActivity(Math.min(1, mouse.speed / 40));
  huePulse *= 0.97;

  requestAnimationFrame(frame);
}

// --- interaction ---

addEventListener('pointermove', e => {
  const dt = Math.max(1, e.timeStamp - mouse.lastT);
  mouse.speed = Math.min(80, Math.hypot(e.clientX - mouse.lastX, e.clientY - mouse.lastY) / dt * 16 + mouse.speed * 0.5);
  mouse.lastX = e.clientX; mouse.lastY = e.clientY; mouse.lastT = e.timeStamp;
  mouse.x = e.clientX; mouse.y = e.clientY;
});
addEventListener('pointerleave', () => { mouse.x = -9999; mouse.y = -9999; });

// press-and-release on empty canvas = drop food; press-and-hold = shine light
let holdTimer = null;
addEventListener('pointerdown', e => {
  if (e.target !== canvas) return;
  holdTimer = setTimeout(() => {
    holdTimer = null;
    lightOn = true;
    huePulse -= 0.7; // cold flash
    audio.light(true);
  }, 250);
});
addEventListener('pointerup', e => {
  if (lightOn) {
    lightOn = false;
    audio.light(false);
    return;
  }
  if (!holdTimer) return;
  clearTimeout(holdTimer);
  holdTimer = null;
  food.push({ x: e.clientX, y: e.clientY, s: 1 });
  if (food.length > 12) food.shift();
  huePulse += 0.7; // warm flash
  audio.feed();
  drawAttractants();
});

// dropped food fades as the colony consumes it
setInterval(() => {
  if (!food.length || !sim) return;
  for (const f of food) f.s *= 0.96;
  for (let i = food.length - 1; i >= 0; i--) if (food[i].s < 0.08) food.splice(i, 1);
  drawAttractants();
}, 400);

// scroll shifts the colony's color
addEventListener('wheel', e => { hueScroll += e.deltaY * 0.0015; });

for (const el of document.querySelectorAll('.food')) {
  el.addEventListener('pointerenter', () => { hovered = el; drawAttractants(); audio.hover(); });
  el.addEventListener('pointerleave', () => { if (hovered === el) hovered = null; drawAttractants(); });
}

for (const btn of document.querySelectorAll('.nav')) {
  btn.addEventListener('click', () => {
    const panel = document.getElementById(btn.dataset.panel);
    const wasOpen = panel.classList.contains('open');
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('open'));
    document.querySelectorAll('.nav').forEach(n => n.classList.remove('open'));
    if (!wasOpen) { panel.classList.add('open'); btn.classList.add('open'); audio.chord(); }
  });
}
document.querySelectorAll('.panel .x').forEach(x => x.addEventListener('click', () => {
  x.closest('.panel').classList.remove('open');
  document.querySelectorAll('.nav').forEach(n => n.classList.remove('open'));
}));

let resizeTimer;
addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { if (device) initSim(); }, 250);
});

const splash = document.getElementById('splash');
const muteBtn = document.getElementById('mute');
function wake() {
  if (splash.classList.contains('gone')) return;
  splash.classList.add('gone');
  audio.start();
  muteBtn.hidden = false;
}
splash.addEventListener('click', wake);
addEventListener('keydown', wake, { once: true });

muteBtn.addEventListener('click', () => {
  const on = audio.toggle();
  muteBtn.textContent = on ? 'SOUND ON' : 'SOUND OFF';
});

document.fonts?.ready.then(() => sim && drawAttractants());
setup();
