/* Social World Model — Lean V2 replay engine.
   Renders ONLY what a real lean_v2 run produced (recording.json). Deterministic step-through:
   the whole world state at step i is recomputed from events[0..i], so play/pause/step/scrub
   are all just "render at index i". */
'use strict';

const S = {
  rec: null, events: [], idx: -1, playing: false, timer: null, speed: 1000,
  activeTab: 'log', selectedCall: null, selectedActor: null, palette: {},
};
const $ = (s) => document.querySelector(s);
const el = (t, c, h) => { const e = document.createElement(t); if (c) e.className = c; if (h != null) e.innerHTML = h; return e; };
const esc = (s) => String(s == null ? '' : s).replace(/[&<>]/g, (m) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[m]));
const pct = (p) => (p == null ? '—' : (p * 100).toFixed(0) + '%');
const HUES = [205, 168, 32, 280, 345, 128, 55, 240];

/* ---------------- load ---------------- */
async function boot() {
  bindTransport();
  let list = [];
  try { list = (await (await fetch('/recordings/index.json')).json()).recordings || []; } catch (e) { }
  const sel = $('#recording-select');
  sel.innerHTML = '';
  if (!list.length) { sel.appendChild(el('option', null, 'no recordings yet')); }
  list.forEach((r) => {
    const o = el('option', null, `${esc(r.title || r.slug)}  ·  ${r.n_llm_calls || 0} calls`);
    o.value = r.file; sel.appendChild(o);
  });
  sel.onchange = () => loadRecording(sel.value);
  const params = new URLSearchParams(location.search);
  const want = params.get('rec');
  const pick = (want && list.find((r) => r.slug === want || r.file === want)) || list[0];
  if (pick) { sel.value = pick.file; await loadRecording(pick.file); }
  // deep-link to a playback position / panel (used for sharing/replay + headless capture)
  const step = params.get('step');
  if (step != null && step !== '') renderAt(parseInt(step, 10));
  if (params.get('actor')) showActor(params.get('actor'));
  if (params.get('call') != null && params.get('call') !== '') showCall(parseInt(params.get('call'), 10), true);
  if (params.get('tab')) setTab(params.get('tab'));
  if (params.get('play') === '1') play();
}

async function loadRecording(file) {
  stop();
  const rec = await (await fetch('/recordings/' + file)).json();
  S.rec = rec; S.events = rec.events || []; S.idx = -1;
  S.selectedCall = null; S.selectedActor = null;
  // stable per-actor color palette
  S.palette = {};
  (rec.cast || []).forEach((a, i) => { S.palette[a.id] = `hsl(${HUES[i % HUES.length]} 70% 62%)`; });
  buildTopbar(); buildPhaseRail(); buildLog();
  $('#scrub').max = String(Math.max(0, S.events.length - 1));
  renderAt(-1);
}

/* ---------------- static UI ---------------- */
function buildTopbar() {
  const m = S.rec.meta || {};
  $('#q-line').textContent = m.question || '';
  $('#q-line').title = m.question || '';
  $('#asof-line').innerHTML = `as of <b>${esc(m.as_of || '?')}</b> → <b>${esc(m.horizon || '?')}</b><br>${esc(m.model || '')} · ${esc(m.n_llm_calls || 0)} LLM calls`;
}
function phaseList() {
  const seen = []; (S.events || []).forEach((e) => { if (e.type === 'phase' && !seen.find((p) => p.phase === e.phase)) seen.push(e); });
  return seen;
}
function buildPhaseRail() {
  const rail = $('#phase-rail'); rail.innerHTML = '';
  phaseList().forEach((p, i) => {
    const pill = el('div', 'phase-pill', `<span class="pn">${i + 1}</span><span>${esc(p.title || p.phase)}</span>`);
    pill.dataset.phase = p.phase; pill.onclick = () => jumpToPhase(p.phase);
    rail.appendChild(pill);
  });
}

/* ---------------- state derivation ---------------- */
function computeState(upto) {
  const st = {
    phase: null, phasesDone: new Set(), introduced: false, actorStates: {},
    votes: {}, tally: {}, prob: null, status: null, active: null, activeCall: null,
    bubble: null, conditions: [], preflight: null, terminal: null,
  };
  for (let i = 0; i <= upto && i < S.events.length; i++) {
    const e = S.events[i];
    switch (e.type) {
      case 'phase':
        if (st.phase) st.phasesDone.add(st.phase); st.phase = e.phase; break;
      case 'world_ready': st.introduced = true; break;
      case 'condition': st.conditions.push(e); break;
      case 'preflight': st.preflight = e; break;
      case 'actor_states': st.actorStates[e.actor_id] = e.hypotheses; break;
      case 'vote':
        st.votes[e.actor_id] = e.vote_option; st.tally = e.tally || st.tally; break;
      case 'terminal': st.terminal = e; break;
      case 'forecast': if (e.headline_probability != null) st.prob = e.headline_probability; break;
      case 'result': st.status = e.status; if (e.headline != null) st.prob = e.headline; break;
    }
    if (i === upto) {
      st.active = e;
      if (e.type === 'llm_call') st.activeCall = e.seq;
      if ((e.type === 'llm_call' && e.decision) || e.type === 'vote' || e.type === 'decision_reused') {
        st.bubble = { actor: e.actor_id, text: bubbleText(e) };
      }
    }
  }
  return st;
}
function bubbleText(e) {
  if (e.type === 'vote') return `votes <span class="b-act">${esc(e.vote_option)}</span>`;
  const p = e.parsed || {};
  if (e.sub_type === 'deliberation') return `reflects again — ${esc((p.summary || '').slice(0, 90) || 'reconsiders')}`;
  const act = e.chosen || p.chosen_action || p.act_or_wait || 'decides';
  const v = e.vote_option || p.vote_option;
  return `<span class="b-act">${esc(act)}</span>${v ? ` · votes ${esc(v)}` : ''}`;
}

/* ---------------- render ---------------- */
function renderAt(i) {
  S.idx = Math.max(-1, Math.min(i, S.events.length - 1));
  const st = computeState(S.idx);
  renderHeadline(st); renderRail(st); renderPlane(st); renderNow(st);
  renderLogHighlight(); renderTransport(st);
  if (st.activeCall != null && S.activeTab !== 'actor') showCall(st.activeCall, false);
}

function renderHeadline(st) {
  $('#headline-p').textContent = st.prob == null ? '—' : pct(st.prob);
  const sp = $('#status-pill');
  sp.textContent = st.status || (st.phase ? st.phase : 'ready');
  sp.className = 'status-pill ' + (st.status || '');
}
function renderRail(st) {
  const order = phaseList().map((p) => p.phase);
  document.querySelectorAll('.phase-pill').forEach((pill) => {
    const ph = pill.dataset.phase; pill.classList.remove('active', 'done');
    if (ph === st.phase) pill.classList.add('active');
    else if (order.indexOf(ph) < order.indexOf(st.phase)) pill.classList.add('done');
  });
}

function renderPlane(st) {
  const plane = $('#plane'), nodesWrap = $('#nodes'), svg = $('#edges');
  $('#plane-empty').classList.toggle('hidden', st.introduced);
  const hub = $('#hub');
  if (!st.introduced) { nodesWrap.innerHTML = ''; svg.innerHTML = ''; hub.classList.add('hidden'); return; }
  const cast = (S.rec.cast || []);
  const W = plane.clientWidth, H = plane.clientHeight;
  const cx = W * 0.5, cy = H * 0.46;
  const Rx = Math.min(W * 0.34, W / 2 - 118);   // leave room for node width
  const Ry = Math.min(H * 0.32, H / 2 - 92);    // keep nodes off the transport bar
  const inst = S.rec.institution || {};

  // hub (centered on the ring)
  hub.classList.remove('hidden');
  hub.style.left = cx + 'px'; hub.style.top = cy + 'px';
  $('#hub-title').textContent = inst.name || 'institution';
  $('#hub-rule').textContent = (inst.decision_rule || '') + (inst.vote_options && inst.vote_options.length ? ' · ' + inst.vote_options.join('/') : '');
  const counts = st.tally || {};
  $('#hub-tally').innerHTML = Object.keys(counts).length
    ? Object.entries(counts).map(([k, v]) => `<span class="tally-chip">${esc(k)} <b>${v}</b></span>`).join('')
    : '<span class="tally-chip">no votes yet</span>';
  $('#hub-verdict').innerHTML = verdictHTML(st);

  // positions (ellipse that always fits the plane)
  const pos = {};
  cast.forEach((a, k) => { const ang = -Math.PI / 2 + (k / cast.length) * 2 * Math.PI; pos[a.id] = { x: cx + Rx * Math.cos(ang), y: cy + Ry * Math.sin(ang) }; });

  // edges (relationships)
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  let edges = '';
  (S.rec.relationships || []).forEach((r) => {
    const a = pos[r.source], b = pos[r.target]; if (!a || !b) return;
    const on = st.bubble && (r.source === st.bubble.actor || r.target === st.bubble.actor);
    edges += `<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" stroke="${on ? 'rgba(56,224,196,.55)' : 'rgba(120,140,180,.14)'}" stroke-width="${on ? 1.6 : 1}"/>`;
  });
  svg.innerHTML = edges;

  // nodes
  nodesWrap.innerHTML = '';
  cast.forEach((a) => {
    const p = pos[a.id]; const color = S.palette[a.id];
    const node = el('div', 'node' + (a.kept ? '' : ' pruned'));
    node.style.left = p.x + 'px'; node.style.top = p.y + 'px';
    const vote = st.votes[a.id];
    node.innerHTML =
      `<div class="accent" style="background:${color}"></div>` +
      `<div class="think"></div>` +
      `<div class="node-name">${esc(a.name)}</div>` +
      `<div class="node-role">${esc(a.role || '')}</div>` +
      `<div class="node-vote ${vote ? 'set' : ''}">${vote ? '🗳 ' + esc(vote) : (a.kept ? 'thinking…' : 'not decisive')}</div>`;
    if (st.bubble && st.bubble.actor === a.id) { node.classList.add('active'); node.appendChild(el('div', 'bubble', st.bubble.text)); }
    node.onclick = () => { S.selectedActor = a.id; setTab('actor'); showActor(a.id); };
    requestAnimationFrame(() => node.classList.add('show'));
    nodesWrap.appendChild(node);
  });
}
function verdictHTML(st) {
  if (!st.terminal && !st.status) return '';
  const counts = st.tally || {}; const keys = Object.keys(counts);
  const total = (S.rec.institution && S.rec.institution.members || []).length || (S.rec.cast || []).length;
  const voted = Object.values(counts).reduce((a, b) => a + b, 0);
  const rule = (S.rec.institution || {}).decision_rule || '';
  if (keys.length === 1 && voted >= total && total > 0) return `<span style="color:var(--good)">unanimous ${esc(keys[0])}</span>`;
  if (keys.length > 1) return `<span style="color:var(--warn)">split ${keys.map((k) => counts[k]).join('-')}</span>`;
  return st.status ? `<span style="color:var(--ink-dim)">${esc(st.status)}</span>` : '';
}

function renderNow(st) {
  const e = st.active; const d = e ? describe(e, st) : { ic: '·', html: 'Ready — press Play or Step ▶.' };
  $('#now-icon').textContent = d.ic; $('#now-text').innerHTML = d.html;
}
function describe(e, st) {
  const who = (id, nm) => `<span style="color:${S.palette[id] || 'var(--accent2)'}">${esc(nm || id)}</span>`;
  switch (e.type) {
    case 'phase': return { ic: '▶', html: `<b>Phase:</b> ${esc(e.title || e.phase)}` };
    case 'llm_call': {
      const meta = `<span style="color:var(--ink-faint)">(${e.prompt_chars}c → ${e.reply_chars}c · ${e.latency_s}s · ${esc(e.tier)})</span>`;
      if (e.sub_type === 'blueprint') return { ic: '🧩', html: `Compiling ONE coherent causal world — structured LLM call ${meta}` };
      if (e.sub_type === 'grounding') return { ic: '📊', html: `Grounding in counted reference classes ${meta}` };
      if (e.sub_type === 'state_generation') return { ic: '🧠', html: `Modelling each actor's genuinely different private realities ${meta}` };
      if (e.sub_type === 'deliberation') return { ic: '🤔', html: `${who(e.actor_id, e.actor_name)} deliberates once more ${meta}` };
      if (e.decision) return { ic: '🗣', html: `${who(e.actor_id, e.actor_name)} — bounded moment of cognition → <b>${esc(e.chosen || (e.parsed && e.parsed.chosen_action) || 'decides')}</b>${e.vote_option ? ' · votes ' + esc(e.vote_option) : ''} ${meta}` };
      return { ic: '💬', html: `${who(e.actor_id, e.actor_name)} — LLM call ${meta}` };
    }
    case 'world_ready': return { ic: '🌍', html: `World compiled: <b>${(e.cast || []).length}</b> actors · institution <b>${esc((e.institution || {}).name || '')}</b> (${esc((e.institution || {}).decision_rule || '')})` };
    case 'condition': return { ic: '📎', html: `Shared condition <b>${esc(e.id)}</b>: ${esc(e.claim)} — counted rate <b>${e.rate}</b> (n=${e.n})` };
    case 'outcome_class': return { ic: '📈', html: `Outcome reference class: rate <b>${e.rate}</b> (n=${e.n})` };
    case 'actor_states': return { ic: '👤', html: `${who(e.actor_id, e.actor_name)} modelled with <b>${(e.hypotheses || []).length}</b> private-state hypotheses` };
    case 'preflight': return { ic: e.verdict === 'answerable' ? '✅' : '⚠️', html: `Answerability preflight: <b>${esc(e.verdict)}</b>` };
    case 'vote': return { ic: '🗳', html: `${who(e.actor_id, e.actor_name)} votes <b>${esc(e.vote_option)}</b>` };
    case 'decision_reused': return { ic: '♻️', html: `${who(e.actor_id, e.actor_name)} — decision reused from an equivalent context (no new LLM call) → ${esc(e.chosen || '')}${e.vote_option ? ' · ' + esc(e.vote_option) : ''}` };
    case 'terminal': return { ic: '🏁', html: `Terminal evaluation — rule <b>${esc(e.decision_rule || '')}</b>` };
    case 'forecast': return { ic: '🧮', html: `Forecast decomposed — grounded prior ${fmt(e.grounded_prior)} · sim ${fmt(e.simulation_conditional)} → <b>${fmt2(e.combined)}</b> via ${esc(e.method || '')}` };
    case 'result': return { ic: '★', html: `Result: <b>${esc(e.status)}</b> — P = <b>${pct(e.headline)}</b> · ${esc(e.source || '')}` };
    default: return { ic: '·', html: esc(e.type) };
  }
}
const fmt = (o) => o && o.p != null ? (o.p) : (o == null ? '—' : o);
const fmt2 = (v) => v == null ? '—' : v;

/* ---------------- log ---------------- */
function buildLog() {
  const list = $('#log-list'); list.innerHTML = '';
  let lastPhase = null;
  S.events.forEach((e, i) => {
    if (e.type === 'phase') { const h = el('div', 'log-phase', esc(e.title || e.phase)); list.appendChild(h); lastPhase = e.phase; return; }
    const d = describe(e, {});
    const callable = e.type === 'llm_call';
    const row = el('div', 'log-row' + (callable ? ' callable' : ''));
    row.dataset.i = i; if (callable) row.dataset.seq = e.seq;
    const tier = callable ? `<span class="log-tier">${esc(e.tier || '')}</span>` : '';
    row.innerHTML = `<div class="log-ic">${d.ic}</div><div class="log-body"><div class="log-t">${d.html}</div>${callable ? `<div class="log-sub">${esc(e.stage)} · ${esc(e.sub_type)}</div>` : ''}</div>${tier}`;
    row.onclick = () => { renderAt(i); if (callable) { setTab('call'); showCall(e.seq, true); } };
    list.appendChild(row);
  });
  $('#log-search').oninput = filterLog;
  $('#only-calls').onchange = filterLog;
}
function filterLog() {
  const q = ($('#log-search').value || '').toLowerCase(); const only = $('#only-calls').checked;
  document.querySelectorAll('#log-list .log-row').forEach((r) => {
    const t = r.textContent.toLowerCase(); const isCall = r.classList.contains('callable');
    r.style.display = ((!only || isCall) && t.includes(q)) ? '' : 'none';
  });
}
function renderLogHighlight() {
  document.querySelectorAll('#log-list .log-row').forEach((r) => {
    const i = +r.dataset.i;
    r.classList.toggle('current', i === S.idx);
    r.classList.toggle('future', i > S.idx);
    if (i === S.idx) r.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  });
}

/* ---------------- call inspector ---------------- */
function showCall(seq, focus) {
  const c = (S.rec.llm_calls || []).find((x) => x.seq === seq); if (!c) return;
  S.selectedCall = seq;
  $('#call-empty').classList.add('hidden'); $('#call-view').classList.remove('hidden');
  const badge = $('#call-badge'); badge.textContent = c.sub_type; badge.className = 'call-badge ' + c.sub_type;
  $('#call-actor').textContent = c.actor_name || labelForStage(c.stage);
  $('#call-meta').textContent = `${c.prompt_chars}c → ${c.reply_chars}c · ${c.latency_s}s · ${c.tier}${c.retried ? ' · retried' : ''}`;
  $('#call-prompt').innerHTML = hlPrompt(c.prompt);
  $('#call-reply').innerHTML = hlJSON(c.reply);
  if (focus) setTab('call');
}
function labelForStage(s) { return ({ structural_generation: 'world compiler', structural_compile: 'world compiler', reference_class_grounding: 'grounding', state_generation: 'state modeler' })[s] || s; }
function hlPrompt(t) {
  return esc(t).replace(/(YOU ARE:[^\n]*)/, '<span class="youare">$1</span>');
}
function hlJSON(t) {
  let s = esc(t);
  s = s.replace(/(&quot;[\w_]+&quot;)(\s*:)/g, '<span class="k">$1</span>$2');
  s = s.replace(/:\s*(&quot;[^&]*&quot;)/g, ': <span class="s">$1</span>');
  s = s.replace(/:\s*(-?\d+\.?\d*)/g, ': <span class="n">$1</span>');
  return s;
}

/* ---------------- actor model ---------------- */
function showActor(aid) {
  const a = (S.rec.cast || []).find((x) => x.id === aid); if (!a) return;
  S.selectedActor = aid; setTab('actor');
  $('#actor-empty').classList.add('hidden');
  const view = $('#actor-view'); view.classList.remove('hidden');
  const stObj = (S.rec.actor_states || {})[aid] || { hypotheses: [] };
  const hyps = stObj.hypotheses || [];
  const color = S.palette[aid];
  let html = `<div class="actor-h"><span class="sw" style="background:${color}"></span><span class="an">${esc(a.name)}</span></div>` +
    `<div class="actor-role">${esc(a.role || '')} · authority: ${esc((a.authority || []).join(', ') || '—')}</div>`;
  if (stObj.unknown_mass != null) html += `<div class="wsens">unknown-state mass: ${(+stObj.unknown_mass).toFixed(3)} — genuinely unmodelled possibilities, never assigned 0.5</div>`;
  if (!hyps.length) html += `<div class="panel-empty" style="padding:14px 0">No private-state hypotheses were generated for this actor in this run.</div>`;
  hyps.forEach((h) => {
    const w = h.weight_mid;
    html += `<div class="hyp"><div class="hyp-top"><span class="hyp-id">${esc(h.state_id)}${h.reversal_capable ? ' ⤾' : ''}</span>` +
      `<span class="hyp-w">${w == null ? '' : 'weight ' + (w).toFixed(2)}</span></div>` +
      `<div class="hyp-claim">${esc(h.claim)}</div>` +
      `<div class="hyp-grid">` +
      `<div class="hyp-cell"><div class="lab">beliefs</div><ul>${(h.beliefs || []).map((b) => `<li>${esc(b)}</li>`).join('') || '<li>—</li>'}</ul></div>` +
      `<div class="hyp-cell"><div class="lab">goals</div><ul>${(h.goals || []).map((g) => `<li>${esc(g)}</li>`).join('') || '<li>—</li>'}</ul></div>` +
      `</div>` +
      (h.stances && h.stances.length ? `<div class="hyp-rel"><b>stances:</b> ${esc(h.stances.join(' · '))}</div>` : '') +
      (h.pressures ? `<div class="hyp-rel"><b>pressures:</b> ${esc(h.pressures)}</div>` : '') +
      (h.relationships && Object.keys(h.relationships).length ? `<div class="hyp-rel"><b>relationships:</b> ${Object.entries(h.relationships).map(([k, v]) => `${esc(k)} <i>(${esc(v)})</i>`).join(' · ')}</div>` : '') +
      (h.action_if_state ? `<div class="hyp-act">→ if true: ${esc(h.action_if_state)}</div>` : '') +
      (w == null ? '' : `<div class="hyp-bar"><i style="width:${Math.round(w * 100)}%"></i></div>`) +
      `</div>`;
  });
  view.innerHTML = html;
}

/* ---------------- transport ---------------- */
function bindTransport() {
  $('#btn-play').onclick = togglePlay;
  $('#btn-step').onclick = () => { stop(); renderAt(S.idx + 1); };
  $('#btn-back').onclick = () => { stop(); renderAt(S.idx - 1); };
  $('#btn-restart').onclick = () => { stop(); renderAt(-1); };
  $('#scrub').oninput = (e) => { stop(); renderAt(+e.target.value); };
  $('#speed').onchange = (e) => { S.speed = +e.target.value; if (S.playing) { stop(); play(); } };
  document.querySelectorAll('.side-tab').forEach((t) => t.onclick = () => setTab(t.dataset.tab));
  window.addEventListener('resize', () => renderAt(S.idx));
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT') return;
    if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
    if (e.code === 'ArrowRight') { stop(); renderAt(S.idx + 1); }
    if (e.code === 'ArrowLeft') { stop(); renderAt(S.idx - 1); }
  });
}
function renderTransport(st) {
  $('#scrub').value = String(Math.max(0, S.idx));
  $('#step-count').textContent = `${S.idx + 1} / ${S.events.length}`;
  $('#phase-now').textContent = st.phase || '—';
  $('#btn-play').textContent = S.playing ? '❚❚' : '▶';
}
function togglePlay() { S.playing ? stop() : play(); }
function play() {
  if (S.idx >= S.events.length - 1) S.idx = -1;
  S.playing = true; $('#btn-play').textContent = '❚❚';
  S.timer = setInterval(() => {
    if (S.idx >= S.events.length - 1) { stop(); return; }
    renderAt(S.idx + 1);
  }, S.speed);
}
function stop() { S.playing = false; if (S.timer) clearInterval(S.timer); S.timer = null; const b = $('#btn-play'); if (b) b.textContent = '▶'; }
function setTab(tab) {
  S.activeTab = tab;
  document.querySelectorAll('.side-tab').forEach((t) => t.classList.toggle('active', t.dataset.tab === tab));
  document.querySelectorAll('.side-panel').forEach((p) => p.classList.toggle('active', p.id === 'panel-' + tab));
}
function jumpToPhase(phase) {
  stop(); const i = S.events.findIndex((e) => e.type === 'phase' && e.phase === phase);
  if (i >= 0) renderAt(i);
}

boot();
