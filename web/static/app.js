// AI Interview Coach — front end (no framework). Talks to web/server.py on 127.0.0.1.
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];

const S = {
  lang: "en", coach: "ollama", coaches: [], bank: [], questions: [], pick: 0,
};

// ---------------- helpers ----------------
function toast(msg, ms = 3500) {
  const t = $("#toast"); t.textContent = msg; t.classList.add("show");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove("show"), ms);
}
async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let d = r.statusText;
    try { d = (await r.json()).detail || d; } catch (_) {}
    throw new Error(d);
  }
  const ct = r.headers.get("content-type") || "";
  return ct.includes("json") ? r.json() : r.text();
}
const postJSON = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
const qText = q => (S.lang === "th" ? q.th : q.en) || q.en || q.th || "";
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const sleep = ms => new Promise(r => setTimeout(r, ms));
const isLocal = id => (S.coaches.find(c => c.id === id) || {}).local;

function needConsent() {
  if (isLocal(S.coach) || $("#consent").checked) return false;
  go("home");
  const box = $("#consentBox"); box.classList.remove("need"); void box.offsetWidth; box.classList.add("need");
  box.scrollIntoView({ behavior: "smooth", block: "center" });
  toast("โค้ช ☁️ ต้องติ๊กยินยอมก่อน หรือเลือกโค้ช 🔒 ในเครื่อง");
  return true;
}

// ---------------- navigation ----------------
function go(view) {
  $$(".view").forEach(v => v.classList.toggle("active", v.id === "v-" + view));
  window.scrollTo({ top: 0, behavior: "smooth" });
  if (view === "practice") renderPractice();
  if (view === "history") loadHistory();
  if (view !== "practice") pRec.stop();
  if (view !== "interview") iRec.stop();
}
document.addEventListener("click", e => {
  const b = e.target.closest("[data-go]");
  if (b) go(b.dataset.go);
});

// ---------------- recorder ----------------
class Recorder {
  constructor(el, onChange) {
    this.el = el; this.onChange = onChange; this.stream = null; this.mr = null; this.file = null;
    el.classList.add("rec");
    el.innerHTML = `
      <div class="stage">
        <video playsinline muted></video>
        <div class="placeholder">กด 📷 เปิดกล้อง แล้วนั่งหน้าตรง แสงด้านหน้าพอ มองเลนส์<br>หรืออัปโหลดคลิปที่อัดไว้</div>
        <div class="count hidden"></div><div class="timer hidden">0:00</div>
      </div>
      <div class="ctrls"></div>
      <p class="hint">💡 คำตอบที่ดีมักยาว 1–2 นาที · กล้องระดับสายตา</p>`;
    this.video = $("video", el); this.ph = $(".placeholder", el);
    this.count = $(".count", el); this.timer = $(".timer", el); this.ctrls = $(".ctrls", el);
    this.state("idle");
  }
  btn(label, cls, fn) {
    const b = document.createElement("button"); b.className = "btn " + cls; b.innerHTML = label; b.onclick = fn;
    this.ctrls.appendChild(b); return b;
  }
  uploadBtn() {
    const l = document.createElement("label"); l.className = "btn ghost file";
    l.innerHTML = '📁 อัปโหลดคลิป<input type="file" accept="video/*" hidden>';
    $("input", l).onchange = e => { const f = e.target.files[0]; if (f) this.useFile(f); };
    this.ctrls.appendChild(l);
  }
  state(s) {
    this.ctrls.innerHTML = "";
    if (s === "idle") { this.btn("📷 เปิดกล้อง", "", () => this.open()); this.uploadBtn(); }
    if (s === "ready") { this.btn("● เริ่มอัด", "red", () => this.start()); this.uploadBtn(); }
    if (s === "recording") this.btn("■ หยุดอัด", "red", () => this.finish());
    if (s === "done") { this.btn("↺ อัดใหม่", "ghost", () => this.redo()); this.uploadBtn(); }
  }
  async open() {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: true });
    } catch (e) { toast("เปิดกล้อง/ไมค์ไม่ได้: " + e.message); return; }
    this.video.srcObject = this.stream; this.video.muted = true; this.video.controls = false;
    this.video.classList.add("mirror"); await this.video.play(); this.ph.classList.add("hidden");
    this.state("ready");
  }
  async start() {
    this.ctrls.innerHTML = "";
    this.count.classList.remove("hidden");
    for (const n of [3, 2, 1]) { this.count.textContent = n; await sleep(800); }
    this.count.classList.add("hidden");
    const mime = ["video/webm;codecs=vp8,opus", "video/webm", "video/mp4"].find(m => MediaRecorder.isTypeSupported(m)) || "";
    this.chunks = []; this.mr = new MediaRecorder(this.stream, mime ? { mimeType: mime } : {});
    this.mr.ondataavailable = e => e.data.size && this.chunks.push(e.data);
    this.mr.onstop = () => {
      const type = this.mr.mimeType || "video/webm";
      const ext = type.includes("mp4") ? "mp4" : "webm";
      this.useFile(new File([new Blob(this.chunks, { type })], `answer.${ext}`, { type }), true);
    };
    this.mr.start(1000); this.t0 = Date.now();
    this.timer.classList.remove("hidden");
    this.tick = setInterval(() => {
      const s = Math.floor((Date.now() - this.t0) / 1000);
      this.timer.textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
    }, 250);
    this.state("recording");
  }
  finish() { clearInterval(this.tick); this.timer.classList.add("hidden"); if (this.mr && this.mr.state !== "inactive") this.mr.stop(); }
  useFile(f, recorded = false) {
    this.file = f;
    if (!recorded) this.stopStream();
    this.video.srcObject = null; this.video.src = URL.createObjectURL(f);
    this.video.muted = false; this.video.controls = true; this.video.classList.remove("mirror");
    this.ph.classList.add("hidden"); this.state("done"); this.onChange && this.onChange(f);
  }
  async redo() {
    this.file = null; this.onChange && this.onChange(null);
    this.video.removeAttribute("src"); this.video.load();
    if (this.stream) { this.video.srcObject = this.stream; this.video.muted = true; this.video.controls = false;
      this.video.classList.add("mirror"); await this.video.play(); this.state("ready"); }
    else { this.ph.classList.remove("hidden"); this.state("idle"); }
  }
  reset() { this.file = null; this.video.removeAttribute("src"); this.video.load(); this.redo(); }
  stopStream() { if (this.stream) { this.stream.getTracks().forEach(t => t.stop()); this.stream = null; } }
  stop() { this.finish(); this.stopStream(); }
}

// ---------------- analysis jobs ----------------
async function submit(file, q) {
  const fd = new FormData();
  fd.append("video", file, file.name);
  fd.append("question", JSON.stringify(q));
  fd.append("lang", S.lang);
  fd.append("coach", S.coach);
  fd.append("consent", $("#consent").checked);
  fd.append("fast", $("#fast").checked);
  fd.append("job", $("#job").value);
  fd.append("resume", $("#resume").value);
  fd.append("skip", $$("#skips input:checked").map(i => i.value).join(","));
  return (await api("/api/analyse", { method: "POST", body: fd })).job;
}
async function waitJob(id, onTick) {
  for (;;) {
    const j = await api("/api/jobs/" + id);
    onTick && onTick(j);
    if (j.status === "done" || j.status === "error") return j;
    await sleep(1500);
  }
}
function waitBox(title) {
  return `<div class="spin"></div><h3>${title}</h3>
    <div class="bar"><i></i></div><p class="note wtext">เตรียมคลิป ...</p>
    <p class="note">Typhoon ในเครื่องใช้ราว 2–10 นาทีต่อคลิป ปิดหน้านี้ไม่ได้ แต่สลับไปทำอย่างอื่นได้</p>`;
}

// ---------------- practice ----------------
const pRec = new Recorder($("#pRecorder"), f => ($("#pGo").disabled = !f));
function renderPractice() {
  const list = $("#qList"); list.innerHTML = "";
  S.questions.forEach((q, i) => {
    const b = document.createElement("button");
    b.className = i === S.pick ? "on" : ""; b.textContent = `${i + 1}. ${qText(q)}`;
    b.onclick = () => { S.pick = i; renderPractice(); };
    list.appendChild(b);
  });
  const q = S.questions[S.pick] || {};
  $("#pQuestion").textContent = qText(q);
  $("#pWhy").textContent = q.why ? "💬 " + q.why : "";
}
$("#pGo").onclick = async () => {
  if (needConsent() || !pRec.file) return;
  const q = S.questions[S.pick];
  $("#practiceSetup").classList.add("hidden");
  const w = $("#practiceWait"); w.classList.remove("hidden"); w.innerHTML = waitBox("กำลังวิเคราะห์คำตอบ");
  pRec.stopStream();
  try {
    const id = await submit(pRec.file, q);
    const j = await waitJob(id, j => {
      $(".bar i", w).style.width = Math.round((j.progress || 0) * 100) + "%";
      $(".wtext", w).textContent = j.text || "";
    });
    if (j.status === "error") throw new Error(j.error);
    $("#practiceReport").innerHTML = await api("/api/report/" + j.key);
    $("#practiceReport").classList.remove("hidden");
  } catch (e) {
    $("#practiceReport").innerHTML = `<p class="note err">วิเคราะห์ไม่สำเร็จ: ${esc(e.message)}<br>ดูรายละเอียดในหน้าต่าง command line</p>`;
    $("#practiceReport").classList.remove("hidden");
  }
  w.classList.add("hidden"); $("#pAgain").classList.remove("hidden");
};
$("#pAgain").onclick = () => {
  $("#practiceReport").classList.add("hidden"); $("#pAgain").classList.add("hidden");
  $("#practiceSetup").classList.remove("hidden"); pRec.reset(); $("#pGo").disabled = true;
};

// ---------------- interview session ----------------
const iRec = new Recorder($("#iRecorder"), f => ($("#iNext").disabled = !f));
const I = { qs: [], idx: 0, jobs: [] };
$("#iN").oninput = e => ($("#iNLabel").textContent = e.target.value);
$("#iStartBtn").onclick = () => {
  if (needConsent()) return;
  I.qs = S.questions.slice(0, +$("#iN").value); I.idx = 0; I.jobs = [];
  $("#iStart").classList.add("hidden"); $("#iRun").classList.remove("hidden");
  showQuestion();
};
function showQuestion() {
  const q = I.qs[I.idx], last = I.idx === I.qs.length - 1;
  $("#iNum").textContent = `ข้อ ${I.idx + 1} / ${I.qs.length}`;
  $("#iQuestion").textContent = qText(q);
  $("#iWhy").textContent = q.why ? "💬 " + q.why : "";
  $("#iNext").textContent = last ? "✅ ส่งข้อสุดท้ายและดูผล" : "ส่งคำตอบ ไปข้อต่อไป ➜";
  $("#iNext").disabled = true;
  $("#iSteps").innerHTML = I.qs.map((_, i) => `<i class="${i < I.idx ? "done" : i === I.idx ? "now" : ""}"></i>`).join("");
  iRec.reset();
}
async function sendCurrent() {
  const q = I.qs[I.idx];
  try { I.jobs.push({ id: await submit(iRec.file, q), q }); }
  catch (e) { toast("ส่งข้อ " + (I.idx + 1) + " ไม่สำเร็จ: " + e.message); }
  I.idx += 1;
}
$("#iNext").onclick = async () => {
  if (!iRec.file) return;
  $("#iNext").disabled = true;
  await sendCurrent();
  if (I.idx >= I.qs.length) return finishSession();
  showQuestion(); toast("ส่งแล้ว ระบบวิเคราะห์ข้อก่อนหน้าอยู่เบื้องหลัง");
};
$("#iStop").onclick = async () => {
  if (iRec.file && I.idx < I.qs.length) await sendCurrent();
  finishSession();
};
async function finishSession() {
  iRec.stop(); $("#iRun").classList.add("hidden");
  const w = $("#iWait"); w.classList.remove("hidden");
  if (!I.jobs.length) {
    w.innerHTML = "<p>หยุดแล้ว ยังไม่ได้ตอบสักข้อ</p>"; $("#iAgain").classList.remove("hidden"); return;
  }
  w.innerHTML = waitBox(`กำลังวิเคราะห์ ${I.jobs.length} ข้อ`) + '<div class="jobs"></div>';
  const box = $(".jobs", w);
  box.innerHTML = I.jobs.map((j, i) => `<div class="job" id="job-${j.id}"><span class="dot"></span><b>ข้อ ${i + 1}</b><span class="jt">รอคิว</span></div>`).join("");
  const states = {};
  const update = () => {
    const vals = Object.values(states);
    const p = vals.reduce((a, j) => a + (j.status === "done" || j.status === "error" ? 1 : j.progress || 0), 0) / I.jobs.length;
    $(".bar i", w).style.width = Math.round(p * 100) + "%";
    const run = vals.find(j => j.status === "running");
    $(".wtext", w).textContent = run ? run.text : "";
  };
  await Promise.all(I.jobs.map(j => waitJob(j.id, s => {
    states[j.id] = s;
    const row = $("#job-" + j.id); row.className = "job " + s.status;
    $(".jt", row).textContent = s.status === "done" ? "เสร็จแล้ว" : s.status === "error" ? "ไม่สำเร็จ" : (s.text || "");
    update();
  }).catch(() => null)));
  try {
    $("#iReport").innerHTML = await postJSON("/api/session",
      { planned: I.qs.length, jobs: I.jobs.map(j => j.id), job: $("#job").value, lang: S.lang });
  } catch (e) { $("#iReport").innerHTML = `<p class="note err">${esc(e.message)}</p>`; }
  w.classList.add("hidden"); $("#iReport").classList.remove("hidden"); $("#iAgain").classList.remove("hidden");
}
$("#iAgain").onclick = () => {
  ["#iReport", "#iWait", "#iAgain"].forEach(s => $(s).classList.add("hidden"));
  $("#iStart").classList.remove("hidden");
};

// ---------------- history ----------------
async function loadHistory() {
  $("#hReport").classList.add("hidden"); $("#hBack").classList.add("hidden");
  const list = $("#hList"); list.classList.remove("hidden"); list.innerHTML = '<p class="note">กำลังโหลด ...</p>';
  try {
    const items = await api("/api/history");
    list.innerHTML = items.length ? "" : '<div class="card"><p>ยังไม่มีผล ลองฝึกสักข้อก่อนนะ</p></div>';
    items.forEach(it => {
      const b = document.createElement("button"); b.className = "history-item";
      b.innerHTML = `<span class="flag">${it.lang === "th" ? "🇹🇭" : "🇬🇧"}</span>
        <span><div class="q">${esc(it.question || it.key)}</div><div class="when">${esc(it.when)}</div></span>`;
      b.onclick = async () => {
        list.classList.add("hidden");
        $("#hReport").innerHTML = await api("/api/report/" + it.key);
        $("#hReport").classList.remove("hidden"); $("#hBack").classList.remove("hidden");
        window.scrollTo({ top: 0 });
      };
      list.appendChild(b);
    });
  } catch (e) { list.innerHTML = `<p class="note err">${esc(e.message)}</p>`; }
}
$("#hBack").onclick = loadHistory;

// ---------------- home: settings, resume, questions ----------------
$$("#langPills button").forEach(b => b.onclick = () => {
  S.lang = b.dataset.lang; $$("#langPills button").forEach(x => x.classList.toggle("on", x === b));
});
function renderCoaches() {
  $("#coaches").innerHTML = "";
  S.coaches.forEach(c => {
    const b = document.createElement("button");
    b.className = c.id === S.coach ? "on" : "";
    b.textContent = (c.local ? "🔒 " : "☁️ ") + c.label;
    b.onclick = () => { S.coach = c.id; renderCoaches(); };
    $("#coaches").appendChild(b);
  });
}
$("#resumeFile").onchange = async e => {
  const f = e.target.files[0]; if (!f) return;
  const fd = new FormData(); fd.append("file", f, f.name);
  $("#resumeNote").textContent = "กำลังอ่าน ...";
  try { $("#resume").value = (await api("/api/resume", { method: "POST", body: fd })).text;
    $("#resumeNote").textContent = "✅ อ่าน resume แล้ว แก้ข้อความได้ด้านล่าง"; }
  catch (err) { $("#resumeNote").textContent = err.message; }
};
$("#genBtn").onclick = async () => {
  if ($("#resume").value.trim() && !$("#consent").checked) {
    const box = $("#consentBox"); box.classList.remove("need"); void box.offsetWidth; box.classList.add("need");
    box.scrollIntoView({ behavior: "smooth", block: "center" });
    return toast("สร้างคำถามจาก resume ต้องส่งให้ Gemini ติ๊กยินยอมก่อนนะ");
  }
  $("#genBtn").disabled = true; $("#genNote").textContent = "✨ AI กำลังอ่าน resume และสร้างคำถาม ...";
  try {
    const r = await postJSON("/api/generate", { job: $("#job").value, resume: $("#resume").value, n: +$("#nq").value, consent: $("#consent").checked });
    S.questions = r.questions; S.pick = 0;
    $("#genNote").textContent = "✨ " + r.note + " · ไปที่ ฝึกทีละข้อ หรือ สัมภาษณ์ต่อเนื่อง ได้เลย";
  } catch (e) { $("#genNote").textContent = e.message; $("#genNote").classList.add("err"); }
  $("#genBtn").disabled = false;
};

(async function init() {
  try {
    const c = await api("/api/config");
    S.coaches = c.coaches; S.bank = c.questions; S.questions = c.questions.slice();
    renderCoaches();
  } catch (e) { toast("ต่อเซิร์ฟเวอร์ไม่ได้: " + e.message, 8000); }
})();
