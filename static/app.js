(() => {
  "use strict";

  /* ------------------------------------------------------------------ */
  /* helpers                                                            */
  /* ------------------------------------------------------------------ */
  const $ = (sel, el = document) => el.querySelector(sel);
  const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
  const esc = (s) =>
    String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  async function api(url, opts) {
    const res = await fetch(url, opts);
    let body = null;
    try { body = await res.json(); } catch { /* ignore */ }
    if (!res.ok) throw new Error((body && body.error) || res.statusText || "Request failed");
    return body;
  }

  const meta = (a) =>
    [a.format, a.year, a.episodes ? `${a.episodes} ep` : null].filter(Boolean).join(", ");

  const fmtNum = (n) => (n == null ? "-" : Number(n).toLocaleString());

  function poster(a, { tl = "", br = "", bl = "", sub } = {}) {
    const img = a.cover
      ? `<img src="${esc(a.cover)}" alt="" loading="lazy" referrerpolicy="no-referrer">`
      : `<span class="nocover">${esc((a.title || "?").slice(0, 1))}</span>`;
    return `<button type="button" class="poster" data-id="${a.id}">
      <span class="cover">${img}${tl}${br}${bl}</span>
      <span class="ptitle">${esc(a.title)}</span>
      <span class="pmeta">${esc(sub ?? meta(a))}</span>
    </button>`;
  }

  const chip = (cls, text) => `<span class="chip ${cls}">${esc(text)}</span>`;

  function mineChip(mine) {
    if (!mine) return "";
    if (mine.status === "Plan to Watch") return chip("bl plan", "Plan to watch");
    if (mine.seen) return chip("bl seen", mine.score ? `Seen, ${mine.score}` : "Seen");
    return "";
  }

  /* ------------------------------------------------------------------ */
  /* detail drawer: shows an anime and the titles similar to it         */
  /* ------------------------------------------------------------------ */
  const drawerEl = $("#drawer");
  const bodyEl = $("#drawer-body");
  const scrim = $("#scrim");
  const backBtn = $("#drawer-back");
  const stack = [];
  let lastFocus = null;
  let current = null; // last payload from /api/anime/<id>
  let hideSeen = false;

  function setOpen(open) {
    drawerEl.classList.toggle("open", open);
    drawerEl.setAttribute("aria-hidden", String(!open));
    scrim.hidden = !open;
    document.body.classList.toggle("noscroll", open);
  }

  async function openAnime(id, { replace = false } = {}) {
    if (!drawerEl.classList.contains("open")) lastFocus = document.activeElement;
    setOpen(true);
    if (!replace) stack.push(id);
    backBtn.hidden = stack.length < 2;
    bodyEl.innerHTML = `<p class="loading">Loading...</p>`;
    drawerEl.scrollTop = 0;
    drawerEl.focus({ preventScroll: true });
    try {
      const data = await api(`/api/anime/${id}`);
      if (stack[stack.length - 1] !== id) return; // user navigated elsewhere meanwhile
      current = data;
      renderDrawer();
    } catch (err) {
      bodyEl.innerHTML = `<p class="error">${esc(err.message)}</p>`;
    }
  }

  function closeDrawer() {
    stack.length = 0;
    setOpen(false);
    if (lastFocus && lastFocus.focus) lastFocus.focus({ preventScroll: true });
  }

  function goBack() {
    if (stack.length < 2) return;
    stack.pop();
    openAnime(stack[stack.length - 1], { replace: true });
  }

  function renderDrawer() {
    const { anime: a, has_list: hasList } = current;
    const cover = a.cover
      ? `<span class="cover"><img src="${esc(a.cover)}" alt="" referrerpolicy="no-referrer"></span>`
      : `<span class="cover"><span class="nocover">${esc(a.title.slice(0, 1))}</span></span>`;

    const facts = [
      ["MAL score", a.score != null ? a.score.toFixed(2) : null],
      ["Popularity", a.popularity != null ? `#${fmtNum(a.popularity)}` : null],
      ["Rank", a.rank != null ? `#${fmtNum(a.rank)}` : null],
      ["Members", a.members != null ? fmtNum(a.members) : null],
      ["Your score", a.mine && a.mine.score ? a.mine.score : null],
    ].filter(([, v]) => v != null)
      .map(([k, v]) => `<div><dt>${k}</dt><dd>${esc(v)}</dd></div>`).join("");

    const mineTag = a.mine
      ? `<span class="tag ${a.mine.status === "Plan to Watch" ? "plan" : "seen"}">${esc(a.mine.status)}</span>` : "";

    const pills = [
      ...a.genres.map((g) => `<span class="pill">${esc(g)}</span>`),
      ...a.tags.slice(0, 8).map((t) => `<span class="pill soft">${esc(t)}</span>`),
    ].join("");

    const sub = [meta(a), a.season, a.studios.join(", ")].filter(Boolean).join(", ");
    const alt = [a.romaji, a.native].filter(Boolean).join(", ");

    bodyEl.innerHTML = `
      <div class="d-top">
        ${cover}
        <div>
          <h2>${esc(a.title)}${mineTag}</h2>
          ${alt ? `<p class="alt">${esc(alt)}</p>` : ""}
          <p class="alt">${esc(sub)}</p>
          <dl class="facts">${facts}</dl>
          <div class="pills">${pills}</div>
          <p class="d-links"><a href="https://myanimelist.net/anime/${a.id}" target="_blank" rel="noopener">Open on MyAnimeList</a></p>
        </div>
      </div>
      ${a.description ? `<p class="desc">${esc(a.description)}</p>` : ""}
      <div class="d-section">
        <h3>Similar anime</h3>
        ${hasList ? `<label><input type="checkbox" id="hide-seen" ${hideSeen ? "checked" : ""}> Hide ones I've already seen</label>` : ""}
      </div>
      <div class="sims" id="sims"></div>`;

    const hs = $("#hide-seen", bodyEl);
    if (hs) hs.addEventListener("change", () => { hideSeen = hs.checked; renderSims(); });
    renderSims();
  }

  function renderSims() {
    const box = $("#sims", bodyEl);
    if (!box) return;
    const all = current.similar;
    const list = hideSeen ? all.filter((s) => !(s.mine && s.mine.seen)) : all;
    if (!all.length) {
      box.outerHTML = `<p class="empty" id="sims">No similarity data for this anime.</p>`;
      return;
    }
    if (!list.length) {
      box.innerHTML = `<p class="empty">You've seen all of the similar anime we have.</p>`;
      return;
    }
    const max = Math.max(...list.map((s) => s.similarity)) || 1;
    box.innerHTML = list.map((s) => {
      const pct = Math.max(4, Math.round((s.similarity / max) * 100));
      return poster(s, {
        bl: mineChip(s.mine),
        br: s.score != null ? chip("br", s.score.toFixed(1)) : "",
        sub: meta(s),
      }).replace(
        /<\/button>$/,
        `<span class="simbar"><i style="width:${pct}%"></i></span><span class="simval">Match ${s.similarity.toFixed(3)}</span></button>`
      );
    }).join("");
  }

  $("#drawer-close").addEventListener("click", closeDrawer);
  backBtn.addEventListener("click", goBack);
  scrim.addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && drawerEl.classList.contains("open")) closeDrawer();
  });

  // One delegated handler for every clickable anime (posters, recommendations).
  function activate(e) {
    const el = e.target.closest("[data-id]");
    if (!el || el.closest("#hide-seen")) return;
    openAnime(Number(el.dataset.id));
  }
  document.addEventListener("click", activate);
  document.addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === " ") && e.target.matches(".rec[data-id]")) {
      e.preventDefault();
      activate(e);
    }
  });

  /* ------------------------------------------------------------------ */
  /* page 1: popular                                                    */
  /* ------------------------------------------------------------------ */
  function initPopular() {
    const grid = $("#grid"), more = $("#more"), empty = $("#empty"), count = $("#count"), search = $("#search");
    let page = 1, query = "", busy = false, token = 0;

    async function load(reset) {
      if (busy && !reset) return;
      if (reset) { page = 1; grid.innerHTML = ""; }
      const mine = ++token;
      busy = true;
      more.disabled = true;
      try {
        const d = await api(`/api/popular?page=${page}&q=${encodeURIComponent(query)}`);
        if (mine !== token) return;
        grid.insertAdjacentHTML("beforeend", d.items.map((a) => poster(a, {
          tl: a.popularity != null ? chip("tl", `#${fmtNum(a.popularity)}`) : "",
          br: a.score != null ? chip("br score", a.score.toFixed(1)) : "",
        })).join(""));
        count.textContent = query
          ? `${fmtNum(d.total)} matches, ordered by popularity rank.`
          : `${fmtNum(d.total)} anime, ordered by popularity rank. Select a title to see what's similar.`;
        empty.hidden = d.total > 0;
        more.hidden = !d.has_more;
        page += 1;
      } catch (err) {
        count.textContent = err.message;
      } finally {
        if (mine === token) { busy = false; more.disabled = false; }
      }
    }

    let t;
    search.addEventListener("input", () => {
      clearTimeout(t);
      t = setTimeout(() => { query = search.value; load(true); }, 250);
    });
    more.addEventListener("click", () => load(false));
    load(true);
  }

  /* ------------------------------------------------------------------ */
  /* page 2: my list                                                    */
  /* ------------------------------------------------------------------ */
  function initMyList() {
    const views = { upload: $("#upload-view"), list: $("#list-view") };
    const fileInput = $("#file"), drop = $("#drop"), errBox = $("#upload-error");
    const cancelBtn = $("#cancel-replace");
    const STATUSES = ["All", "Completed", "Watching", "On-Hold", "Dropped", "Plan to Watch"];
    const STATUS_LABEL = { "On-Hold": "On hold", "Plan to Watch": "Plan to watch" };
    const PAGE = 120;

    let entries = [], statusFilter = "All", listQuery = "", shown = PAGE, recsLoaded = false;

    function show(which) {
      views.upload.hidden = which !== "upload";
      views.list.hidden = which !== "list";
    }

    async function init() {
      let d;
      try { d = await api("/api/my-list"); } catch (err) { show("upload"); showError(err.message); return; }
      if (!d.loaded) { cancelBtn.hidden = true; show("upload"); return; }
      entries = d.entries;
      $("#who").textContent = d.user ? `${d.user}'s list` : "My list";
      const rated = entries.filter((e) => e.my_score > 0);
      const mean = rated.length ? (rated.reduce((s, e) => s + e.my_score, 0) / rated.length).toFixed(2) : "-";
      $("#summary").textContent =
        `${fmtNum(entries.length)} titles, ${fmtNum(rated.length)} rated, average score ${mean}.`;
      show("list");
      renderChips();
      renderList(true);
      recsLoaded = false;
      setTab("recs");
    }

    /* ---- upload ---- */
    function showError(msg) { errBox.textContent = msg; errBox.hidden = !msg; }

    async function upload(file) {
      if (!file) return;
      showError("");
      const fd = new FormData();
      fd.append("file", file);
      try {
        await api("/api/my-list", { method: "POST", body: fd });
        await init();
      } catch (err) {
        showError(err.message);
      }
    }
    fileInput.addEventListener("change", () => upload(fileInput.files[0]));
    ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
    drop.addEventListener("drop", (e) => upload(e.dataTransfer.files[0]));

    $("#replace").addEventListener("click", () => { fileInput.value = ""; showError(""); cancelBtn.hidden = false; show("upload"); });
    cancelBtn.addEventListener("click", () => show("list"));
    $("#remove").addEventListener("click", async () => {
      if (!confirm("Remove your list from this browser?")) return;
      await api("/api/my-list", { method: "DELETE" });
      closeDrawer();
      entries = [];
      cancelBtn.hidden = true;
      show("upload");
    });

    /* ---- tabs ---- */
    function setTab(name) {
      $$("[data-tab]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
      $("#panel-recs").hidden = name !== "recs";
      $("#panel-list").hidden = name !== "list";
      if (name === "recs" && !recsLoaded) loadRecs();
    }
    $$("[data-tab]").forEach((b) => b.addEventListener("click", () => setTab(b.dataset.tab)));

    /* ---- recommendations ---- */
    const recsEl = $("#recs"), recsEmpty = $("#recs-empty");
    async function loadRecs() {
      recsLoaded = true;
      recsEl.innerHTML = `<li class="loading">Working out your recommendations...</li>`;
      recsEmpty.hidden = true;
      const minScore = $("#min-score").value;
      const plan = $("#include-plan").checked ? 1 : 0;
      try {
        const d = await api(`/api/my-list/recommendations?min_score=${minScore}&include_plan=${plan}&limit=100`);
        recsEmpty.hidden = d.items.length > 0;
        recsEl.innerHTML = d.items.map((a) => {
          const why = a.reasons.length
            ? `<p class="why">Because you rated ${a.reasons.map((r) => `<b>${esc(r.title)}</b> ${r.score}`).join(", ")}${
                a.n_sources > a.reasons.length ? ` and ${a.n_sources - a.reasons.length} more` : ""}</p>` : "";
          const cover = a.cover
            ? `<img src="${esc(a.cover)}" alt="" loading="lazy" referrerpolicy="no-referrer">`
            : `<span class="nocover">${esc(a.title.slice(0, 1))}</span>`;
          return `<li class="rec" data-id="${a.id}" tabindex="0" role="button">
            <span class="cover">${cover}</span>
            <div>
              <h3>${esc(a.title)}${a.planned ? `<span class="tag plan">In your plan to watch</span>` : ""}</h3>
              <span class="pmeta">${esc(meta(a))}${a.score != null ? `, MAL ${a.score.toFixed(1)}` : ""}</span>
              <div class="bar"><span class="track"><i style="width:${Math.round(a.relative * 100)}%"></i></span>
                <span>Priority ${a.priority.toFixed(2)}</span></div>
              ${why}
            </div></li>`;
        }).join("");
      } catch (err) {
        recsEl.innerHTML = `<li class="error">${esc(err.message)}</li>`;
      }
    }
    $("#min-score").addEventListener("change", loadRecs);
    $("#include-plan").addEventListener("change", loadRecs);

    /* ---- list ---- */
    const grid = $("#grid"), more = $("#more"), empty = $("#empty");

    function renderChips() {
      const counts = Object.fromEntries(STATUSES.map((s) => [s, s === "All" ? entries.length : 0]));
      entries.forEach((e) => { if (counts[e.status] != null) counts[e.status] += 1; });
      $("#status-chips").innerHTML = STATUSES.filter((s) => counts[s] > 0).map((s) =>
        `<button type="button" data-status="${esc(s)}" aria-pressed="${s === statusFilter}">${esc(STATUS_LABEL[s] || s)} ${fmtNum(counts[s])}</button>`
      ).join("");
    }
    $("#status-chips").addEventListener("click", (e) => {
      const b = e.target.closest("[data-status]");
      if (!b) return;
      statusFilter = b.dataset.status;
      $$("#status-chips button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      renderList(true);
    });

    let t;
    $("#list-search").addEventListener("input", (e) => {
      clearTimeout(t);
      t = setTimeout(() => { listQuery = e.target.value.trim().toLowerCase(); renderList(true); }, 200);
    });
    more.addEventListener("click", () => { shown += PAGE; renderList(false); });

    function renderList(reset) {
      if (reset) shown = PAGE;
      const filtered = entries.filter((e) =>
        (statusFilter === "All" || e.status === statusFilter) &&
        (!listQuery || e.title.toLowerCase().includes(listQuery) || (e.romaji || "").toLowerCase().includes(listQuery)));
      grid.innerHTML = filtered.slice(0, shown).map((a) => poster(a, {
        tl: a.my_score ? chip("tl score", a.my_score) : "",
        br: a.status !== "Completed" ? chip("br", STATUS_LABEL[a.status] || a.status) : "",
        sub: meta(a) || a.status,
      })).join("");
      empty.hidden = filtered.length > 0;
      more.hidden = filtered.length <= shown;
    }

    init();
  }

  /* ------------------------------------------------------------------ */
  /* page 3: settings (score weights)                                   */
  /* ------------------------------------------------------------------ */
  function initSettings() {
    const form = $("#settings-form"), box = $("#scores"), split = $("#split");
    const status = $("#status"), meta = $("#meta"), apply = $("#apply"), reset = $("#reset");
    const COLORS = ["var(--accent)", "var(--plan)", "var(--seen)", "var(--navy)"];
    let scores = [], busy = false;

    const sliders = () => $$("input[type=range]", box);
    const total = () => sliders().reduce((s, el) => s + Number(el.value), 0);
    const say = (msg, cls = "") => { status.textContent = msg; status.className = `status ${cls}`; };

    function refresh() {
      const sum = total();
      sliders().forEach((el, i) => {
        const share = sum > 0 ? Math.round((Number(el.value) / sum) * 100) : 0;
        $(`#o-${el.dataset.key}`).textContent = `${share}%`;
        split.children[i].style.flexBasis = `${sum > 0 ? (Number(el.value) / sum) * 100 : 0}%`;
      });
      apply.disabled = busy || sum <= 0;
      reset.disabled = busy;
      if (sum <= 0) { say("Give at least one score a weight above zero.", "err"); status.dataset.zero = "1"; }
      else if (status.dataset.zero) { say(""); delete status.dataset.zero; }
    }

    function describe(d) {
      const bits = [];
      if (d.last) {
        const when = new Date(d.last.updated).toLocaleString();
        bits.push(`Last applied ${when}: ${fmtNum(d.last.pairs)} pairs scored.`);
      }
      bits.push(`${fmtNum(d.links)} similarity links in use. Recalculation is done by the weights API at ${d.api_url}.`);
      meta.textContent = bits.join(" ");
    }

    async function load() {
      try {
        const d = await api("/api/settings");
        scores = d.scores;
        box.innerHTML = scores.map((sc, i) => `
          <div class="score-row" data-i="${i}">
            <div class="score-head"><label for="w-${sc.key}">${esc(sc.label)}</label><output id="o-${sc.key}" for="w-${sc.key}"></output></div>
            <p class="hint">${esc(sc.help)}</p>
            <input type="range" id="w-${sc.key}" data-key="${sc.key}" min="0" max="1" step="0.05" value="${d.weights[sc.key] ?? 0.5}">
          </div>`).join("");
        split.innerHTML = scores.map((_, i) => `<i style="background:${COLORS[i % COLORS.length]}"></i>`).join("");
        sliders().forEach((el) => el.addEventListener("input", refresh));
        describe(d);
        refresh();
      } catch (err) {
        box.innerHTML = `<p class="error">${esc(err.message)}</p>`;
      }
    }

    reset.addEventListener("click", () => { sliders().forEach((el) => { el.value = 0.5; }); say(""); refresh(); });

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      if (busy || total() <= 0) return;
      busy = true; refresh();
      sliders().forEach((el) => { el.disabled = true; });
      say("Recalculating. This can take a while with large files...", "busy");
      const weights = Object.fromEntries(sliders().map((el) => [el.dataset.key, Number(el.value)]));
      try {
        const d = await api("/api/settings", {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(weights),
        });
        say(`Saved. ${fmtNum(d.pairs)} pairs rescored and ${fmtNum(d.links)} similarity links reloaded. New weights apply everywhere now.`, "ok");
        describe(await api("/api/settings"));
      } catch (err) {
        say(err.message, "err");
      } finally {
        busy = false;
        sliders().forEach((el) => { el.disabled = false; });
        refresh();
      }
    });

    load();
  }

  /* ------------------------------------------------------------------ */
  const page = document.body.dataset.page;
  if (page === "popular") initPopular();
  if (page === "mylist") initMyList();
  if (page === "settings") initSettings();
})();
