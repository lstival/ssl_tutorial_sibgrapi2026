/* ------------------------------------------------------------
   Page wiring: nav, reveals, notebook/model mounts, charts,
   KaTeX rendering, copy buttons, table filter.
   ------------------------------------------------------------ */

document.documentElement.classList.add("js");

const pending = SITE_STATUS !== "live";
const pendAttr = pending ? ' data-pending aria-disabled="true" title="Available once the code repository is public"' : "";
const href = (u) => (pending ? "#" : u);

/* ---------- repo links ---------- */
["navRepo", "heroRepo", "footRepo"].forEach((id) => {
  const a = document.getElementById(id);
  if (!a) return;
  a.href = href(GH);
  if (pending) { a.setAttribute("data-pending", ""); a.setAttribute("aria-disabled", "true"); }
  else a.target = "_blank";
});

if (!pending) {
  const n = document.getElementById("statusNote");
  if (n) n.hidden = true;
}

/* ---------- notebooks ---------- */
(function mountNotebooks() {
  const mount = document.getElementById("nbMount");
  if (!mount) return;
  let html = "";
  for (const [domain, block] of Object.entries(NOTEBOOKS)) {
    html += `<div class="domain-hd">
        <h3>${block.label}</h3><span>${block.sub}</span>
      </div><div class="nb-grid">`;
    for (const it of block.items) {
      const path = `${domain}/${it.f}`;
      const cls = it.k ? ` nb--${it.k}` : "";
      const tgt = pending ? "" : ' target="_blank" rel="noopener"';
      html += `<article class="nb${cls}">
          <p class="nb__n">NOTEBOOK ${it.n}</p>
          <h3>${it.t}</h3>
          <p>${it.d}</p>
          <div class="nb__links">
            <a href="${href(COLAB(path))}"${pendAttr}${tgt}>Colab ↗</a>
            <a href="${href(RAW_NB(path))}"${pendAttr}${tgt}>GitHub ↗</a>
          </div>
        </article>`;
    }
    html += `</div>`;
  }
  mount.innerHTML = html;
})();

/* ---------- models ---------- */
(function mountModels() {
  const mount = document.getElementById("mdlMount");
  if (!mount) return;
  const color = { con: "var(--c-con)", mask: "var(--c-mask)", dist: "var(--c-dist)" };
  mount.innerHTML = MODELS.map((m) => `<tr>
      <td class="nm"><span class="dot" style="background:${color[m.k]}"></span>${m.n}</td>
      <td>${m.p}</td><td>${m.m}</td><td>${m.a}</td><td>${m.d}</td><td>${m.s}</td>
    </tr>`).join("");
})();

/* ---------- nav ---------- */
const nav = document.getElementById("nav");
const onScroll = () => nav && nav.classList.toggle("stuck", window.scrollY > 8);
onScroll();
window.addEventListener("scroll", onScroll, { passive: true });

const navToggle = document.getElementById("navToggle");
const navLinks = document.getElementById("navLinks");
if (navToggle && navLinks) {
  navToggle.addEventListener("click", () => {
    const open = navLinks.classList.toggle("open");
    navToggle.setAttribute("aria-expanded", String(open));
  });
  navLinks.addEventListener("click", (e) => {
    if (e.target.tagName === "A") {
      navLinks.classList.remove("open");
      navToggle.setAttribute("aria-expanded", "false");
    }
  });
}

/* ---------- reveal on scroll ---------- */
if (!window.matchMedia("(prefers-reduced-motion:reduce)").matches && "IntersectionObserver" in window) {
  const io = new IntersectionObserver((entries) => {
    entries.forEach((en) => {
      if (en.isIntersecting) { en.target.classList.add("in"); io.unobserve(en.target); }
    });
  }, { rootMargin: "0px 0px -8% 0px", threshold: 0.04 });
  document.querySelectorAll(".rv").forEach((n) => io.observe(n));
} else {
  document.querySelectorAll(".rv").forEach((n) => n.classList.add("in"));
}

/* ---------- copy buttons ---------- */
document.querySelectorAll(".copy").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const src = document.getElementById(btn.dataset.copy);
    if (!src) return;
    try {
      await navigator.clipboard.writeText(src.innerText.trim());
      const old = btn.textContent;
      btn.textContent = "Copied";
      setTimeout(() => (btn.textContent = old), 1600);
    } catch (e) {
      btn.textContent = "Press ⌘C";
      setTimeout(() => (btn.textContent = "Copy"), 1600);
    }
  });
});

/* ---------- table filter ---------- */
document.querySelectorAll("[data-filter]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const f = btn.dataset.filter;
    btn.parentElement.querySelectorAll("button").forEach((b) =>
      b.setAttribute("aria-selected", String(b === btn)));
    document.querySelectorAll("tr[data-row]").forEach((tr) => {
      tr.hidden = !(f === "all" || tr.dataset.row === "all" || tr.dataset.row === f);
    });
  });
});

/* ---------- charts ---------- */
(async function initCharts() {
  await loadData();
  renderModality("rs");
  renderScaling();

  document.querySelectorAll("[data-mode]").forEach((btn) => {
    btn.addEventListener("click", () => {
      btn.parentElement.querySelectorAll("button").forEach((b) =>
        b.setAttribute("aria-selected", String(b === btn)));
      renderModality(btn.dataset.mode);
    });
  });
})();
