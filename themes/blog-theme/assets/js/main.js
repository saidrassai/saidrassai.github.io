// Copy-link button in the share bar
document.querySelectorAll(".share-copy").forEach((el) => {
  el.addEventListener("click", () => {
    const url = el.getAttribute("data-url") || window.location.href;
    const done = () => {
      el.setAttribute("data-copied", "");
      setTimeout(() => el.removeAttribute("data-copied"), 1200);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(url).then(done).catch(done);
    } else {
      const ta = document.createElement("textarea");
      ta.value = url;
      document.body.appendChild(ta);
      ta.select();
      try { document.execCommand("copy"); } catch (e) {}
      document.body.removeChild(ta);
      done();
    }
  });
});

// Highlight the active TOC entry while scrolling
(function () {
  const toc = document.querySelector(".rail-toc #TableOfContents");
  if (!toc) return;
  const links = Array.from(toc.querySelectorAll("a[href^='#']"));
  if (!links.length) return;
  const byId = new Map(links.map((a) => [decodeURIComponent(a.hash.slice(1)), a]));
  const headings = Array.from(document.querySelectorAll(".prose h2[id], .prose h3[id]"))
    .filter((h) => byId.has(h.id));
  if (!headings.length) return;
  let active = null;
  const update = () => {
    let current = headings[0];
    for (const h of headings) {
      if (h.getBoundingClientRect().top <= 90) current = h;
      else break;
    }
    if (active === current) return;
    active = current;
    links.forEach((a) => (a.style.color = ""));
    const link = byId.get(current.id);
    if (link) link.style.color = "var(--red)";
  };
  document.addEventListener("scroll", update, { passive: true });
  update();
})();
