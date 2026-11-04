/* Figure embedding, ported from the Zephon blog post's FigureCarousel block in
 * ../datology-web/src/payload/blocks/FigureCarousel/Component.tsx. The post's
 * hash deep-linking into a slide is left out: the guide does not link to one. */

/* The embedded figures measure themselves and ask for a height, exactly as they
 * do on the blog. A same-origin frame sets its own height directly; this
 * listener is the fallback the figure uses when it cannot reach its frame
 * element, which is the case when the built docs are opened over file://.
 * Without it the taller figures keep the page's initial height and clip. */
window.addEventListener("message", (event) => {
  if (!event.data || event.data.type !== "EMBED_RESIZE") return;
  const height = Number(event.data.height);
  if (!Number.isFinite(height) || height <= 0) return;
  for (const frame of document.querySelectorAll(".zephon-post iframe")) {
    // A few pixels of slack: the figure reports the height of its own element,
    // and sizing the frame to exactly that leaves a sub-pixel rounding
    // difference between engines enough to raise a scrollbar.
    if (frame.contentWindow === event.source) frame.style.height = `${height + 4}px`;
  }
});

function setUpCarousel(root) {
  const viewport = root.querySelector(".figure-carousel-viewport");
  const items = [...root.querySelectorAll(".figure-carousel-item")];
  const choices = [...root.querySelectorAll(".figure-carousel-choices button")];
  const count = root.querySelector(".figure-carousel-count");
  const live = root.querySelector(".figure-carousel-live");
  let active = 0;

  /* The viewport is sized to the slide on show, because the others are taken
   * out of flow. The figures inside report their own height as they settle, so
   * the slide is re-measured rather than measured once. */
  const measure = () => {
    viewport.style.height = `${items[active].getBoundingClientRect().height}px`;
  };

  function select(index) {
    const next = ((index % items.length) + items.length) % items.length;
    // Do not leave keyboard focus in a slide that is about to go inert.
    if (next !== active && items[active].contains(document.activeElement)) {
      root.focus({ preventScroll: true });
    }
    active = next;
    items.forEach((item, i) => {
      item.dataset.active = String(i === next);
      item.setAttribute("aria-hidden", String(i !== next));
      item.toggleAttribute("inert", i !== next);
    });
    choices.forEach((button, i) => {
      button.setAttribute("aria-pressed", String(i === next));
    });
    if (count) count.textContent = `${next + 1} / ${items.length}`;
    if (live) live.textContent = `${next + 1} of ${items.length}: ${choices[next].textContent.trim()}`;
    measure();
  }

  choices.forEach((button, i) => button.addEventListener("click", () => select(i)));
  root
    .querySelector('[data-carousel-step="previous"]')
    ?.addEventListener("click", () => select(active - 1));
  root
    .querySelector('[data-carousel-step="next"]')
    ?.addEventListener("click", () => select(active + 1));

  if ("ResizeObserver" in window) {
    const observer = new ResizeObserver(measure);
    items.forEach((item) => observer.observe(item));
  }
  select(0);
}

window.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".figure-carousel").forEach(setUpCarousel);
});
