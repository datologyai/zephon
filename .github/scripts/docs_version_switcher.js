/* Version picker and "not the latest release" banner for the versioned docs.
 *
 * publish_docs.py copies this to _versions/switcher.js at the site root and
 * adds a tag for it to every page, carrying the page's version and its path
 * within that version. versions.json, beside _versions/, lists what exists:
 *   {"latest": "v0.2.1", "versions": ["main", "v0.2.1", "v0.2.0"]}
 * The latest release is served from the site root, every other version from a
 * directory named for it. */
(() => {
  const script = document.currentScript;
  if (!script) return;
  const current = script.dataset.version;
  const page = script.dataset.page || "index.html";
  const siteRoot = new URL("../", script.src);

  const baseFor = (version, latest) =>
    version === latest ? siteRoot : new URL(`${version}/`, siteRoot);

  const labelFor = (version, latest) => {
    if (version === latest) return `${version} (latest)`;
    if (version === "main") return "main (unreleased)";
    return version;
  };

  /* The same page in another version, keeping its query (a search) and anchor,
   * or that version's front page when the page doesn't exist there. Over
   * file:// the check can't run; go anyway. */
  async function go(version, latest) {
    const base = baseFor(version, latest);
    const target = new URL(page, base);
    target.search = location.search;
    target.hash = location.hash;
    try {
      const response = await fetch(target, { method: "HEAD" });
      location.href = response.ok ? target.href : base.href;
    } catch {
      location.href = target.href;
    }
  }

  function addStyles() {
    const style = document.createElement("style");
    style.textContent = `
      .zephon-versions { margin: 0 var(--sidebar-item-spacing-horizontal, 1rem) 0.75rem; }
      .zephon-versions label { display: block; font-size: var(--font-size--small, 0.875rem);
        color: var(--color-foreground-secondary); margin-bottom: 0.25rem; }
      /* Match Furo's sidebar search box rather than the darker content area. */
      .zephon-versions select { width: 100%; padding: 0.3rem 0.4rem; border-radius: 0.25rem;
        font: inherit; font-size: var(--font-size--small, 0.875rem);
        color: var(--color-sidebar-search-foreground, var(--color-foreground-primary));
        background: var(--color-sidebar-search-background, var(--color-sidebar-background));
        border: 1px solid var(--color-sidebar-search-border, var(--color-background-border)); }
      /* Furo themes with CSS variables only, so tell the browser when the
       * native dropdown list should be dark too. */
      body[data-theme="dark"] .zephon-versions select { color-scheme: dark; }
      @media (prefers-color-scheme: dark) {
        body:not([data-theme="light"]) .zephon-versions select { color-scheme: dark; }
      }
      /* Above Furo's floated page icons, at full width, with the same top
       * spacing the icons have. */
      .zephon-version-banner { margin: 1.5rem 0 0; padding: 0.6rem 0.9rem; border-radius: 0.25rem;
        font-size: var(--font-size--small, 0.875rem);
        background: var(--color-admonition-title-background--note, var(--color-background-secondary));
        border-left: 0.25rem solid var(--color-brand-primary); }
    `;
    document.head.append(style);
  }

  function addPicker(index) {
    const brand = document.querySelector(".sidebar-brand");
    if (!brand) return;
    const wrapper = document.createElement("div");
    wrapper.className = "zephon-versions";
    const label = document.createElement("label");
    label.textContent = "Version";
    label.htmlFor = "zephon-version-select";
    const select = document.createElement("select");
    select.id = "zephon-version-select";
    for (const version of index.versions) {
      const option = new Option(labelFor(version, index.latest), version);
      option.selected = version === current;
      select.append(option);
    }
    select.addEventListener("change", () => go(select.value, index.latest));
    wrapper.append(label, select);
    brand.after(wrapper);
  }

  function addBanner(index) {
    if (current === index.latest) return;
    const container =
      document.querySelector(".article-container") ||
      document.querySelector("article[role=main]") ||
      document.querySelector(".content");
    if (!container) return;
    let kind = "an older release";
    if (current === "main") kind = "the unreleased development version";
    else if (index.versions.indexOf(current) < index.versions.indexOf(index.latest)) kind = "a pre-release";
    const banner = document.createElement("div");
    banner.className = "zephon-version-banner";
    banner.setAttribute("role", "note");
    const link = document.createElement("a");
    link.href = new URL(page, siteRoot).href;
    link.textContent = `the latest release, ${index.latest}`;
    link.addEventListener("click", (event) => {
      event.preventDefault();
      go(index.latest, index.latest);
    });
    banner.append(`You're reading the docs for ${current}, ${kind}. See `, link, ".");
    container.prepend(banner);
  }

  fetch(new URL("versions.json", siteRoot))
    .then((response) => (response.ok ? response.json() : null))
    .then((index) => {
      if (!index || !Array.isArray(index.versions)) return;
      addStyles();
      addPicker(index);
      addBanner(index);
    })
    .catch(() => {});
})();
