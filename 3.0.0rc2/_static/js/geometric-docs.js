(() => {
  const header = document.querySelector(".md-header__inner");
  const navigation = document.querySelector(".md-header .md-tabs");
  const sidebar = document.querySelector(".md-sidebar--primary");
  const home = document.querySelector(".home-intro");
  const section = navigation?.querySelector(".md-tabs__item--active > a");
  const palette = document.querySelector('[data-md-component="palette"]');
  const radios = [...(palette?.querySelectorAll('input[name="__palette"]') || [])];
  const favicon = document.querySelector('link[rel="icon"]');
  const logos = document.querySelectorAll(".md-logo img");

  if (header && navigation) {
    navigation.classList.add("dashboard-tabbar--ruled");
    navigation.querySelector(".md-tabs__list").classList.add("dashboard-tabs");
    section?.setAttribute("aria-current", "true");
    header.querySelector(".md-header__title")?.after(navigation);
  }

  const pypi = document.querySelector('.md-social__link[href="https://pypi.org/project/py-fatigue/"]');
  if (header && pypi) {
    const link = pypi.cloneNode(true);
    link.className = "docs-pypi-link";
    link.setAttribute("aria-label", pypi.title);
    header.querySelector(".md-header__source")?.after(link);
  }

  const scrollwrap = sidebar?.querySelector(".md-sidebar__scrollwrap");
  for (const scroll of document.querySelectorAll(".md-sidebar > .md-sidebar__scrollwrap")) {
    const panel = document.createElement("div");
    panel.className = "docs-sidebar-panel";
    scroll.before(panel);
    panel.append(scroll);
  }
  if (scrollwrap && section && !home) {
    const indexHeading = document.createElement("div");
    indexHeading.className = "docs-index-heading";
    indexHeading.textContent = section.textContent.trim();
    scrollwrap.prepend(indexHeading);
  }

  if (header && sidebar && radios.length === 3) {
    const controls = [];
    function selector() {
      const group = document.createElement("div");
      group.className = "docs-theme-segments";
      group.setAttribute("role", "group");
      group.setAttribute("aria-label", "Color theme");
      for (const [label, index] of [["Light", 1], ["Dark", 2], ["System", 0]]) {
        const button = document.createElement("button");
        button.type = "button";
        button.setAttribute("aria-label", label);
        button.title = label;
        const icon = radios[index].nextElementSibling?.querySelector("svg");
        if (icon) {
          const image = icon.cloneNode(true);
          image.setAttribute("aria-hidden", "true");
          button.append(image);
        } else {
          button.textContent = label;
        }
        button.addEventListener("click", () => radios[index].click());
        group.append(button);
        controls.push([button, index]);
      }
      return group;
    }
    if (home) {
      const headerSelector = selector();
      headerSelector.classList.add("docs-theme-header");
      header.insertBefore(headerSelector, header.querySelector(".md-search"));
    }

    const utility = document.createElement("div");
    utility.className = "docs-sidebar-utility";
    utility.append(selector());
    sidebar.append(utility);

    function selectPalette() {
      const index = window.__md_get?.("__palette")?.index ?? 0;
      for (const [button, mode] of controls) {
        button.setAttribute("aria-pressed", String(mode === index));
      }
      if (favicon) {
        const scheme = document.body.dataset.mdColorScheme === "slate" ? "dark" : "light";
        const mark = new URL(`py-fatigue-logomark-${scheme}-theme.svg`, favicon.href).href;
        if (favicon.href !== mark) favicon.href = mark;
        for (const logo of logos) {
          if (logo.src !== mark) logo.src = mark;
        }
      }
    }
    new MutationObserver(selectPalette).observe(document.body, {
      attributes: true,
      attributeFilter: ["data-md-color-media", "data-md-color-scheme"],
    });
    palette.addEventListener("change", () => queueMicrotask(selectPalette));
    selectPalette();
    document.body.classList.add("docs-theme-ready");
  }

  const notebooks = document.querySelectorAll("iframe.docs-notebook");
  if (notebooks.length) {
    document.querySelector("article:has(#runnable-notebook)")?.classList.add("docs-notebook-page");
    const syncNotebookThemes = () => {
      const scheme = document.body.dataset.mdColorScheme === "slate" ? "dark" : "light";
      for (const frame of notebooks) {
        const root = frame.contentDocument?.documentElement;
        if (root) {
          root.dataset.docsTheme = scheme;
          root.style.colorScheme = scheme;
        }
      }
    };
    for (const frame of notebooks) {
      frame.addEventListener("load", () => {
        const doc = frame.contentDocument;
        if (!doc?.documentElement) return;
        syncNotebookThemes();
        doc.documentElement.classList.add("docs-embedded-notebook");
        const mounting = new MutationObserver(observeApp);
        function observeApp() {
          const app = doc.getElementById("App");
          if (!app) return;
          mounting.disconnect();
          const resize = () => {
            frame.style.height = Math.ceil(app.getBoundingClientRect().height) + "px";
          };
          const observer = new ResizeObserver(resize);
          frame.contentWindow.addEventListener("pagehide", () => observer.disconnect(), { once: true });
          observer.observe(app);
        }
        mounting.observe(doc.body, { subtree: true, childList: true });
        observeApp();
      });
    }
    new MutationObserver(syncNotebookThemes).observe(document.body, {
      attributes: true,
      attributeFilter: ["data-md-color-scheme"],
    });
    syncNotebookThemes();
  }

  const drawer = document.querySelector("#__drawer");
  const trigger = header?.querySelector('label[for="__drawer"]');
  if (drawer && trigger) {
    header.querySelector(".md-logo")?.before(trigger);
    trigger.tabIndex = 0;
    trigger.setAttribute("role", "button");
    trigger.setAttribute("aria-controls", "docs-primary-index");
    sidebar?.setAttribute("id", "docs-primary-index");
    const update = () => trigger.setAttribute("aria-expanded", String(drawer.checked));
    update();
    drawer.addEventListener("change", update);
    trigger.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      drawer.click();
      if (drawer.checked) sidebar?.querySelector("a")?.focus();
    });
    document.addEventListener("keydown", (event) => {
      if (event.key !== "Escape" || !drawer.checked) return;
      drawer.click();
      trigger.focus();
    });
  }

  const heading = document.querySelector("article.md-content__inner > h1");
  const toc = document.querySelector(".md-sidebar--secondary .md-nav--secondary .md-nav__list");
  if (heading && toc && toc.querySelectorAll("a").length > 1) {
    const disclosure = document.createElement("details");
    disclosure.className = "docs-inline-toc";
    const summary = document.createElement("summary");
    summary.textContent = "On this page";
    const list = toc.cloneNode(true);
    list.removeAttribute("data-md-component");
    disclosure.append(summary, list);
    disclosure.addEventListener("click", (event) => {
      if (event.target.closest("a")) disclosure.open = false;
    });
    heading.after(disclosure);
    document.body.classList.add("docs-inline-toc-ready");
  }

  const activeGroup = sidebar?.querySelector(".md-nav--primary > ul > .md-nav__item--active.md-nav__item--nested");
  if (scrollwrap && heading && !home && !activeGroup) {
    const index = document.createElement("nav");
    index.className = "docs-page-index";
    const changelog = heading.id === "changelog";
    index.setAttribute("aria-label", changelog ? "Major versions" : "Page sections");
    const list = document.createElement("ul");
    const majors = new Set();
    for (const topic of heading.parentElement.querySelectorAll(":scope > h2[id]")) {
      const major = topic.textContent.match(/^(\d+)\.\d+\.\d+/)?.[1];
      if (changelog && (!major || majors.has(major))) continue;
      if (major) majors.add(major);
      const item = document.createElement("li");
      const link = document.createElement("a");
      link.href = "#" + topic.id;
      link.textContent = changelog ? "Version " + major : topic.firstChild.textContent.trim();
      item.append(link);
      list.append(item);
    }
    if (list.children.length) {
      index.append(list);
      scrollwrap.append(index);
      if (changelog) index.classList.add("docs-version-index");
    }
  }

  const sectionPath = new URL(section?.href || location.href).pathname;
  for (const item of sidebar?.querySelectorAll("li.md-nav__item--nested") || []) {
    const checkbox = item.querySelector(":scope > .md-nav__toggle");
    const label = item.querySelector(":scope > label.md-nav__link, :scope > .md-nav__link > label");
    if (!checkbox || !label) continue;
    const nav = item.querySelector(":scope > nav");
    const key = "py-fatigue-docs-nav:" + sectionPath + ":" + checkbox.id;
    let storage;
    try {
      storage = window.sessionStorage;
      const saved = storage.getItem(key);
      if (saved !== null && !item.classList.contains("md-nav__item--active")) {
        checkbox.checked = saved === "open";
      }
    } catch {
      storage = null;
    }
    function updateDisclosure() {
      const expanded = String(checkbox.checked);
      label.setAttribute("aria-expanded", expanded);
      nav?.setAttribute("aria-expanded", expanded);
    }
    label.tabIndex = 0;
    label.setAttribute("role", "button");
    if (nav) {
      nav.id = checkbox.id + "-panel";
      label.setAttribute("aria-controls", nav.id);
    }
    // Zensical already handles Enter on navigation labels.
    label.addEventListener("keydown", (event) => {
      if (event.key !== " ") return;
      event.preventDefault();
      checkbox.click();
    });
    checkbox.addEventListener("change", () => {
      updateDisclosure();
      try {
        storage?.setItem(key, checkbox.checked ? "open" : "closed");
      } catch {
        // Disclosure also works when storage is full.
      }
    });
    updateDisclosure();
  }
})();

for (const group of document.querySelectorAll(".tabbed-set")) {
  const labels = group.querySelector(":scope > .tabbed-labels");
  const inputs = [...group.querySelectorAll(":scope > input[type=radio]")];
  const panels = [...group.querySelectorAll(":scope > .tabbed-content > .tabbed-block")];
  if (!labels || inputs.length < 2 || inputs.length !== panels.length) continue;

  const strip = document.createElement("div");
  strip.className = "dashboard-tabbar dashboard-tabbar--ruled";
  const tabs = document.createElement("nav");
  tabs.className = "tabs dashboard-tabs";
  tabs.setAttribute("role", "tablist");
  tabs.setAttribute("aria-label", "Example options");
  strip.append(tabs);
  const parameter = inputs[0].id + "-tab";

  const buttons = inputs.map((input, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "tab";
    button.id = input.id + "-label";
    button.setAttribute("role", "tab");
    const label = document.createElement("span");
    label.className = "dashboard-tab-label";
    label.textContent = labels.querySelector('label[for="' + input.id + '"]').textContent;
    const panel = panels[index];
    panel.id ||= input.id + "-panel";
    panel.setAttribute("role", "tabpanel");
    panel.setAttribute("aria-labelledby", button.id);
    panel.tabIndex = 0;
    button.setAttribute("aria-controls", panel.id);
    button.append(label);
    tabs.append(button);
    return button;
  });

  function show(index, push = false) {
    const url = new URL(location.href);
    inputs[index].checked = true;
    for (const [position, button] of buttons.entries()) {
      const active = position === index;
      button.setAttribute("aria-selected", String(active));
      button.tabIndex = active ? 0 : -1;
      panels[position].hidden = !active;
    }
    const bounds = buttons[index].getBoundingClientRect();
    const viewport = tabs.getBoundingClientRect();
    tabs.scrollLeft += bounds.left < viewport.left
      ? bounds.left - viewport.left
      : Math.max(0, bounds.right - viewport.right);
    if (push) {
      url.searchParams.set(parameter, inputs[index].id);
      history.pushState(null, "", url);
    }
  }
  function readLocation() {
    const selected = new URL(location.href).searchParams.get(parameter);
    show(Math.max(0, inputs.findIndex((input) => input.id === selected)));
  }

  buttons.forEach((button, index) => {
    button.addEventListener("click", () => {
      if (!inputs[index].checked) show(index, true);
    });
    button.addEventListener("keydown", (event) => {
      const last = buttons.length - 1;
      const next = {
        ArrowLeft: (index + last) % buttons.length,
        ArrowRight: (index + 1) % buttons.length,
        Home: 0,
        End: last,
      }[event.key];
      if (next === undefined) return;
      event.preventDefault();
      if (index !== next) show(next, true);
      buttons[next].focus({ preventScroll: true });
    });
  });
  labels.hidden = true;
  labels.before(strip);
  group.classList.add("docs-ruled-tabs");
  window.addEventListener("popstate", readLocation);
  readLocation();
}
