(() => {
  const header = document.querySelector(".md-header__inner");
  const legacyArticle = document.querySelector("article.bd-article");
  if (!header && !legacyArticle) return;

  const root = new URL(".", document.currentScript.src);
  const directory = location.pathname.slice(root.pathname.length).split("/")[0];

  fetch(new URL("versions.json", root), { cache: "no-cache" })
    .then((response) => response.ok ? response.json() : null)
    .then((catalog) => {
      if (!catalog?.versions.length) return;

      const picker = document.createElement("select");
      picker.setAttribute("aria-label", "Documentation version");
      picker.className =
        "docs-version-picker form-select form-select-sm w-auto mb-3";
      for (const version of catalog.versions) {
        const label = version === catalog.stable ? `${version} (stable)` : version;
        picker.add(new Option(label, version));
      }

      if (catalog.versions.includes(directory)) {
        picker.value = directory;
      } else if (catalog.stable) {
        picker.value = catalog.stable;
      } else {
        picker.prepend(new Option("Versions", "", true, true));
        picker.options[0].disabled = true;
      }
      picker.addEventListener("change", () => {
        location.assign(new URL(`${picker.value}/`, root));
      });
      if (header) {
        header.append(picker);
      } else {
        legacyArticle.prepend(picker);
      }
    })
    .catch(() => {});
})();
