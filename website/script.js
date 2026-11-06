// One explicit theme choice for desktop and mobile; OS preferences do not override it.
const themeButtons = [...document.querySelectorAll("[data-theme-toggle]")];
function applyTheme(dark, persist = false) {
  document.documentElement.dataset.theme = dark ? "partidesign-night" : "partidesign";
  document.documentElement.style.colorScheme = dark ? "dark" : "light";
  document.querySelector('meta[name="theme-color"]').content = dark ? "#0c1120" : "#ffffff";
  themeButtons.forEach((button) => {
    const label = dark ? "Switch to light mode" : "Switch to dark mode";
    button.setAttribute("aria-label", label);
    button.setAttribute("title", label);
    button.setAttribute("aria-pressed", String(dark));
    button.classList.toggle("swap-active", dark);
    button.hidden = false;
  });
  if (persist) {
    try { localStorage.setItem("partidesign-theme", dark ? "dark" : "light"); } catch {}
  }
}
applyTheme(document.documentElement.dataset.theme === "partidesign-night");
themeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    applyTheme(document.documentElement.dataset.theme !== "partidesign-night", true);
  });
});
window.addEventListener("storage", (event) => {
  if (event.key === "partidesign-theme" || event.key === null) applyTheme(event.newValue === "dark");
});

// Keyboard-accessible example tabs.
document.querySelectorAll("[data-tabs]").forEach((group) => {
  const tabs = [...group.querySelectorAll('[role="tab"]')];
  const activate = (selected, focus = false) => {
    tabs.forEach((tab) => {
      const active = tab === selected;
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
      tab.classList.toggle("tab-active", active);
      document.getElementById(tab.getAttribute("aria-controls")).hidden = !active;
    });
    if (focus) selected.focus();
  };
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => activate(tab));
    tab.addEventListener("keydown", (event) => {
      let next;
      if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
      if (event.key === "ArrowLeft") next = (index - 1 + tabs.length) % tabs.length;
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = tabs.length - 1;
      if (next === undefined) return;
      event.preventDefault();
      activate(tabs[next], true);
    });
  });
});

// Copy exact text, preserving line breaks in shell commands.
const copyStatus = document.getElementById("copy-status");
const copyTimers = new WeakMap();
async function copyText(value) {
  if (navigator.clipboard && window.isSecureContext) {
    try {
      await navigator.clipboard.writeText(value);
      return;
    } catch {
      // A local preview or browser policy may require the selection fallback.
    }
  }
  const previousFocus = document.activeElement;
  const selection = window.getSelection();
  const ranges = selection ? [...Array(selection.rangeCount)].map((_, i) => selection.getRangeAt(i).cloneRange()) : [];
  const input = document.createElement("textarea");
  input.value = value;
  input.setAttribute("readonly", "");
  input.className = "fixed -left-[9999px] top-0";
  document.body.append(input);
  input.select();
  const copied = document.execCommand("copy");
  input.remove();
  previousFocus?.focus({ preventScroll: true });
  if (selection) {
    selection.removeAllRanges();
    ranges.forEach((range) => selection.addRange(range));
  }
  if (!copied) throw new Error("Clipboard is unavailable");
}
document.querySelectorAll("[data-copy]").forEach((button) => {
  const label = button.querySelector("[data-copy-label]");
  const original = label.textContent;
  button.addEventListener("click", async () => {
    const source = document.getElementById(button.dataset.copy);
    if (!source) return;
    try {
      await copyText(source.textContent.trim());
      clearTimeout(copyTimers.get(button));
      label.textContent = "Copied";
      copyStatus.textContent = button.dataset.copyName + " copied.";
      copyTimers.set(button, setTimeout(() => { label.textContent = original; }, 2200));
    } catch {
      copyStatus.textContent = "Copy is unavailable. Select the command or prompt text to copy it manually.";
    }
  });
});

// Native dialog focus handling, with a scrollable enlarged image on small screens.
const figureDialog = document.getElementById("figure-dialog");
const figureImage = document.getElementById("figure-image");
const figureTitle = document.getElementById("figure-title");
const figureOriginal = document.getElementById("figure-original");
const figureViewport = document.getElementById("figure-viewport");
const fitButton = document.getElementById("figure-fit");
let activeFigureLink;
let fitFigure = false;
function updateFigureFit() {
  figureImage.style.minWidth = fitFigure ? "0" : Math.min(Number(figureImage.getAttribute("width")), 1600) + "px";
  fitButton.setAttribute("aria-pressed", String(fitFigure));
  fitButton.textContent = fitFigure ? "Enlarge labels" : "Fit to screen";
}
document.querySelectorAll("[data-figure]").forEach((link) => {
  link.addEventListener("click", (event) => {
    if (typeof figureDialog.showModal !== "function" || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    activeFigureLink = link;
    figureTitle.textContent = link.dataset.title;
    figureImage.src = link.href;
    figureImage.alt = link.closest("figure").querySelector("img").alt;
    figureImage.width = Number(link.dataset.width);
    figureImage.height = Number(link.dataset.height);
    figureOriginal.href = link.href;
    fitFigure = false;
    updateFigureFit();
    figureDialog.showModal();
    figureViewport.scrollTo(0, 0);
  });
});
fitButton.addEventListener("click", () => {
  fitFigure = !fitFigure;
  updateFigureFit();
});
figureDialog.addEventListener("close", () => {
  activeFigureLink?.focus({ preventScroll: true });
});

// Close the compact navigation after a choice, an outside click or Escape.
const mobileNav = document.querySelector("[data-mobile-nav]");
mobileNav.querySelectorAll("a").forEach((link) => {
  link.addEventListener("click", () => { mobileNav.open = false; });
});
document.addEventListener("click", (event) => {
  if (!mobileNav.contains(event.target)) mobileNav.open = false;
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && mobileNav.open) {
    mobileNav.open = false;
    mobileNav.querySelector("summary").focus();
  }
});
