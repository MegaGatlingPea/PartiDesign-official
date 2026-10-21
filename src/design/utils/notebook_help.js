const helpCounter = Symbol.for('partidesign.notebook.helpCounter');

function render({model, el}) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'pd-help-button';
  button.setAttribute('aria-label', 'Help');
  button.setAttribute('aria-expanded', 'false');
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('width', '19');
  svg.setAttribute('height', '19');
  svg.setAttribute('aria-hidden', 'true');
  for (const [tag, attrs] of [
    ['circle', {cx: '12', cy: '12', r: '9', fill: 'none', stroke: 'currentColor', 'stroke-width': '1.7'}],
    ['path', {d: 'M9.5 9a2.5 2.5 0 0 1 5 0c0 2-2.5 2-2.5 4', fill: 'none', stroke: 'currentColor', 'stroke-width': '1.7', 'stroke-linecap': 'round'}],
    ['circle', {cx: '12', cy: '16.4', r: '1', fill: 'currentColor'}],
  ]) {
    const child = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [name, value] of Object.entries(attrs)) child.setAttribute(name, value);
    svg.appendChild(child);
  }
  button.appendChild(svg);
  el.appendChild(button);
  el.classList.add('pd-help');
  const tooltip = document.createElement('div');
  document[helpCounter] = (document[helpCounter] || 0) + 1;
  tooltip.id = `pd-help-${document[helpCounter]}`;
  tooltip.className = 'pd-help-tooltip';
  tooltip.setAttribute('role', 'tooltip');
  tooltip.hidden = true;
  Object.assign(tooltip.style, {
    pointerEvents: 'none', position: 'fixed', zIndex: '100000', boxSizing: 'border-box',
    maxWidth: 'min(340px, calc(100vw - 24px))', maxHeight: 'calc(100vh - 24px)',
    overflowY: 'auto', padding: '10px 12px',
    border: '1px solid #d8dee9', borderRadius: '8px',
    color: '#3b4252', background: '#fff', boxShadow: '0 4px 18px #2e344026',
    font: '13px/1.5 system-ui, sans-serif', whiteSpace: 'normal', overflowWrap: 'anywhere',
  });
  document.body.appendChild(tooltip);
  let pinned = false, hovering = false, focused = false, suppressed = false, disposed = false;
  const position = () => {
    if (tooltip.hidden || disposed) return;
    const anchor = button.getBoundingClientRect(), box = tooltip.getBoundingClientRect();
    const left = Math.min(window.innerWidth - box.width - 12, Math.max(12, anchor.left + anchor.width / 2 - box.width / 2));
    const below = anchor.bottom + 8;
    const top = below + box.height <= window.innerHeight - 12
      ? below : Math.max(12, anchor.top - box.height - 8);
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
  };
  const update = () => {
    // Use the controllable portal instead of a second native title popup.
    el.removeAttribute('title');
    const text = model.get('help_text') || '';
    tooltip.textContent = text;
    const shown = Boolean(text && !suppressed && (pinned || hovering || focused));
    tooltip.hidden = !shown;
    button.setAttribute('aria-expanded', String(shown));
    button.setAttribute('aria-pressed', String(pinned));
    if (shown) { button.setAttribute('aria-describedby', tooltip.id); position(); }
    else button.removeAttribute('aria-describedby');
  };
  const enter = () => { hovering = true; suppressed = false; update(); };
  const leave = () => { hovering = false; update(); };
  const focus = () => { focused = true; suppressed = false; update(); };
  const blur = () => { focused = false; update(); };
  const click = () => { pinned = !pinned; suppressed = !pinned; update(); };
  const outside = event => {
    if (!el.contains(event.target) && !tooltip.contains(event.target)) {
      pinned = false; suppressed = true; update();
    }
  };
  const keyboard = event => {
    if (event.key === 'Escape') { pinned = false; suppressed = true; update(); }
  };
  button.addEventListener('pointerenter', enter);
  button.addEventListener('pointerleave', leave);
  button.addEventListener('focus', focus);
  button.addEventListener('blur', blur);
  button.addEventListener('click', click);
  document.addEventListener('pointerdown', outside, true);
  document.addEventListener('keydown', keyboard, true);
  document.addEventListener('scroll', position, true);
  window.addEventListener('resize', position);
  model.on('change:help_text', update);
  update();
  return () => {
    disposed = true;
    model.off('change:help_text', update);
    document.removeEventListener('pointerdown', outside, true);
    document.removeEventListener('keydown', keyboard, true);
    document.removeEventListener('scroll', position, true);
    window.removeEventListener('resize', position);
    button.removeEventListener('pointerenter', enter);
    button.removeEventListener('pointerleave', leave);
    button.removeEventListener('focus', focus);
    button.removeEventListener('blur', blur);
    button.removeEventListener('click', click);
    tooltip.remove();
    button.remove();
  };
}
export default {render};
