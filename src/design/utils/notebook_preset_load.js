function render({model, el}) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'jupyter-button mod-primary pd-preset-load-button';
  button.textContent = 'Load preset';
  el.appendChild(button);
  let serial = 0;
  const acknowledge = () => {
    button.disabled = false;
    button.textContent = 'Load preset';
  };
  const load = () => {
    // Carry the visible selection in the click message. A remote dropdown's
    // state update can otherwise arrive after a separate Button click.
    const panel = el.closest('.pd-task-preset');
    const select = panel?.querySelectorAll('select')[1];
    const entry = (model.get('entries') || []).find(item => item.label === select?.value);
    if (!entry) return;
    button.disabled = true;
    button.textContent = 'Loading…';
    model.set('request', {key: entry.key, id: `${Date.now()}-${++serial}`});
    model.save_changes();
  };
  button.addEventListener('click', load);
  model.on('change:ack', acknowledge);
  return () => {
    button.removeEventListener('click', load);
    model.off('change:ack', acknowledge);
    button.remove();
  };
}
export default {render};
