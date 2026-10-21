// Selection widgets synchronize indices, while scalar widgets synchronize values.
export function snapshotBindings(bindings) {
  return Object.fromEntries(Object.entries(bindings).map(([key, control]) => {
    const labels = control.get('_options_labels');
    const selection = Array.isArray(labels) && control.get('index') !== undefined;
    return [key, {
      model_id: control.model_id,
      kind: selection ? 'index' : 'value',
      value: structuredClone(control.get(selection ? 'index' : 'value')),
      ...(selection ? {labels: [...labels]} : {}),
    }];
  }));
}

export async function resolveBindings(model) {
  return Object.fromEntries(await Promise.all(
    Object.entries(model.get('bindings') || {}).map(async ([key, reference]) => {
      if (reference && typeof reference.get === 'function') return [key, reference];
      if (typeof reference !== 'string' || !reference.startsWith('IPY_MODEL_')) {
        throw new Error('The action controls are unavailable. Reload this task.');
      }
      const control = await model.widget_manager.get_model(reference.slice(10));
      if (!control) throw new Error('The action controls are not ready. Reload this task.');
      return [key, control];
    }),
  ));
}

function render({model, el}) {
  const button = document.createElement('button');
  button.type = 'button';
  button.style.width = '100%';
  el.appendChild(button);
  let bindings = null;
  let revision = 0;
  let pending = null;
  let serial = 0;
  let disposed = false;
  const update = () => {
    button.textContent = model.get('description') || '';
    button.className = 'jupyter-button widget-button' +
      (model.get('button_style') ? ` mod-${model.get('button_style')}` : '');
    button.disabled = Boolean(model.get('disabled') || !bindings || pending);
    button.setAttribute('aria-busy', String(Boolean(pending)));
    button.title = model.get('error') || '';
  };
  const loadBindings = async () => {
    const current = ++revision;
    bindings = null;
    update();
    try {
      const resolved = await resolveBindings(model);
      if (disposed || current !== revision) return;
      bindings = resolved;
      update();
    } catch (error) {
      if (!disposed && current === revision) {
        button.title = error.message;
      }
    }
  };
  const acknowledge = () => {
    if (pending === model.get('ack')) pending = null;
    update();
  };
  const click = () => {
    if (button.disabled || !bindings) return;
    // Capture everything before sending; no separate comm update is relied on.
    const values = snapshotBindings(bindings);
    pending = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${++serial}-${Math.random()}`;
    model.set('request', {id: pending, values});
    model.save_changes();
    update();
  };
  button.addEventListener('click', click);
  model.on('change:description change:button_style change:disabled change:error', update);
  model.on('change:bindings', loadBindings);
  model.on('change:ack', acknowledge);
  loadBindings();
  return () => {
    disposed = true;
    button.removeEventListener('click', click);
    model.off('change:description change:button_style change:disabled change:error', update);
    model.off('change:bindings', loadBindings);
    model.off('change:ack', acknowledge);
    button.remove();
  };
}
export default {render};
