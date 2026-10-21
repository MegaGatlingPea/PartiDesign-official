function render({model, el}) {
  const input = document.createElement('textarea');
  input.className = 'widget-input';
  input.setAttribute('aria-label', 'Prepared command');
  input.rows = 5;
  input.spellcheck = false;
  const value = () => {
    if (input.value !== model.get('value')) input.value = model.get('value');
  };
  const editable = () => {
    input.readOnly = !model.get('editable');
    input.setAttribute('aria-readonly', String(input.readOnly));
  };
  const update = () => {
    if (input.readOnly) return;
    model.set('value', input.value);
    model.save_changes();
  };
  value(); editable();
  input.addEventListener('input', update);
  model.on('change:value', value);
  model.on('change:editable', editable);
  el.appendChild(input);
  return () => {
    model.off('change:value', value);
    model.off('change:editable', editable);
    input.removeEventListener('input', update);
    input.remove();
  };
}
export default {render};
