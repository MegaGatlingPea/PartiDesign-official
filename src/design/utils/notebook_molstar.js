/*
Adapted from ipymolstar 0.1.0 (https://github.com/molstar/ipymolstar).
MIT License
Copyright (c) 2024 Jochem Smit

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
import "https://cdn.jsdelivr.net/npm/pdbe-molstar@3.3.2/build/pdbe-molstar-plugin.js";

function standardize_color(str) {
  var ctx = document.createElement("canvas").getContext("2d");
  ctx.fillStyle = str;
  return ctx.fillStyle;
}
function toRgb(color) {
  var hex = standardize_color(color);
  var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16),
      }
    : null;
}

function getHideStructure(model) {
  var hideStructure = [];

  if (model.get("hide_polymer")) {
    hideStructure.push("polymer");
  }
  if (model.get("hide_water")) {
    hideStructure.push("water");
  }
  if (model.get("hide_heteroatoms")) {
    hideStructure.push("het");
  }
  if (model.get("hide_carbs")) {
    hideStructure.push("carbs");
  }
  if (model.get("hide_non_standard")) {
    hideStructure.push("nonStandard");
  }
  if (model.get("hide_coarse")) {
    hideStructure.push("coarse");
  }

  return hideStructure;
}

function getVisibility(model) {
  var visibility = {
    polymer: !model.get("hide_polymer"),
    het: !model.get("hide_heteroatoms"),
    water: !model.get("hide_water"),
    carbs: !model.get("hide_carbs"),
    // maps ?
  };

  return visibility;
}

function getHideCanvasControls(model) {
  var hideCanvasControls = [];
  if (model.get("hide_controls_icon")) {
    hideCanvasControls.push("controlToggle");
  }
  if (model.get("hide_expand_icon")) {
    hideCanvasControls.push("expand");
  }
  if (model.get("hide_settings_icon")) {
    hideCanvasControls.push("controlInfo");
  }
  if (model.get("hide_selection_icon")) {
    hideCanvasControls.push("selection");
  }
  if (model.get("hide_animation_icon")) {
    hideCanvasControls.push("animation");
  }

  return hideCanvasControls;
}

function getCustomData(model) { return null; }

const EmptyFocusBindings = {
  clickCenterFocus: { triggers: [], action: '', description: '' },
  clickCenterFocusSelectMode: { triggers: [], action: '', description: '' },
  clickResetCameraOnEmpty: { triggers: [], action: '', description: '' },
  clickResetCameraOnEmptySelectMode: { triggers: [], action: '', description: '' }
};


function getOptions(model) {
  var options = {
    moleculeId: model.get("molecule_id"),
    customData: getCustomData(model),
    assemblyId: model.get("assembly_id"),
    defaultPreset: model.get("default_preset"),
    ligandView: model.get("ligand_view"),
    alphafoldView: model.get("alphafold_view"),
    superposition: model.get("superposition"),
    superpositionParams: model.get("superposition_params"),
    visualStyle: model.get("visual_style"),
    loadMaps: model.get("load_maps"),
    bgColor: toRgb(model.get("bg_color")),
    selectColor: toRgb(model.get("select_color")),
    lighting: model.get("lighting"),
    validationAnnotation: model.get("validation_annotation"),
    symmetryAnnotation: model.get("symmetry_annotation"),
    pdbeUrl: model.get("pdbe_url"),
    encoding: model.get("encoding"),
    lowPrecisionCoords: model.get("low_precision_coords"),
    selectInteraction: model.get("select_interaction"),
    // selectBindings: EmptySelectBindings,
    granularity: model.get("granularity"),
    subscribeEvents: model.get("subscribe_events"),
    hideControls: model.get("hide_controls"),
    hideCanvasControls: getHideCanvasControls(model),
    sequencePanel: model.get("sequence_panel"),
    pdbeLink: model.get("pdbe_link"),
    loadingOverlay: model.get("loading_overlay"),
    expanded: model.get("expanded"),
    landscape: model.get("landscape"),
    reactive: model.get("reactive"),
  };

  if (model.get('click_focus') == false) {
      options.focusBindings = EmptyFocusBindings;
  }

  return options;
}

// Molstar 4.8 OrderedSet is either a sorted index array or an IntTuple
// interval (the two Int32 endpoints share a Float64). Match the pinned viewer.
function orderedIndices(indices) {
  if (typeof indices !== 'number') return indices;
  const value = new Float64Array([indices]);
  const pair = new Int32Array(value.buffer);
  return Array.from({length: Math.max(0, pair[1] - pair[0])}, (_, i) => pair[0] + i);
}

function atomTarget(unit, index) {
  if (unit.kind !== 0 || index == null) return null;
  const element = unit.elements[index];
  const h = unit.model.atomicHierarchy;
  const residue = h.residueAtomSegments.index[element];
  const chain = h.chainAtomSegments.index[element];
  return {
    label_asym_id: h.chains.label_asym_id.value(chain),
    seq_id: h.residues.label_seq_id.value(residue),
    auth_asym_id: h.chains.auth_asym_id.value(chain),
    auth_seq_id: h.residues.auth_seq_id.value(residue),
    ins_code: h.residues.pdbx_PDB_ins_code.value(residue),
    comp_id: h.atoms.label_comp_id.value(element),
    atom_id: [h.atoms.label_atom_id.value(element)],
  };
}

/** Read the pinned Molstar native loci, including both sides of inter-residue bonds. */
export function normalizeInteractionLoci(loci, requestedScope = 'residue') {
  const scope = ['chain', 'residue', 'atom'].includes(requestedScope) ? requestedScope : 'residue';
  const seeds = [], seedIndices = new Map();
  const add = (unit, index) => {
    if (unit?.kind !== 0 || !Number.isInteger(index) || index < 0 || index >= unit.elements.length) return;
    if (!seedIndices.has(unit)) seedIndices.set(unit, new Set());
    if (!seedIndices.get(unit).has(index)) { seedIndices.get(unit).add(index); seeds.push({unit, index}); }
  };
  if (loci?.kind === 'bond-loci') {
    for (const bond of loci.bonds) {
      add(bond.aUnit, bond.aIndex);
      add(bond.bUnit, bond.bIndex);
    }
  } else if (loci?.kind === 'element-loci') {
    for (const item of loci.elements) {
      for (const index of orderedIndices(item.indices)) add(item.unit, index);
    }
  }
  if (!seeds.length) return null;
  const chosen = new Map();
  const select = (unit, index) => {
    if (!chosen.has(unit)) chosen.set(unit, new Set());
    chosen.get(unit).add(index);
  };
  const expanded = new Map();
  for (const seed of seeds) {
    if (scope === 'atom') { select(seed.unit, seed.index); continue; }
    const unit = seed.unit, h = unit.model.atomicHierarchy;
    const segment = scope === 'chain' ? h.chainAtomSegments : h.residueAtomSegments;
    const segmentId = segment.index[unit.elements[seed.index]];
    if (!expanded.has(unit.model)) expanded.set(unit.model, new Set());
    const expansionKey = `${unit.conformation.operator.name}:${segmentId}`;
    if (expanded.get(unit.model).has(expansionKey)) continue;
    expanded.get(unit.model).add(expansionKey);
    // Separate symmetry copies and models while including a chain split across units.
    for (const candidate of loci.structure.units) {
      if (candidate.kind !== 0 || candidate.model !== unit.model
          || candidate.conformation.operator.name !== unit.conformation.operator.name) continue;
      for (let index = 0; index < candidate.elements.length; index++) {
        if (segment.index[candidate.elements[index]] === segmentId) select(candidate, index);
      }
    }
  }
  const targets = [], residues = new Map(), seenAtoms = new Set(), elements = [];
  for (const [unit, indices] of chosen) {
    const sorted = [...indices].sort((a, b) => a - b);
    elements.push({unit, indices: Int32Array.from(sorted)});
    for (const index of sorted) {
      const target = atomTarget(unit, index);
      const key = JSON.stringify([target.label_asym_id, target.seq_id, target.auth_asym_id,
        target.auth_seq_id, target.ins_code, target.comp_id]);
      if (scope === 'atom') {
        const atomKey = key + ':' + target.atom_id[0];
        if (!seenAtoms.has(atomKey)) { seenAtoms.add(atomKey); targets.push(target); }
      } else if (residues.has(key)) {
        const atoms = residues.get(key).atom_id;
        if (!atoms.includes(target.atom_id[0])) atoms.push(target.atom_id[0]);
      } else { residues.set(key, target); targets.push(target); }
    }
  }
  return {
    event: {scope, targets, kind: scope === 'atom' ? (loci.kind === 'bond-loci' ? 'bond' : 'atom') : scope},
    loci: {kind: 'element-loci', structure: loci.structure, elements},
  };
}

// Keep one viewer instance per widget, and apply visual changes only after loading.
async function render({ model, el }) {
  const container = document.createElement('div');
  container.dataset.partidesignViewer = '';
  Object.assign(container.style, {position: 'relative', height: model.get('height'), width: model.get('width'), maxWidth: '100%', boxSizing: 'border-box'});
  el.appendChild(container);
  const viewer = new window.PDBeMolstarPlugin();
  let disposed = false, ready = false, initialized = false, pendingData = false;
  let ownsPointer = false, updating = Promise.resolve();
  const subscriptions = [];
  const objectUrls = new Set();
  const savedTypes = new Map();
  let scopeUpdates = Promise.resolve(), nativeClick = null, clickSerial = 0;
  const toolbar = document.createElement('div');
  toolbar.className = 'pd-scope-toolbar';
  toolbar.setAttribute('role', 'group');
  toolbar.setAttribute('aria-label', 'Select scope');
  const label = document.createElement('span');
  label.textContent = 'Select scope';
  toolbar.appendChild(label);
  const scopeButtons = new Map();
  for (const [value, text] of [['chain', 'Chain'], ['residue', 'Residue'], ['atom', 'Atom']]) {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = text;
    button.dataset.scope = value;
    button.title = value === 'atom' ? 'Select atoms or both endpoints of a bond' : `Select a whole ${value}`;
    button.addEventListener('click', () => {
      model.set('selection_scope', value);
      model.save_changes();
      applyScope();
    });
    scopeButtons.set(value, button);
    toolbar.appendChild(button);
  }
  // Mount outside PDBe's React root so subsequent viewer renders retain the controls.
  el.style.position = 'relative';
  el.appendChild(toolbar);
  const mountToolbar = () => {
    const header = el.closest('.pd-viewer')?.querySelector('.pd-view-header');
    if (header && toolbar.parentElement !== header) header.appendChild(toolbar);
  };
  mountToolbar();
  const currentScope = () => ['chain', 'residue', 'atom'].includes(model.get('selection_scope'))
    ? model.get('selection_scope') : 'residue';
  const scopeVisual = () => {
    scopeUpdates = scopeUpdates.then(async () => {
      if (disposed || !ready || !viewer.plugin?.managers) return;
      const plugin = viewer.plugin, atom = currentScope() === 'atom';
      const provider = plugin.representation.structure.registry.get('ball-and-stick');
      for (const structure of plugin.managers.structure.hierarchy.current.structures) {
        for (const component of structure.components) {
          for (const repr of component.representations) {
            const ref = repr.cell.transform.ref, params = repr.cell.params?.values;
            if (!params?.type) continue;
            if (atom && !savedTypes.has(ref)) savedTypes.set(ref, structuredClone(params.type));
            const type = atom ? {name: 'ball-and-stick', params: structuredClone(provider.defaultValues)} : savedTypes.get(ref);
            if (!type) continue;
            if (params.type.name !== type.name || !atom) {
              await plugin.managers.structure.component.updateRepresentations([component], repr, {...params, type});
            }
            if (!atom) savedTypes.delete(ref);
          }
        }
      }
    }).catch(report);
  };
  const applyScope = () => {
    const scope = currentScope();
    for (const [value, button] of scopeButtons) button.setAttribute('aria-pressed', String(value === scope));
    if (!viewer.plugin?.managers?.interactivity) return;
    viewer.plugin.managers.interactivity.setProps({granularity: scope === 'atom' ? 'element' : scope});
    scopeVisual();
  };
  const options = () => {
    const opts = getOptions(model);
    opts.rightPanel = false;
    const data = model.get('custom_data');
    if (data?.data != null) {
      const url = URL.createObjectURL(new Blob([data.data]));
      objectUrls.add(url);
      opts.customData = {...data, url};
      delete opts.customData.data;
    }
    return opts;
  };
  const report = error => {
    if (!disposed) {
      model.set('viewer_error', String(error?.message || error));
      model.save_changes();
    }
  };
  const visual = () => {
    if (disposed || !ready || !viewer.plugin?.managers) return;
    const colors = model.get('color_data');
    const highlight = model.get('highlight');
    if (colors != null) viewer.visual.select(colors);
    if (highlight != null) viewer.visual.highlight(highlight);
  };
  const loaded = viewer.events.loadComplete.subscribe(() => {
    if (disposed) return;
    ready = true;
    applyScope();
    visual();
  });
  const on = (name, callback) => {
    model.on(name, callback);
    subscriptions.push(() => model.off(name, callback));
  };
  const update = () => {
    if (!initialized) { pendingData = true; return; }
    updating = updating.then(async () => {
      if (disposed) return;
      ready = false;
      savedTypes.clear();
      if (model.get('custom_data')?.data == null && !model.get('molecule_id')) {
        await viewer.plugin.clear();
      } else {
        await viewer.visual.update(options(), true);
      }
      if (disposed) return;
      ready = true;
      applyScope();
      visual();
    }).catch(report);
  };
  // Changing the display theme must not reload coordinates or reset the camera.
  const applyBackground = () => {
    if (disposed || !viewer.plugin?.canvas3d) return;
    viewer.canvas.setBgColor(toRgb(model.get('bg_color'))).catch(report);
  };
  on('change:bg_color', applyBackground);
  on('change:custom_data', update);
  on('change:molecule_id', update);
  on('change:visual_style', update);
  on('change:color_data', visual);
  on('change:highlight', visual);
  on('change:selection_scope', applyScope);
  const pointer = event => { ownsPointer = container.contains(event.target); };
  document.addEventListener('pointerdown', pointer, true);
  const click = event => {
    // Native events preserve bond endpoints; legacy PDBe is a fallback only.
    if (nativeClick || disposed || !ownsPointer || !container.isConnected || !event.eventData) return;
    model.set('click_event', event.eventData);
    model.save_changes();
  };
  document.addEventListener('PDB.molstar.click', click);
  try {
    const initialOptions = options();
    const empty = !initialOptions.moleculeId && !initialOptions.customData;
    const sourceUrl = viewer.getMoleculeSrcUrl;
    if (empty) {
      // PDBe validates a source before mounting. Defer source lookup so an
      // ordinary empty canvas mounts without downloads or dummy coordinates.
      initialOptions.moleculeId = '__partidesign_empty__';
      viewer.getMoleculeSrcUrl = () => undefined;
    }
    try {
      await viewer.render(container, initialOptions);
    } finally {
      viewer.getMoleculeSrcUrl = sourceUrl;
      if (empty && viewer.initParams) viewer.initParams.moleculeId = undefined;
    }
    initialized = true;
    ready = true;
    nativeClick = viewer.plugin.behaviors.interaction.click.subscribe(event => {
      if (disposed || event.button !== 1 || !container.isConnected) return;
      const normalized = normalizeInteractionLoci(event.current?.loci, currentScope());
      if (!normalized) return;
      viewer.plugin.managers.interactivity.lociHighlights.highlightOnly({loci: normalized.loci}, false);
      model.set('click_event', {...normalized.event, event_id: ++clickSerial});
      model.save_changes();
    });
    mountToolbar();
    applyScope();
    applyBackground();
    visual();
    if (pendingData) update();
  } catch (error) {
    report(error);
  }
  return () => {
    disposed = true;
    ready = false;
    subscriptions.forEach(off => off());
    loaded.unsubscribe();
    nativeClick?.unsubscribe();
    toolbar.remove();
    savedTypes.clear();
    document.removeEventListener('pointerdown', pointer, true);
    document.removeEventListener('PDB.molstar.click', click);
    viewer.plugin?.dispose();
    objectUrls.forEach(url => URL.revokeObjectURL(url));
    container.remove();
  };
}
export default {render};
