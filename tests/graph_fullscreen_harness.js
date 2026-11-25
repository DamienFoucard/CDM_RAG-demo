// Exercise the graph page's actual fullscreen event handlers without a browser dependency.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const html = fs.readFileSync(process.argv[2], 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)?.[1];
assert.ok(script, 'graph page should contain its script');
assert.match(html, /id="fullscreen-toggle"[^>]*aria-pressed="false"/, 'control begins unpressed');
const end = script.lastIndexOf('    initialize();');
assert.ok(end > 0, 'find the page initialization call');

class ClassList {
  constructor() { this.values = new Set(); }
  contains(name) { return this.values.has(name); }
  toggle(name, force) {
    const enabled = force === undefined ? !this.values.has(name) : Boolean(force);
    if (enabled) this.values.add(name); else this.values.delete(name);
    return enabled;
  }
}
class Target {
  constructor() { this.listeners = new Map(); }
  addEventListener(type, listener) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(listener);
  }
  async dispatch(type, event = {}) {
    for (const listener of this.listeners.get(type) || []) await listener(event);
  }
}
class Element extends Target {
  constructor() { super(); this.classList = new ClassList(); this.attributes = {}; }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  getAttribute(name) { return this.attributes[name]; }
}
const elements = new Map();
const document = new Target();
document.getElementById = id => {
  if (!elements.has(id)) elements.set(id, new Element());
  return elements.get(id);
};
document.body = new Element();
document.fullscreenEnabled = true;
document.fullscreenElement = null;

const exercise = `
  (async()=>{
    const workspace=$('graph-workspace'),button=$('fullscreen-toggle');
    const expectButton=active=>{
      assert.equal(button.getAttribute('aria-pressed'),String(active));
      assert.equal(button.getAttribute('aria-label'),active?'Exit full screen':'Full screen');
      assert.equal(button.textContent,active?'Exit full screen':'Full screen');
      assert.equal(button.disabled,false);
    };
    workspace.requestFullscreen=async()=>{
      document.fullscreenElement=workspace;
      await document.dispatch('fullscreenchange');
    };
    document.exitFullscreen=async()=>{
      document.fullscreenElement=null;
      await document.dispatch('fullscreenchange');
    };
    await button.dispatch('click');
    assert.equal(document.fullscreenElement,workspace);
    assert.equal(workspace.classList.contains('is-fullscreen-fallback'),false);
    expectButton(true);
    // Native Escape should be left to the browser and must not discard selection.
    state.undockedAttribute={entity_id:'A'};state.hits=[{title:'result'}];
    let prevented=false;
    await document.dispatch('keydown',{key:'Escape',preventDefault(){prevented=true;}});
    assert.equal(prevented,false);
    assert.ok(state.undockedAttribute);assert.equal(state.hits.length,1);
    await document.exitFullscreen();expectButton(false);
    // The button itself also exits native fullscreen.
    await button.dispatch('click');expectButton(true);
    await button.dispatch('click');assert.equal(document.fullscreenElement,null);expectButton(false);
    // Browsers can reject fullscreen (e.g. an embedded browser); fallback stays usable.
    workspace.requestFullscreen=async()=>{throw new Error('fullscreen denied');};
    await button.dispatch('click');expectButton(true);
    assert.equal(workspace.classList.contains('is-fullscreen-fallback'),true);
    assert.equal(document.body.classList.contains('graph-fullscreen-fallback'),true);
    prevented=false;
    await document.dispatch('keydown',{key:'Escape',preventDefault(){prevented=true;}});
    assert.equal(prevented,true);expectButton(false);
    assert.equal(workbodyFallback(),false);
    assert.ok(state.undockedAttribute);assert.equal(state.hits.length,1);
    // The fallback also works when the API is unavailable and closes via the button.
    delete workspace.requestFullscreen;
    await button.dispatch('click');expectButton(true);assert.equal(workbodyFallback(),true);
    await button.dispatch('click');expectButton(false);assert.equal(workbodyFallback(),false);
    function workbodyFallback(){return document.body.classList.contains('graph-fullscreen-fallback');}
  })()
`;

const result = vm.runInNewContext(script.slice(0, end) + exercise,
  {document, assert, console, Map, Set, URL, CSS: {escape: String}}, {timeout: 10000});
Promise.resolve(result).catch(error => { console.error(error); process.exitCode = 1; });
