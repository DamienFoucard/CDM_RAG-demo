// Exercise the actual inline graph renderer without a browser dependency.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const html = fs.readFileSync(process.argv[2], 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)?.[1];
assert.ok(script, 'graph page should contain its rendering script');
const cutoff = script.indexOf("    $('search-form').addEventListener");
assert.ok(cutoff > 0, 'find the event-binding boundary');

class Element {
  constructor(tag) {
    this.tagName = tag;
    this.attributes = {};
    this.children = [];
    this.style = {setProperty() {}};
    this.classList = {toggle() {}, contains: value => this.classes().includes(value)};
  }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  getAttribute(name) { return this.attributes[name]; }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this.children = items; }
  addEventListener() {}
  classes() { return (this.attributes.class || this.className || '').split(/\s+/); }
  focus() {}
}
const elements = new Map();
const document = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, new Element(id));
    return elements.get(id);
  },
  createElement(tag) { return new Element(tag); },
  createElementNS(ns, tag) { return new Element(tag); },
};

const test = `
  // The legend and layout are orthogonal to visibility. Keep the graph's
  // real renderer and semantic edge/node builders, using a small fixture.
  renderGraphLegend=()=>{}; updateNodeDetailToggle=()=>{}; appendNodeResizeHandle=()=>{};
  const nodes=[{id:'A',name:'A',model:'applicationCommon'}, {id:'B',name:'B',model:'applicationCommon'}];
  const edges=[{id:'ab',source_id:'A',target_id:'B',attribute:'ab'},
               {id:'ba',source_id:'B',target_id:'A',attribute:'ba'}];
  const trait=(id)=>({id,name:id,entity_applications:1,attribute_applications:0,data_type_applications:0,manifest_applications:0});
  const semantics={
    manifests:{nodes:[{id:'M',name:'M'},{id:'M2',name:'M2'}],edges:[
      {type:'sub_manifest',source_id:'M',target_id:'M2'},
      {type:'declares_entity',source_id:'M',target_id:'A'},
      {type:'declares_entity',source_id:'M',target_id:'B'}]},
    traits:{nodes:[trait('T'),trait('T2')],edges:[
      {source_id:'A',target_id:'T',entity_application_count:1,attribute_application_count:0,data_type_application_count:0},
      {source_id:'B',target_id:'T',entity_application_count:1,attribute_application_count:0,data_type_application_count:0}],
      inheritance_edges:[{source_id:'T',target_id:'T2'}]},
    inheritance:{nodes:[{id:'I',name:'I'}],edges:[{source_id:'A',target_id:'I'},{source_id:'B',target_id:'I'}]}
  };
  const positions=new Map([['A',{x:100,y:100}],['B',{x:400,y:100}]]);
  const layout={semantics,orderedTraits:semantics.traits.nodes,
    manifestPositions:new Map([['M',{x:100,y:300}],['M2',{x:400,y:300}]]),
    traitPositions:new Map([['T',{x:100,y:500}],['T2',{x:400,y:500}]]),
    basePositions:new Map([['I',{x:650,y:300}]])};
  buildSemanticOverlayLayout=()=>layout;
  Object.assign(state,{fullGraph:{nodes,edges,semantics,entity_details:{A:{attributes:[]},B:{attributes:[]}}},
    neighborhood:{nodes,edges,focus_entity:nodes[0]},focusId:'A',infiniteMode:true,
    infinitePositions:positions,showClusters:false,infiniteAnalysis:{category_by_entity:{A:'one',B:'two'},categories:[]}});
  const graph=$('graph');
  const visible=(klass)=>graph.children.filter(el=>el.classes().includes(klass)
    && !el.classes().includes('layer-hidden') && !el.classes().includes('category-hidden'));
  function counts(expected){
    for(const [klass,count] of Object.entries(expected))assert.equal(visible(klass).length,count,klass);
  }
  const full={node:2,edge:2,'edge-label':2,'manifest-child':1,'manifest-declaration':2,
    'trait-use':2,'trait-inheritance':1,'entity-inheritance':2};
  renderGraph(); counts(full);
  // The entity toggle removes only edges that have an entity endpoint.
  state.showEntityNodes=false;renderGraph();counts({node:0,edge:0,'edge-label':0,
    'manifest-child':1,'manifest-declaration':0,'trait-use':0,'trait-inheritance':1,'entity-inheritance':0});
  state.showEntityNodes=true;renderGraph();counts(full);
  // Hide one category: neither direction of a relationship touching A remains,
  // nor its labels or semantic links; B's semantic links remain.
  state.hiddenCategories.add('one');renderGraph();counts({node:1,edge:0,'edge-label':0,
    'manifest-child':1,'manifest-declaration':1,'trait-use':1,'trait-inheritance':1,'entity-inheritance':1});
  state.hiddenCategories.clear();renderGraph();counts(full);
  // Removing a trait also removes both its applications and its inheritance.
  state.hiddenTraits.add('T');renderGraph();counts({'trait-use':0,'trait-inheritance':0,'manifest-declaration':2});
  state.hiddenTraits.clear();renderGraph();counts(full);
  state.showManifests=false;renderGraph();counts({'manifest-child':0,'manifest-declaration':0,'trait-use':2,'entity-inheritance':2});
  state.showManifests=true;state.showInheritance=false;renderGraph();counts({'manifest-declaration':2,'trait-use':2,'entity-inheritance':0});
  state.showInheritance=true;state.showRelationshipEdges=false;renderGraph();counts({node:2,edge:0,'edge-label':0,'manifest-declaration':2,'trait-use':2});
  state.showRelationshipEdges=true;renderGraph();counts(full);
  // A detached attribute card and connector cannot outlive its owner.
  state.showNodeDetails=true;
  state.undockedAttribute={entity_id:'A',attribute:{name:'name'}};
  renderGraph();counts({'undocked-card':1,'undocked-link':1});
  state.hiddenCategories.add('one');renderGraph();counts({'undocked-card':0,'undocked-link':0});
  state.hiddenCategories.clear();state.showEntityNodes=false;renderGraph();counts({'undocked-card':0,'undocked-link':0});
  state.showEntityNodes=true;renderGraph();counts({'undocked-card':1,'undocked-link':1});
`;

vm.runInNewContext(script.slice(0, cutoff) + test, {document, assert, console, Map, Set, URL, CSS: {escape: String}},
                   {timeout: 10000});
