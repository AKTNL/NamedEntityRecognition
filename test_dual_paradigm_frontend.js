// Execute the generated application's actual functions without starting the UI.
// Browser layout and interaction checks complement these data-boundary regressions.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');

const html = fs.readFileSync('workbench.html', 'utf8');
assert.equal(html, fs.readFileSync('templates/index.html', 'utf8'), 'Build outputs must match');
const markup = html.split('<script>')[0];
const ids = [...markup.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]);
assert.equal(ids.length, new Set(ids).size, 'DOM IDs must be unique');
const scripts = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)].map(match => match[1]);
assert.equal(scripts.length, 2);
const nodes = new Map();
function getNode(id) {
  if (!nodes.has(id)) nodes.set(id, {value: '', textContent: '', innerHTML: '', style: {}, classList: {add() {}, remove() {}, toggle() {}}, setAttribute() {}});
  return nodes.get(id);
}
const context = vm.createContext({
  window: {}, document: {getElementById: getNode}, console,
  setTimeout, clearTimeout, Blob, URL,
});
vm.runInContext(scripts[0], context);
const boot = /if \(document\.readyState === 'loading'\) \{[\s\S]*?\} else \{\s*main\(\);\s*\}/;
assert.match(scripts[1], boot);
const executable = scripts[1].replace(boot, `window.testAPI = {
  state, bundle, percent, runOfflineSimulation, renderHighlightedSpanText,
  renderGeneration, renderProtocol, renderGenerativeCase, renderGalleryCases,
  selectExperiment(index) { selectedExperiment = experiments[index]; },
  clearExperiment() { selectedExperiment = null; }
};`);
vm.runInContext(executable, context);
const api = context.window.testAPI;
assert.equal(api.percent(0), '0.00%');
assert.equal(api.percent(null), '—');
assert.equal(api.percent(NaN), '—');

// Reusing a stored span for a substring or prefixed sentence corrupts its offset.
const preset = api.bundle.presets[0];
assert.equal(api.runOfflineSimulation(preset.text).source, 'recorded');
assert.equal(api.runOfflineSimulation(`前缀${preset.text}`).source, 'dictionary');
assert.equal(api.runOfflineSimulation(preset.text.slice(0, 5)).source, 'dictionary');
const simulated = api.runOfflineSimulation('腾讯测试文本');
assert.equal(simulated.results.macbert.entities[0].confidence, null);
const highlighted = api.renderHighlightedSpanText('腾讯测试文本', simulated.results.macbert.entities);
assert.ok(highlighted.includes('不适用'));
assert.ok(!highlighted.includes('96.5%'));

// Values must preserve units and the selected artifact, including literal zero.
getNode('generationMetric').value = 'f1';
for (const [index, experiment] of api.bundle.experiments.entries()) {
  // Original run artifacts are optional; shipped HTML must be testable alone.
  const raw = fs.existsSync(experiment.source)
    ? JSON.parse(fs.readFileSync(experiment.source, 'utf8'))
    : {metrics: experiment.metrics};
  if (fs.existsSync(experiment.source)) {
    assert.deepEqual(JSON.parse(JSON.stringify(experiment.metrics)), raw.metrics);
  }
  api.selectExperiment(index);
  api.renderGeneration();
  assert.equal(getNode('experimentTitle').textContent, experiment.label);
  assert.ok(getNode('experimentStats').innerHTML.includes(api.percent(raw.metrics.micro_f1)));
  assert.ok(getNode('experimentStats').innerHTML.includes(api.percent(raw.metrics.malformed_rate * 100)));
  assert.ok(getNode('generationSource').textContent.includes(experiment.source));
}

// Protocol examples must retain the original text and exact inclusive offsets.
for (const category of ['name', 'address', 'company']) {
  getNode('protocolCategory').value = category;
  api.renderProtocol();
  const [entity] = JSON.parse(getNode('protocolEntities').textContent);
  assert.equal('张三在北京加入了腾讯。'.slice(entity.start, entity.end + 1), entity.text);
  assert.equal(getNode('protocolOutput').textContent.replace('@@', '').replace('##', ''), '张三在北京加入了腾讯。');
}

// Source text and raw generated output remain inert in all case views.
const hostile = '<img src=x onerror=alert(1)>';
const safe = api.renderGenerativeCase({text: hostile, model_output: hostile, ground_truth: [], generative_prediction: [{text: hostile, category: '\" onclick=\"alert(1)'}]});
assert.ok(!safe.includes('<img'));
assert.ok(safe.includes('&lt;img'));
assert.ok(!safe.includes('badge-" onclick='));
api.state.activeFilter = 'generative';
getNode('caseSearch').value = '';
api.renderGalleryCases();
assert.ok(getNode('caseCount').textContent.includes(String(api.bundle.cases.category_c_generative_failures.length)));
getNode('caseSearch').value = 'no-match-123456';
api.renderGalleryCases();
assert.equal(getNode('caseCount').textContent, '显示 0 个案例');
assert.ok(getNode('casesGridContainer').innerHTML.includes('暂无案例'));

api.clearExperiment();
api.renderGeneration();
assert.equal(getNode('exportExperiment').disabled, true);
assert.ok(getNode('experimentStats').innerHTML.includes('暂无实验数据'));
console.log('✓ Generated frontend: artifact fidelity, units, exact spans, protocol, XSS, case filtering, empty state');
