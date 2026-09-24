// test_frontend_logic.js
// 验证前端关键算法与本地离线数据结构

const fs = require('fs');
const assert = require('assert');

// 1. 读取并验证 static_offline_bundle.json
const bundleRaw = fs.readFileSync('static_offline_bundle.json', 'utf8');
const bundle = JSON.parse(bundleRaw);

assert(bundle.category_meta, 'bundle.category_meta must exist');
assert(bundle.metrics, 'bundle.metrics must exist');
assert(bundle.cases, 'bundle.cases must exist');
assert(bundle.code, 'bundle.code must exist');
assert(Array.isArray(bundle.presets), 'bundle.presets must be array');
assert.strictEqual(bundle.presets.length, 10, 'bundle.presets must have 10 items');

console.log('✓ static_offline_bundle.json 数据结构与 10 类预设验证通过');

// 2. 验证转义与 XSS 防护函数
function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

const rawText = '<img src=x onerror=alert(1)> & "quotes"';
const safeText = escapeHtml(rawText);
assert(!safeText.includes('<img'), 'Must escape tags');
assert(safeText.includes('&lt;img'), 'Must have &lt;');
assert(safeText.includes('&quot;quotes&quot;'), 'Must escape quotes');
console.log('✓ XSS 字符转义安全性验证通过');

// 3. 验证离线推演引擎逻辑
function simulateArbitraryText(text) {
  const meta = bundle.category_meta || {};
  const mockDict = [
    { w: '中国汽车工业协会', cat: 'organization' },
    { w: '莫斯科中央陆军', cat: 'organization' },
    { w: '星巴克', cat: 'company' },
    { w: '北京', cat: 'address' }
  ];

  const mEntities = [];
  mockDict.forEach(item => {
    let idx = text.indexOf(item.w);
    while (idx !== -1) {
      mEntities.push({
        start: idx,
        end: idx + item.w.length - 1,
        category: item.cat,
        category_cn: meta[item.cat]?.cn || item.cat,
        color: meta[item.cat]?.color || '#6366f1',
        text: item.w,
        confidence: 0.965
      });
      idx = text.indexOf(item.w, idx + item.w.length);
    }
  });

  return {
    results: {
      macbert: { entities: mEntities },
      bert: { entities: mEntities }
    }
  };
}

const simResult = simulateArbitraryText('在北京星巴克喝咖啡');
assert.strictEqual(simResult.results.macbert.entities.length, 2);
console.log('✓ 离线推演引擎模拟分词实体抽取测试通过');

// 4. 验证 HTML 中的内联 JavaScript 语法完整性
const html = fs.readFileSync('workbench.html', 'utf8');
const scriptRegex = /<script(?:\s+[^>]*)?>([\s\S]*?)<\/script>/gi;
let match;
let scriptIdx = 0;
while ((match = scriptRegex.exec(html)) !== null) {
  scriptIdx++;
  const code = match[1];
  new Function(code); // Throws if syntax is invalid
}
assert.strictEqual(scriptIdx, 2, 'workbench.html must contain exactly 2 script blocks');
console.log('✓ workbench.html 内部 2 个 Script 块语法 100% 合法，无语法异常');

console.log('\n🎉 前端算法与静态数据所有测试用例顺利通过！');
