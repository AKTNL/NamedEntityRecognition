"""
自动化全方位质量验证套件：
MacBERT × 中文 NER 交互式复现工作台与全栈模型体验系统
"""

import os
import sys
import json
import re
import unittest
from typing import Dict, Any

from app import app, load_models, ensure_models_loaded, APP_ROOT, CATEGORIES, CATEGORY_META

class WorkbenchSystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """初始化测试环境与模型检查点"""
        ensure_models_loaded()
        cls.client = app.test_client()

    # =========================================================================
    # 1. 核心 HTTP 路由与 REST API 验证
    # =========================================================================
    def test_root_index_endpoint(self):
        """测试根路由 / 能够正确提供完整 HTML 工作台"""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("MacBERT", html)
        self.assertIn("CLUENER2020", html)
        self.assertIn("tab-arena", html)

    def test_standalone_workbench_route(self):
        """测试 /workbench.html 路由正常响应"""
        res = self.client.get("/workbench.html")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn("window.__NER_BUNDLE__", html)

    def test_api_health(self):
        """测试健康检查与状态上报接口"""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["models"]["bert"])
        self.assertTrue(data["models"]["macbert"])
        self.assertEqual(data["meta"]["categories_count"], 10)
        self.assertEqual(set(data["meta"]["categories"]), set(CATEGORIES))

    def test_api_metrics(self):
        """测试学术评测指标大盘接口"""
        res = self.client.get("/api/metrics")
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertTrue(payload["success"])
        data = payload["data"]
        
        # 验证指标数值与模型配置
        self.assertIn("models", data)
        self.assertIn("bert", data["models"])
        self.assertIn("macbert", data["models"])
        self.assertEqual(data["models"]["macbert"]["best_f1"], 76.58)
        self.assertEqual(data["models"]["bert"]["best_f1"], 76.18)
        # 头条增益必须是同口径的 dev 差值 (+0.40pp)，不得再使用跨口径的 1.62
        self.assertEqual(data["headline_delta_f1"], 0.4)
        self.assertEqual(data["dev_delta_f1"], 0.4)
        # 论文基线：BERT 有原文出处 78.82；MacBERT 论文未做 CLUENER，应为 None
        self.assertEqual(data["models"]["bert"]["paper_test_f1"], 78.82)
        self.assertIsNone(data["models"]["macbert"]["paper_test_f1"])
        
        # 验证 10 大细粒度类别
        comp = data["categories_comparison"]
        self.assertEqual(len(comp), 10)
        cat_names = [c["category"] for c in comp]
        self.assertEqual(set(cat_names), set(CATEGORIES))
        
        # 验证超参数记录
        hp = data["hyperparameters"]
        self.assertEqual(hp["batch_size"], 16)
        self.assertEqual(hp["learning_rate"], "3e-5")
        self.assertEqual(hp["epochs"], 3)
        self.assertEqual(hp["max_len"], 128)

    def test_api_cases(self):
        """测试真实错例与典型对比案例接口"""
        res = self.client.get("/api/cases")
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertTrue(payload["success"])
        data = payload["data"]
        self.assertIn("category_a_macbert_advantages", data)
        self.assertIn("category_b_hard_cases", data)
        self.assertGreater(len(data["category_a_macbert_advantages"]), 0)
        self.assertGreater(len(data["category_b_hard_cases"]), 0)

    def test_api_code(self):
        """测试复现代码深度透视与技术批注接口"""
        res = self.client.get("/api/code")
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertTrue(payload["success"])
        data = payload["data"]
        files = data["files"]
        self.assertEqual(len(files), 3)
        file_ids = [f["id"] for f in files]
        self.assertIn("dataset", file_ids)
        self.assertIn("train", file_ids)
        self.assertIn("analyze_cases", file_ids)
        
        for f in files:
            self.assertGreater(len(f["source_code"]), 100)
            self.assertGreater(len(f["annotations"]), 0)

    # =========================================================================
    # 2. 预测接口边界条件与推理健壮性测试
    # =========================================================================
    def test_api_predict_dual_models(self):
        """测试双模型前向序列标注与差异对比"""
        test_text = "莫斯科中央陆军vs波兹南、拉科vs费耶诺德、加拉塔萨雷vs梅塔利斯特，"
        res = self.client.post("/api/predict", json={"text": test_text, "model": "both"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["text"], test_text)
        
        # 验证两个模型输出
        results = data["results"]
        self.assertIn("macbert", results)
        self.assertIn("bert", results)
        
        m_res = results["macbert"]
        b_res = results["bert"]
        self.assertGreater(len(m_res["entities"]), 0)
        self.assertGreater(len(b_res["entities"]), 0)
        self.assertEqual(len(m_res["tokens"]), len(test_text))
        
        # 验证差异计算模块
        diff = data["diff"]
        self.assertIsNotNone(diff)
        self.assertTrue(diff["has_diff"])
        self.assertGreater(len(diff["boundary_repairs"]), 0)

    def test_api_predict_single_model_macbert(self):
        """测试单模型预测 (MacBERT)"""
        res = self.client.post("/api/predict", json={"text": "腾讯与阿里巴巴在杭州", "model": "macbert"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("macbert", data["results"])
        self.assertNotIn("bert", data["results"])
        self.assertIsNone(data["diff"])

    def test_api_predict_single_model_bert(self):
        """测试单模型预测 (BERT)"""
        res = self.client.post("/api/predict", json={"text": "中国科学院软件研究所", "model": "bert"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("bert", data["results"])
        self.assertNotIn("macbert", data["results"])

    def test_api_predict_empty_text(self):
        """测试空文本与纯空白字符输入防御"""
        res = self.client.post("/api/predict", json={"text": ""})
        self.assertEqual(res.status_code, 400)
        self.assertFalse(res.get_json()["success"])

        res2 = self.client.post("/api/predict", json={"text": "   \n\t  "})
        self.assertEqual(res2.status_code, 400)
        self.assertFalse(res2.get_json()["success"])

    def test_api_predict_invalid_payload(self):
        """测试非预期数据类型请求体容错"""
        res = self.client.post("/api/predict", json=[1, 2, 3])
        self.assertEqual(res.status_code, 400)
        self.assertFalse(res.get_json()["success"])

        res2 = self.client.post("/api/predict", json={"text": 12345})
        self.assertEqual(res2.status_code, 400)
        self.assertFalse(res2.get_json()["success"])

    def test_api_predict_truncation(self):
        """测试超长文本安全截断"""
        long_text = "中国" * 100  # 200 字符
        res = self.client.post("/api/predict", json={"text": long_text, "max_len": 64})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["is_truncated"])
        self.assertEqual(len(data["text"]), 64)
        self.assertEqual(data["original_length"], 200)

    def test_api_predict_special_characters(self):
        """测试特殊符号、标点、XSS 注入标签及生僻字符的鲁棒性"""
        special_text = "【快讯】<script>alert(1)</script> 字节跳动在北京海淀发布《豆包大模型》！#AI@2026"
        res = self.client.post("/api/predict", json={"text": special_text})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("macbert", data["results"])

    # =========================================================================
    # 3. 前端自包含、无外网强依赖与离线降级验证
    # =========================================================================
    def test_frontend_zero_external_dependency(self):
        """验证 index.html 与 workbench.html 绝无任何外部 CDN 资源依赖"""
        for filename in ["workbench.html", os.path.join("templates", "index.html")]:
            path = os.path.join(APP_ROOT, filename)
            self.assertTrue(os.path.exists(path), f"File {path} must exist")
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 确保没有外部 CDN 的 <script src="http"> 或 <link href="http">
            script_srcs = re.findall(r'<script[^>]+src=["\']([^"\']+)["\']', content, re.IGNORECASE)
            link_hrefs = re.findall(r'<link[^>]+href=["\']([^"\']+)["\']', content, re.IGNORECASE)
            
            for src in script_srcs:
                self.assertFalse(src.startswith("http://") or src.startswith("https://") or src.startswith("//"),
                                 f"External script dependency found: {src}")
            for href in link_hrefs:
                self.assertFalse(href.startswith("http://") or href.startswith("https://") or href.startswith("//"),
                                 f"External style dependency found: {href}")

    def test_offline_bundle_integrity(self):
        """验证离线数据包 static_offline_bundle.json 的完整性"""
        bundle_path = os.path.join(APP_ROOT, "static_offline_bundle.json")
        self.assertTrue(os.path.exists(bundle_path))
        with open(bundle_path, "r", encoding="utf-8") as f:
            bundle = json.load(f)
            
        self.assertIn("category_meta", bundle)
        self.assertIn("metrics", bundle)
        self.assertIn("cases", bundle)
        self.assertIn("code", bundle)
        self.assertIn("presets", bundle)
        self.assertIn("preset_predictions", bundle)
        self.assertEqual(len(bundle["presets"]), 10)
        self.assertGreater(len(bundle["preset_predictions"]), 0)

    def test_dom_elements_presence(self):
        """验证四大模块核心 DOM 组件在 workbench.html 中齐全"""
        workbench_path = os.path.join(APP_ROOT, "workbench.html")
        with open(workbench_path, "r", encoding="utf-8") as f:
            html = f.read()

        # 核心导航
        self.assertIn('data-tab="tab-arena"', html)
        self.assertIn('data-tab="tab-code"', html)
        self.assertIn('data-tab="tab-benchmark"', html)
        self.assertIn('data-tab="tab-gallery"', html)
        
        # 模块 1: Arena
        self.assertIn('id="arenaInput"', html)
        self.assertIn('id="btnRunArena"', html)
        self.assertIn('id="diffAlertContainer"', html)
        self.assertIn('id="macbertVisualBox"', html)
        self.assertIn('id="bertVisualBox"', html)
        self.assertIn('id="probeTableWrapper"', html)
        self.assertIn('id="customCasesPanel"', html)
        self.assertIn('id="btnExportCases"', html)
        self.assertIn('id="btnImportCases"', html)
        
        # 模块 2: Code
        self.assertIn('id="codeFilesNav"', html)
        self.assertIn('id="codeSourceDisplay"', html)
        self.assertIn('id="codeAnnotationsContainer"', html)
        
        # 模块 3: Benchmark
        self.assertIn('id="svgChartContainer"', html)
        self.assertIn('id="benchmarkTableBody"', html)
        
        # 模块 4: Gallery
        self.assertIn('id="casesGridContainer"', html)
        
        # LocalStorage 规范键
        self.assertIn('macbert_ner_custom_cases', html)
        self.assertIn('macbert_ner_theme', html)


if __name__ == "__main__":
    unittest.main()
