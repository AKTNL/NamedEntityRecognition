"""API and alignment regressions; fake model engine, real routes and service."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import app as backend
from generative_service import GenerativeNERService, parse_generation, validate_request, GenerativeServiceError


class GenerativeApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        model, adapter = root / "base", root / "trained_lora"
        model.mkdir()
        adapter.mkdir()
        for path in (model / "config.json", model / "model.safetensors",
                     adapter / "adapter_config.json", adapter / "adapter_model.safetensors"):
            path.write_text("{}")
        self.engine = Mock(device="cuda")
        self.engine.generate.return_value = ["@@张三##在北京。"]
        self.factory = Mock(return_value=self.engine)
        self.service = GenerativeNERService(model, adapter, engine_factory=self.factory)
        self.service_patch = patch.object(backend, "GENERATIVE_SERVICE", self.service)
        self.service_patch.start()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.service_patch.stop)
        self.client = backend.app.test_client()

    def predict(self, **changes):
        payload = {"text": "张三在北京。", "categories": ["name"]}
        payload.update(changes)
        return self.client.post("/api/generative/predict", json=payload)

    def test_status_is_lazy_and_model_is_reused_with_adapter(self):
        status = self.client.get("/api/generative/status").get_json()["data"]
        self.assertTrue(status["available"])
        self.assertFalse(status["loaded"])
        self.factory.assert_not_called()
        for _ in range(2):
            response = self.predict()
            self.assertEqual(response.status_code, 200)
            data = response.get_json()["data"]
            self.assertEqual(data["source"], "live")
            self.assertEqual(data["entities"][0]["text"], "张三")
            self.assertNotIn("confidence", data["entities"][0])
        self.factory.assert_called_once()
        args = self.factory.call_args.kwargs
        self.assertEqual(args["lora_dir"], str(self.service.lora_dir))
        self.assertTrue(args["local_files_only"])
        self.assertFalse(args["chat_template"])
        self.assertEqual(self.engine.generate.call_args.kwargs["max_new_tokens"], 96)
        self.assertTrue(self.client.get("/api/generative/status").get_json()["data"]["loaded"])

    def test_validation_happens_before_model_load(self):
        cases = [None, [], {"text": None}, {"text": "  "}, {"text": "a" * 129},
                 {"text": "a\nb"}, {"text": "a\rb"}, {"text": "a\x00b"},
                 {"text": "@@张三##"}, {"text": "张三", "categories": []},
                 {"text": "张三", "categories": "name"}, {"text": "张三", "categories": [{}]},
                 {"text": "张三", "categories": ["bad"]}, {"text": "张三", "categories": ["name", "name"]},
                 {"text": "张三", "max_new_tokens": True}, {"text": "张三", "max_new_tokens": "96"},
                 {"text": "张三", "max_new_tokens": 257}, {"text": "张三", "max_new_tokens": 31}]
        for case in cases:
            with self.subTest(case=case):
                response = self.client.post("/api/generative/predict", json=case)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.get_json()["code"], "invalid_request")
        self.factory.assert_not_called()

    def test_invalid_json_returns_json_error(self):
        response = self.client.post("/api/generative/predict", data="{broken", content_type="application/json")
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()["success"])

    def test_missing_adapter_never_falls_back_to_base(self):
        (self.service.lora_dir / "adapter_model.safetensors").unlink()
        self.assertFalse(self.service.status()["available"])
        response = self.predict()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()["code"], "unavailable")
        self.factory.assert_not_called()

    def test_missing_dependency_is_actionable(self):
        with patch("generative_service.importlib.util.find_spec", return_value=None):
            response = self.predict()
        self.assertEqual(response.status_code, 503)
        self.assertIn("peft", response.get_json()["error"])
        self.factory.assert_not_called()

    def test_busy_is_rejected_without_queueing(self):
        self.service._lock.acquire()
        try:
            self.assertTrue(self.service.status()["busy"])
            response = self.predict()
            self.assertEqual(response.status_code, 409)
            self.assertEqual(response.headers["Retry-After"], "3")
        finally:
            self.service._lock.release()
        self.factory.assert_not_called()

    def test_failed_loading_releases_lock_and_can_retry(self):
        self.factory.side_effect = [RuntimeError("load failed"), self.engine]
        with patch.object(backend.app.logger, "exception"):
            self.assertEqual(self.predict().status_code, 503)
        self.assertFalse(self.service.status()["loaded"])
        self.assertFalse(self.service.status()["busy"])
        self.assertEqual(self.predict().status_code, 200)

    def test_generation_error_has_no_prediction_fallback(self):
        self.engine.generate.side_effect = RuntimeError("out of memory")
        with patch.object(backend.app.logger, "exception"):
            response = self.predict()
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("data", response.get_json())
        self.assertEqual(response.get_json()["code"], "inference_failed")
        self.assertFalse(self.service.status()["busy"])

    def test_output_count_is_checked(self):
        self.engine.generate.return_value = []
        with patch.object(backend.app.logger, "exception"):
            self.assertEqual(self.predict().status_code, 503)

    def test_malformed_is_returned_but_not_highlighted(self):
        self.engine.generate.return_value = ["@@张三在北京。"]
        data = self.predict().get_json()["data"]
        self.assertEqual(data["warning_count"], 1)
        self.assertEqual(data["entities"], [])
        self.assertEqual(data["outputs"][0]["output"], "@@张三在北京。")


class GenerativeAlignmentTests(unittest.TestCase):
    def test_duplicate_names_retain_exact_occurrences(self):
        data = parse_generation("张三遇见张三。", "name", "@@张三##遇见@@张三##。")
        self.assertTrue(data["valid"])
        self.assertEqual([(e["start"], e["end"]) for e in data["entities"]], [(0, 1), (4, 5)])

    def test_unicode_offsets_match_python_codepoints(self):
        data = parse_generation("😀张三在北京。", "name", "😀@@张三##在北京。")
        self.assertEqual(data["entities"][0]["start"], 1)
        self.assertEqual(data["entities"][0]["end"], 2)

    def test_changed_original_or_bad_markers_never_create_spans(self):
        for output in ["@@李四##在北京。", "@@张三##在上海。", "@@张三##", "@@张三@@在北京。", "@@##张三在北京。", "@@张@@三####在北京。", ""]:
            with self.subTest(output=output):
                data = parse_generation("张三在北京。", "name", output)
                self.assertFalse(data["valid"])
                self.assertEqual(data["entities"], [])
                self.assertTrue(data["warnings"])

    def test_empty_entity_result_is_valid_if_original_is_preserved(self):
        data = parse_generation("没有实体。", "name", "没有实体。")
        self.assertTrue(data["valid"])
        self.assertEqual(data["entities"], [])

    def test_input_is_preserved_and_all_types_are_default(self):
        text, categories, tokens = validate_request({"text": " 张三 "})
        self.assertEqual(text, " 张三 ")
        self.assertEqual(len(categories), 10)
        self.assertEqual(tokens, 96)


if __name__ == "__main__":
    unittest.main()
