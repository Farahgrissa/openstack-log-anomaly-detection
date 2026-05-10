"""Tests unitaires pour le module parsers (compatible unittest et pytest)."""
import unittest
from anomaly_detection.pipeline.parsers import (
    parse_uwsgi_payload,
    normalize_endpoint,
    status_class,
    template_payload,
    detect_log_kind,
)


class TestParseUwsgi(unittest.TestCase):
    def test_basic_get(self):
        p = "GET /healthcheck => generated 2 bytes in 36 msecs (HTTP/1.0 200) 3 headers in 98 bytes (1 switches on core 0)"
        r = parse_uwsgi_payload(p)
        self.assertIsNotNone(r)
        self.assertEqual(r["http_method"], "GET")
        self.assertEqual(r["http_path"], "/healthcheck")
        self.assertEqual(r["http_status"], 200)
        self.assertEqual(r["http_duration_ms"], 36)
        self.assertEqual(r["http_bytes"], 2)

    def test_post_with_query(self):
        p = "POST /v3/auth/tokens => generated 4138 bytes in 280 msecs (HTTP/1.1 201) 6 headers in 398 bytes (1 switches on core 0)"
        r = parse_uwsgi_payload(p)
        self.assertEqual(r["http_method"], "POST")
        self.assertEqual(r["http_status"], 201)

    def test_404_status(self):
        p = "GET /v2.1/servers/abc-def => generated 100 bytes in 50 msecs (HTTP/1.1 404) 4 headers in 162 bytes (1 switches on core 0)"
        r = parse_uwsgi_payload(p)
        self.assertEqual(r["http_status"], 404)

    def test_non_uwsgi_returns_none(self):
        p = " Agent rpc_loop - iteration:18234 completed."
        self.assertIsNone(parse_uwsgi_payload(p))

    def test_empty_returns_none(self):
        self.assertIsNone(parse_uwsgi_payload(""))


class TestStatusClass(unittest.TestCase):
    def test_2xx(self):
        self.assertEqual(status_class(200), "2xx")
        self.assertEqual(status_class(204), "2xx")

    def test_4xx(self):
        self.assertEqual(status_class(404), "4xx")
        self.assertEqual(status_class(401), "4xx")

    def test_5xx(self):
        self.assertEqual(status_class(500), "5xx")
        self.assertEqual(status_class(503), "5xx")

    def test_none(self):
        self.assertEqual(status_class(None), "none")


class TestNormalizeEndpoint(unittest.TestCase):
    def test_uuid_replaced(self):
        ep = normalize_endpoint("DELETE", "/v2.1/servers/abc12345-def6-7890-abcd-ef1234567890")
        self.assertEqual(ep, "DELETE /v2.1/servers/<UUID>")

    def test_numeric_id_replaced(self):
        ep = normalize_endpoint("PUT", "/allocations/123456")
        self.assertEqual(ep, "PUT /allocations/<ID>")

    def test_query_string_removed(self):
        ep = normalize_endpoint("GET", "/v2.0/ports?device_owner=network:dhcp")
        self.assertEqual(ep, "GET /v2.0/ports")

    def test_no_change_simple(self):
        ep = normalize_endpoint("GET", "/healthcheck")
        self.assertEqual(ep, "GET /healthcheck")


class TestTemplatePayload(unittest.TestCase):
    def test_uuid_normalized(self):
        # Utiliser de vrais UUIDs hexadécimaux
        p1 = "Server abc12345-def6-7890-abcd-ef1234567890 created"
        p2 = "Server 11112222-3333-4444-5555-666677778888 created"
        self.assertEqual(template_payload(p1), template_payload(p2))

    def test_ip_normalized(self):
        p1 = "Connection from 192.168.1.42"
        p2 = "Connection from 10.0.0.5"
        self.assertEqual(template_payload(p1), template_payload(p2))

    def test_numbers_normalized(self):
        p1 = "Processed 42 items in 1.5 sec"
        p2 = "Processed 99 items in 3.7 sec"
        self.assertEqual(template_payload(p1), template_payload(p2))

    def test_different_messages_differ(self):
        p1 = "Server created"
        p2 = "Server deleted"
        self.assertNotEqual(template_payload(p1), template_payload(p2))

    def test_empty(self):
        self.assertEqual(template_payload(""), "")

    def test_truncates_very_long(self):
        long_p = "abc " * 500  # 2000 chars
        result = template_payload(long_p)
        self.assertLessEqual(len(result), 510)


class TestDetectLogKind(unittest.TestCase):
    def test_uwsgi_is_http_access(self):
        self.assertEqual(detect_log_kind("nova-api-uwsgi", "...", True), "http_access")
        self.assertEqual(detect_log_kind("keystone-uwsgi", "weird payload", False), "http_access")

    def test_infra_services(self):
        self.assertEqual(detect_log_kind("rabbit", "x", False), "infra_log")
        self.assertEqual(detect_log_kind("mariadb", "x", False), "infra_log")
        self.assertEqual(detect_log_kind("openvswitch", "x", False), "infra_log")

    def test_agent_log(self):
        self.assertEqual(detect_log_kind("neutron-openvswitch-agent", "x", False), "agent_log")
        self.assertEqual(detect_log_kind("neutron-l3-agent", "x", False), "agent_log")

    def test_app_log_default(self):
        self.assertEqual(detect_log_kind("nova-api", "x", False), "app_log")
        self.assertEqual(detect_log_kind("cinder-volume", "x", False), "app_log")


if __name__ == "__main__":
    unittest.main(verbosity=2)
