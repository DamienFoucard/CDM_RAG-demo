"""The documentation hub links to an accurate, served MCP guide."""

from html.parser import HTMLParser

from fastapi.testclient import TestClient

from app.main import app


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.append(dict(attrs))


def test_hub_links_to_served_mcp_guide():
    client = TestClient(app)
    hub = client.get("/docs")
    assert hub.status_code == 200
    links = LinkParser()
    links.feed(hub.text)
    assert any(link.get("href") == "/docs/mcp" for link in links.links)
    # Hide the token link until the token manager is available in this install.
    assert any(link.get("href") == "/api-tokens" and "hidden" in link for link in links.links)

    guide = client.get("/docs/mcp")
    assert guide.status_code == 200
    assert "First-party MCP server" in guide.text
    assert "/api/v1/mcp/" in guide.text
    assert "ask_schema_question" in guide.text
    assert "service_status" in guide.text
    assert "app.mcp_stdio" in guide.text
    assert "CDM_RAG_API_TOKEN" in guide.text
    assert "does not provide OAuth discovery" in guide.text
    assert "MCP server not yet available" not in guide.text
    links = LinkParser()
    links.feed(guide.text)
    assert any(link.get("href") == "/docs/api" for link in links.links)
    assert any(link.get("href") == "https://modelcontextprotocol.io/" for link in links.links)
