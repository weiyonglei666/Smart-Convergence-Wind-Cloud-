from app.config import Settings


def test_settings_reads_required_values(monkeypatch):
    monkeypatch.setenv("MYSQL_HOST", "example")
    monkeypatch.setenv("MYSQL_PORT", "3306")
    monkeypatch.setenv("MYSQL_USER", "reader")
    monkeypatch.setenv("MYSQL_PASSWORD", "secret")
    monkeypatch.setenv("MYSQL_DATABASE", "power")
    settings = Settings()
    assert settings.mysql_host == "example"
    assert settings.mysql_port == 3306
    assert settings.timezone_name == "Asia/Shanghai"


def test_nwp_host_is_configurable(monkeypatch):
    monkeypatch.setenv("NWP_SSH_HOST", "example-nwp")
    settings = Settings()
    assert settings.nwp_ssh_host == "example-nwp"
