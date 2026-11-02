from newsclf.serving.registry import Registry


def test_registry_labels_and_config(tmp_project):
    reg = Registry(tmp_project)
    for v in (1, 2, 3):
        reg.register(v, {"source": f"E{v}", "input_type": "text", "labels": ["A"]})
    reg.promote(1)
    reg.set_label("canary", 2)
    reg.promote(2)
    assert reg.data["labels"] == {"deprecated": 1, "stable": 2}
    reg.set_label("canary", 3)
    reg.save()
    native = (reg.root / "models.config").read_text()
    docker = (reg.root / "models.docker.config").read_text()
    assert "versions: 3" in native and "key: 'canary'" in native
    assert "base_path: '/models/newsapp_model'" in docker
    assert Registry(tmp_project).version_for("stable") == 2
    assert Registry(tmp_project).version_for() == 3
