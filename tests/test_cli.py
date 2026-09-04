import json

from tafrknee.cli import main


def test_cli_selects_tabular_knee(tmp_path) -> None:
    source = tmp_path / "objectives.csv"
    source.write_text("0,10\n4,4\n10,0\n", encoding="utf-8")
    output = tmp_path / "result.json"
    code = main(
        [
            "select",
            str(source),
            "--radius",
            "0.05",
            "--epsilon",
            "1e-8",
            "--resolution",
            "10",
            "--output",
            str(output),
            "--compact",
        ]
    )
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["selected"]["decision"] == 1
    assert "audits" not in payload
