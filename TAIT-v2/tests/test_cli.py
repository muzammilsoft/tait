from tait.cli.main import main


def test_version(capsys):
    main(["version"])
    assert "TAIT 2.1.0" in capsys.readouterr().out
