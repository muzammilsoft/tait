from tait.cli.main import main


def test_version(capsys):
    main(["version"])
    assert "TAIT 2.3.1" in capsys.readouterr().out
