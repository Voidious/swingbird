from swingbird.daemon import main


def test_main(capsys):
    main()
    assert capsys.readouterr().out == "Hello from swingbird!\n"
