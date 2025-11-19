from eesizer_core.spice import (
    ControlDeck,
    ac_simulation,
    augment_netlist,
    describe_measurements,
    measure_gain,
    parse_measure_log,
)


def test_control_deck_renders_and_augment_netlist():
    deck = ControlDeck(
        name="ac_combo",
        directives=(ac_simulation("dec", 10, 1.0, 1e6),),
        measurements=(measure_gain("gain_db", "out", "in"),),
    )
    rendered = deck.render()
    assert rendered.startswith(".control")
    assert ".measure ac gain_db" in rendered
    netlist = "V1 in 0 1\n.end\n"
    augmented = augment_netlist(netlist, deck)
    assert ".endc" in augmented


def test_parse_measurement_log_and_summary():
    log = """\nMeasure gain_db = 42.5\npower_mw = 3.1\n"""
    metrics = parse_measure_log(log)
    assert metrics["gain_db"] == 42.5
    summary = describe_measurements(metrics)
    assert "gain_db=42.500" in summary
