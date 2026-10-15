from isa.body import Body, BodyParams, SCRIPTED_A, EVENTS, CHANNELS


def test_scripted_curve_follows_script():
    b = Body(BodyParams(noise=0))
    a = [b.step(set_a=x)["a"] for x in SCRIPTED_A]
    assert a == [round(x, 4) for x in SCRIPTED_A]


def test_open_loop_returns_to_baseline():
    b = Body(BodyParams(noise=0, gain=0))
    a = [b.step(event=e)["a"] for e in EVENTS]
    assert max(a) > 0.5
    assert a[-1] < 0.25


def test_closed_loop_threat_runs_away_effort_settles():
    def run(threat):
        b = Body(BodyParams(noise=0, gain=0.2))
        return [b.step(event=e, threat=threat)["a"] for e in EVENTS]
    hot, cool = run(1.0), run(-1.0)
    assert hot[-1] > 0.6
    assert cool[-1] < 0.2


def test_slow_channels_lag():
    b = Body(BodyParams(noise=0))
    rows = [b.step(set_a=x) for x in SCRIPTED_A]
    peak_a = max(range(len(rows)), key=lambda i: rows[i]["a"])
    peak_gut = max(range(len(rows)), key=lambda i: rows[i]["a_gut"])
    assert peak_gut > peak_a


def test_report_mentions_all_channels():
    b = Body()
    b.step(set_a=0.9)
    r = b.report(CHANNELS)
    for c in CHANNELS:
        assert c in r
    assert "rising" in r
