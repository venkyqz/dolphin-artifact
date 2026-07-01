from sweagent.config.schema import parse_swebench_uri


def test_issue_schema():
    issue = "swebench://lite.test/django__django-14411"
    config = parse_swebench_uri(issue)
    print(config)

    issue = "hello, hi"
    config = parse_swebench_uri(issue)
    print(config)

