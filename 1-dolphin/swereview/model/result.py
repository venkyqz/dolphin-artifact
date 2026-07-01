from pydantic import BaseModel


class SWEResults(BaseModel):
    applied: list[str] = []
    generated: list[str] = []
    install_fail: list[str] = []
    no_apply: list[str] = []
    no_generation: list[str] = []
    reset_failed: list[str] = []
    resolved: list[str] = []
    test_errored: list[str] = []
    test_timeout: list[str] = []
    with_logs: list[str] = []


# Example usage:
# data = {
#     "applied": ["test1", "test2"],
#     "generated": ["gen1", "gen2", "gen3"],
#     "install_fail": ["fail1"],
#     "no_apply": [],
#     "no_generation": ["nogen1"],
#     "reset_failed": [],
#     "resolved": ["res1", "res2"],
#     "test_errored": ["err1"],
#     "test_timeout": ["timeout1"],
#     "with_logs": ["log1", "log2"]
# }
# results = TestResults(**data)
