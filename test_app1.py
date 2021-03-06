from fastapi.testclient import TestClient
from main import app
from datetime import datetime
# test to check the correct functioning of the /ping route
def test_ping():
    with TestClient(app) as client:
        response = client.get("/ping")
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"ping": "pong"}


def test_pred_defects():
# defining a sample payload for the testcase
    payload = {
        "mccabe_line_count_of_code": 28,
        "mccabe_cyclomatic_complexity": 6,
        "mccabe_essential_complexity": 5,
        "mccabe_design_complexity": 5,
        "halstead_total_operators_operands": 104,
        "halstead_volume": 564.33,
        "halstead_program_length": 0.06,
        "halstead_difficulty": 16.09,
        "halstead_intelligence": 35.08,
        "halstead_effort": 9078.38,
        "halstead_b": 0.19,
        "halstead_time_estimator": 504.35,
        "halstead_line_count": 2,
        "halstead_count_of_lines_of_comments": 7,
        "halstead_count_of_blank_lines": 0,
        "lOCodeAndComment": 0,
        "unique_operators": 20,
        "unique_operands": 23,
        "total_operators": 67,
        "total_operands": 37,
        "branchCount": 11
        }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json()["defects"] == "defects" # Changing the syntax due to added timestamp