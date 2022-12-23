import json
import pprint
import traceback
from pathlib import Path

import requests
from requests import HTTPError

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1"
port = "8000"

with open("config.json", "r") as f:
    config = json.load(f)

output_model_path = Path(config["output_model_path"])
api_returns_filename = config["api_returns_filename"]


def make_request(endpoint, url=URL, port=port, method="GET", params=None):
    print(f"Making {method} request on {url}:{port}/{endpoint} with params = {params}")

    try:
        if method == "POST":
            response = requests.post(url=f"{url}:{port}/{endpoint}", params=params)
        else:
            response = requests.get(url=f"{url}:{port}/{endpoint}")

        response.raise_for_status()
    except HTTPError as exc:
        print(f"Error: {exc}")
        traceback.print_exc()

    print(f"return {response.text}")
    return response.text


def make_calls(api_returns_filename=api_returns_filename):
    responses_path = Path(output_model_path, "apireturns.txt")
    # Call each API endpoint and store the responses
    response1 = make_request(endpoint="prediction", params={"filename": "testdata/testdata.csv"}, method="POST")
    response2 = make_request(endpoint="scoring")
    response3 = make_request(endpoint="summarystats")
    response4 = make_request(endpoint="diagnostics")

    # combine all API responses
    # combine reponses here
    responses = {
        "prediction": response1,
        "scoring": eval(response2),
        "summarystats": eval(response3),
        "diagnostics": eval(response4),
    }

    # write the responses to your workspace
    with open(Path(output_model_path, api_returns_filename), "w") as fd:
        pprint.pprint(responses, stream=fd)


if __name__ == "__main__":
    make_calls()
