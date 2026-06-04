import argparse
import json
import os

import requests

DEFAULT_URL = "http://127.0.0.1:5000/#/models/sml_system"


def post_prediction(url, payload):
	headers = {"Content-Type": "application/json"}
	response = requests.post(url, json=payload, headers=headers, timeout=10)
	response.raise_for_status()
	return response.json()


def parse_args():
	parser = argparse.ArgumentParser(description="Call MLflow model REST endpoint.")
	parser.add_argument(
		"--url",
		default=os.getenv("MLFLOW_MODEL_URL", DEFAULT_URL),
		help="Model REST endpoint, default from MLFLOW_MODEL_URL or localhost.",
	)
	parser.add_argument(
		"--data",
		default="[[1, 2, 3, 4]]",
		help="JSON array for model input, example: '[[1,2,3,4]]'.",
	)
	return parser.parse_args()


def main():
	args = parse_args()
	data = json.loads(args.data)
	payload = {"data": data}
	result = post_prediction(args.url, payload)
	print(result.get("predictions"))


if __name__ == "__main__":
	main()