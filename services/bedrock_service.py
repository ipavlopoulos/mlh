import json

import boto3


class BedrockService:
    def __init__(self, region_name, model_id):
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region_name)

    def generate(self, prompt):
        response = self.client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "prompt": prompt,
                "max_gen_len": 300,
                "temperature": 0.7,
                "top_p": 0.9,
            }),
        )
        body = json.loads(response["body"].read())
        return body["generation"]
