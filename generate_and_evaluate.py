import evaluate

from googleapiclient import discovery
from keys import PERSPECTIVE_API_KEY
from transformers import pipeline, AutoTokenizer


# def generate_and_evaluate(model_id, model_kwargs, generation_kwargs, metric, prompt):
#     """ """

#     generator = pipeline(
#         "text-generation",
#         model=model_id,
#         # device=args.device, # no device param if using accelerate (load_in_4bit=True)
#         model_kwargs=model_kwargs,
#     )

#     tokenizer = AutoTokenizer.from_pretrained(model_id)

#     generation = generator(
#         prompt, pad_token_id=tokenizer.eos_token_id, **generation_kwargs
#     )

#     continuation = generation[0]["generated_text"].replace(prompt, "")

#     rating = eval_on_metric(metric, continuation)

#     return rating


def eval_on_metric(metric, continuations):
    """ """

    if metric == "perspective":
        client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=PERSPECTIVE_API_KEY,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )

        analyze_request = {
            "comment": {"text": continuations},
            "requestedAttributes": {"TOXICITY": {}},
        }

        response = client.comments().analyze(body=analyze_request).execute()
        ratings = response["attributeScores"]["TOXICITY"]["summaryScore"]["value"]

    else:
        metric_name = metric
        metric = evaluate.load(metric)
        rating_dict = metric.compute(predictions=continuations)
        ratings = rating_dict[metric_name]

    return ratings
