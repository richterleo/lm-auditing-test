import evaluate


from transformers import pipeline, AutoTokenizer


def generate_and_evaluate(model_id, model_kwargs, generation_kwargs, metric, prompt):
    """ """

    generator = pipeline(
        "text-generation",
        model=model_id,
        # device=args.device, # no device param if using accelerate (load_in_4bit=True)
        model_kwargs=model_kwargs,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    generation = generator(
        prompt, pad_token_id=tokenizer.eos_token_id, **generation_kwargs
    )

    continuation = generation[0]["generated_text"].replace(prompt, "")

    rating = eval_on_metric(metric, continuation)

    return rating


def eval_on_metric(metric, continuations):
    """ """

    if isinstance(metric, str):
        metric_name = metric
        metric = evaluate.load(metric)
        rating_dict = metric.compute(predictions=continuations)
        ratings = rating_dict[metric_name]

    return ratings
