import evaluate
import wandb

from datasets import load_dataset
from googleapiclient import discovery
from keys import PERSPECTIVE_API_KEY
from transformers import pipeline, AutoTokenizer

from utils.utils import translate_model_kwargs, get_random_prompts


def generate_and_evaluate(
    dataset_name: str,
    metric: str,
    model_cfg,
    num_samples,
    num_epochs: int = 1,
    comp_model_cfg=None,
    save_continuations=True,
    save_prompts=False,
    seed=0,
    use_wandb=True,
):
    """ """

    prompt_dataset = load_dataset(dataset_name, split="train")

    # wandb only logs strings, floats, ... so need to modify torch_dtype
    model_kwargs = translate_model_kwargs(model_cfg["model_kwargs"])
    gen_kwargs = model_cfg["generation_kwargs"]

    generator = pipeline(
        "text-generation",
        model=model_cfg["model_id"],
        # device=args.device, # no device param if using accelerate (load_in_4bit=True)
        model_kwargs=model_kwargs,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_id"])

    # if you wand to get model comparison on the same prompts
    if comp_model_cfg:
        comp_model_kwargs = translate_model_kwargs(comp_model_cfg["model_kwargs"])
        comp_gen_kwargs = comp_model_cfg["generation_kwargs"]

        comp_generator = pipeline(
            "text-generation",
            model=comp_model_cfg["model_id"],
            # device=args.device, # no device param if using accelerate (load_in_4bit=True)
            model_kwargs=comp_model_kwargs,
        )

    sample = get_random_prompts(prompt_dataset, num_examples=num_samples)
    prompts = [p["text"] for p in sample["prompt"]]

    results = {}

    all_data_table = wandb.Table(columns=["epoch", "step", "ratings"])

    for epoch in range(num_epochs):
        model_continuations = []
        for prompt in prompts:
            generation = generator(
                prompt, pad_token_id=tokenizer.eos_token_id, **gen_kwargs
            )

            continuation = generation[0]["generated_text"].replace(prompt, "")
            model_continuations.append(continuation)

        ratings = metric.compute(predictions=model_continuations)

        results[epoch] = {}
        results[epoch]["generations"] = model_continuations
        results[epoch]["ratings"] = ratings[args.metric]

        for i, rating in enumerate(ratings[args.metric]):
            all_data_table.add_data(epoch, i, rating)

        # upload json to wandb
        log_scores(results)

        file_name = f"{args.metric}_scores.json"

        with open(file_name, "w") as file:
            json.dump(results, file, indent=4)

        print(f"Saved scores epoch {epoch} out of {args.epochs}.")

    run.log(
        {
            "Ratings Histogram": wandb.plot.histogram(
                all_data_table,
                "ratings",
                title=f"{args.metric} ratings",
                # group_by="epoch"
            )
        }
    )

    print("Finished evaluating.")

    wandb.finish()

    return rating


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
