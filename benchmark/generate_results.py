import json
import re
from urllib.request import urlopen

import pandas as pd
from gaveta.files import ensure_dir
from google.genai.types import GenerateContentResponse
from langchain_core.utils.json import parse_json_markdown
from loguru import logger
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, ValidationError

from constants import ALL_RESULTS, ALL_VAL, GOOGLE_RAW_RESULTS, INPUT, OPEN_AI_RAW_RESULTS
from data_models import BatchOutputFile, CustomId
from utils import denormalize_xy_point, parse_custom_id, parse_example_id, process_xyxy_bbox


class FinalResult(CustomId):
    chart_design: str
    dataset_gen: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    raw_response: str | None
    pred_response: str | None


def concat_h(im1: Image.Image, im2: Image.Image) -> Image.Image:
    new_im = Image.new("RGB", (im1.width + im2.width, im1.height))

    new_im.paste(im1, (0, 0))
    new_im.paste(im2, (im1.width, 0))

    return new_im


def extract_count(response: str | None) -> str | None:
    if not response or not response.strip():
        return None

    pattern = r"\{(\d+)\}"

    all_matches = re.findall(pattern, response)

    if all_matches:
        return str(all_matches[-1])

    return None


def extract_cluster_bboxes(response: str | None, encoded_image: str) -> str | None:
    if not response:
        return None

    try:
        extracted_bboxes = parse_json_markdown(response, parser=json.loads)
        pred_bboxes = [process_xyxy_bbox(encoded_image, bbox) for bbox in extracted_bboxes["clusters"]]

        return json.dumps(pred_bboxes)
    except (json.decoder.JSONDecodeError, KeyError, TypeError, IndexError):
        return None


def extract_cluster_points(response: str | None, encoded_image: str) -> str | None:
    if not response:
        return None

    try:
        extracted_points = parse_json_markdown(response, parser=json.loads)
        pred_bboxes = [denormalize_xy_point(encoded_image, point) for point in extracted_points["cluster_centers"]]

        return json.dumps(pred_bboxes)
    except (json.decoder.JSONDecodeError, KeyError, TypeError):
        return None


def extract_outlier_points(response: str | None, encoded_image: str) -> str | None:
    if not response:
        return None

    try:
        extracted_points = parse_json_markdown(response, parser=json.loads)
        pred_bboxes = [denormalize_xy_point(encoded_image, point) for point in extracted_points["outliers"]]

        return json.dumps(pred_bboxes)
    except (json.decoder.JSONDecodeError, KeyError, TypeError):
        return None


def process_raw_open_ai_results(dataset: pd.DataFrame) -> list[FinalResult]:
    final_results: list[FinalResult] = []

    for raw_results in OPEN_AI_RAW_RESULTS.glob("*.jsonl"):
        results = [BatchOutputFile.model_validate_json(line) for line in raw_results.read_text().splitlines()]

        for result in results:
            custom_id = parse_custom_id(result.custom_id)
            example_id = parse_example_id(custom_id["example_id"])

            pred_response: str | None = None

            input_tokens = result.response.body.usage.prompt_tokens if result.response.body.usage else 0
            output_tokens = result.response.body.usage.completion_tokens if result.response.body.usage else 0
            reasoning_tokens = (
                result.response.body.usage.completion_tokens_details.reasoning_tokens
                if result.response.body.usage
                and result.response.body.usage.completion_tokens_details
                and result.response.body.usage.completion_tokens_details.reasoning_tokens
                else 0
            )

            raw_response = result.response.body.choices[0].message.content

            if custom_id["prompt_id"] == "cluster_count" or custom_id["prompt_id"] == "outlier_count":
                pred_response = extract_count(raw_response)
            elif custom_id["prompt_id"] == "cluster_bboxes":
                encoded_image = dataset.loc[dataset["id"] == custom_id["example_id"], "image"].item()
                pred_response = extract_cluster_bboxes(raw_response, encoded_image)
            elif custom_id["prompt_id"] == "cluster_points":
                encoded_image = dataset.loc[dataset["id"] == custom_id["example_id"], "image"].item()
                pred_response = extract_cluster_points(raw_response, encoded_image)
            elif custom_id["prompt_id"] == "outlier_points":
                encoded_image = dataset.loc[dataset["id"] == custom_id["example_id"], "image"].item()
                pred_response = extract_outlier_points(raw_response, encoded_image)

            final_results.append(
                {
                    **custom_id,
                    "chart_design": example_id["chart_design"],
                    "dataset_gen": example_id["dataset_gen"],
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "raw_response": raw_response,
                    "pred_response": pred_response,
                }
            )

            logger.info("{example_id} processed", example_id=result.custom_id)

    return final_results


class GoogleBatchOutputFile(BaseModel):
    key: str
    response: GenerateContentResponse

    model_config = ConfigDict(extra="allow")


def parse_google_batch_output_line(line: str) -> GoogleBatchOutputFile:
    try:
        return GoogleBatchOutputFile.model_validate(json.loads(line))
    except ValidationError:
        logger.error(line)
        raise


def parse_google_custom_id(result: GoogleBatchOutputFile) -> CustomId:
    custom_id = parse_custom_id(result.key)

    if result.response.usage_metadata and result.response.usage_metadata.thoughts_token_count is None:
        return custom_id

    custom_id["model"] = f"{custom_id['model']}-reasoning"

    return custom_id


def process_raw_google_results(dataset: pd.DataFrame) -> list[FinalResult]:
    final_results: list[FinalResult] = []

    for raw_results in GOOGLE_RAW_RESULTS.glob("*.jsonl"):
        results = [parse_google_batch_output_line(line) for line in raw_results.read_text().splitlines()]

        for result in results:
            custom_id = parse_google_custom_id(result)
            example_id = parse_example_id(custom_id["example_id"])

            input_tokens = (
                result.response.usage_metadata.prompt_token_count
                if result.response.usage_metadata and result.response.usage_metadata.prompt_token_count
                else 0
            )
            output_tokens = (
                result.response.usage_metadata.candidates_token_count
                if result.response.usage_metadata and result.response.usage_metadata.candidates_token_count
                else 0
            )
            reasoning_tokens = (
                result.response.usage_metadata.thoughts_token_count
                if result.response.usage_metadata and result.response.usage_metadata.thoughts_token_count
                else 0
            )

            raw_response = (
                result.response.candidates[0].content.parts[0].text
                if result.response.candidates
                and result.response.candidates[0].content
                and result.response.candidates[0].content.parts
                else None
            )

            if custom_id["prompt_id"] == "cluster_count" or custom_id["prompt_id"] == "outlier_count":
                pred_response = extract_count(raw_response)
            elif custom_id["prompt_id"] == "cluster_bboxes":
                encoded_image = dataset.loc[dataset["id"] == custom_id["example_id"], "image"].item()
                pred_response = extract_cluster_bboxes(raw_response, encoded_image)
            elif custom_id["prompt_id"] == "cluster_points":
                encoded_image = dataset.loc[dataset["id"] == custom_id["example_id"], "image"].item()
                pred_response = extract_cluster_points(raw_response, encoded_image)
            elif custom_id["prompt_id"] == "outlier_points":
                encoded_image = dataset.loc[dataset["id"] == custom_id["example_id"], "image"].item()
                pred_response = extract_outlier_points(raw_response, encoded_image)

            final_results.append(
                {
                    **custom_id,
                    "chart_design": example_id["chart_design"],
                    "dataset_gen": example_id["dataset_gen"],
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "raw_response": raw_response,
                    "pred_response": pred_response,
                }
            )

            logger.info("{example_id} processed", example_id=result.key)

    return final_results


def process_raw_results(dataset: pd.DataFrame) -> pd.DataFrame:
    open_ai_results = process_raw_open_ai_results(dataset)
    google_results = process_raw_google_results(dataset)

    return pd.DataFrame.from_records(open_ai_results + google_results)


def generate_annotated_charts(results: pd.DataFrame) -> None:
    prompt_id_ignore = ["cluster_count", "outlier_count"]

    for example in (
        results.query("prompt_id not in @prompt_id_ignore").dropna(subset=["pred_response"]).itertuples(index=False)
    ):
        example_id = str(example.example_id)
        image_id = f"{example.model}+{example.prompt_strategy}+{example_id}"

        cluster_color = "#c7d2fe" if "dark" in example_id else "#4338ca"
        outlier_color = "#fde68a" if "dark" in example_id else "#b45309"

        parsed_values = json.loads(str(example.pred_response))

        with Image.open(urlopen(str(example.image))) as im:
            original = im.convert("RGB") if im.mode == "P" else im
            modified = original.copy()

            original_draw = ImageDraw.Draw(original)
            modified_draw = ImageDraw.Draw(modified)

            if example.prompt_id == "cluster_bboxes":
                for cluster_bbox in json.loads(str(example.response)):
                    original_draw.rectangle(cluster_bbox, outline=cluster_color, width=4)

                for cluster_bbox in parsed_values:
                    modified_draw.rectangle(cluster_bbox, outline=cluster_color, width=4)

                ensure_dir(ALL_VAL / "cluster_bboxes")
                concat_h(original, modified).save(ALL_VAL / "cluster_bboxes" / f"{image_id}.png")
            elif example.prompt_id == "cluster_points":
                for cluster_point in json.loads(str(example.response)):
                    original_draw.circle(cluster_point, radius=10, fill=cluster_color, outline="white", width=2)

                for cluster_point in parsed_values:
                    modified_draw.circle(cluster_point, radius=10, fill=cluster_color, outline="white", width=2)

                ensure_dir(ALL_VAL / "cluster_points")
                concat_h(original, modified).save(ALL_VAL / "cluster_points" / f"{image_id}.png")
            elif example.prompt_id == "outlier_points":
                for outlier_point in json.loads(str(example.response)):
                    original_draw.circle(outlier_point, radius=10, fill=cluster_color, outline="white", width=2)

                for outlier_point in parsed_values:
                    modified_draw.circle(outlier_point, radius=10, fill=outlier_color, outline="white", width=2)

                ensure_dir(ALL_VAL / "outlier_points")
                concat_h(original, modified).save(ALL_VAL / "outlier_points" / f"{image_id}.png")


if __name__ == "__main__":
    ensure_dir(ALL_RESULTS)

    dataset = pd.read_parquet(INPUT / "dataset.parquet", engine="pyarrow")

    results = process_raw_results(dataset)

    dataset = pd.melt(
        dataset,
        id_vars=["id", "image"],
        value_vars=["cluster_count", "outlier_count", "cluster_bboxes", "cluster_points", "outlier_points"],
        var_name="prompt_id",
        value_name="response",
    )

    results = results.merge(dataset, left_on=["example_id", "prompt_id"], right_on=["id", "prompt_id"])

    results["is_valid_response"] = results["pred_response"].notna()

    valid_results = results.groupby(["prompt_id"], as_index=False, dropna=False).agg(
        valid_pct=("is_valid_response", "mean")
    )

    logger.info("Percentage of valid responses per task:\n{valid_results}", valid_results=valid_results)
    logger.info("Percentage of missing values per column:\n{df}", df=results.isna().sum() / results.shape[0])

    valid_results = results.groupby(["model"], as_index=False, dropna=False).agg(
        valid_pct=("is_valid_response", "mean")
    )
    valid_results = valid_results.assign(valid_pct_report=(valid_results["valid_pct"] * 100).round(2)).sort_values(
        ["valid_pct"], ascending=False
    )

    logger.info("Percentage of valid responses per model:\n{valid_results}", valid_results=valid_results)
    valid_results.to_csv(ALL_RESULTS / "valid_model.csv", index=False)

    valid_results = results.groupby(["chart_design"], as_index=False, dropna=False).agg(
        valid_pct=("is_valid_response", "mean")
    )
    valid_results = valid_results.assign(valid_pct_report=(valid_results["valid_pct"] * 100).round(2)).sort_values(
        ["valid_pct"], ascending=False
    )

    logger.info("Percentage of valid responses per chart_design:\n{valid_results}", valid_results=valid_results)
    valid_results.to_csv(ALL_RESULTS / "valid_chart_design.csv", index=False)

    # generate_annotated_charts(results)

    results.drop(columns=["raw_response", "id", "image", "is_valid_response"]).astype({"response": "str"}).to_parquet(
        ALL_RESULTS / "results.parquet", index=False
    )

    results = pd.read_parquet(ALL_RESULTS / "results.parquet", engine="pyarrow")
    logger.info("Percentage of missing values per column:\n{df}", df=results.isna().sum() / results.shape[0])
