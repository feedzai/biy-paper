from pathlib import Path

from PIL import Image

# Source: Vision Language Models are Biased (https://arxiv.org/abs/2505.23941)
Q1_COUNT_CLUSTERS_PROMPT = (
    "How many clusters are there in the scatterplot? Answer with a number in curly brackets, e.g., {9}."
)

# Source:
# - Re-Reading Improves Reasoning in Large Language Models (https://arxiv.org/abs/2309.06275)
# - https://learnprompting.org/docs/advanced/zero_shot/re_reading
# - https://github.com/codelion/optillm/blob/v0.1.20/optillm/reread.py
Q1_RE2_COUNT_CLUSTERS_PROMPT = f"How many clusters are there in the scatterplot?\nRead the question again: How many clusters are there in the scatterplot?\nAnswer with a number in curly brackets, e.g., {9}."

# More alternatives: point masses, data collections, etc.
# https://en.wikipedia.org/wiki/Point_group
Q1_ALT_COUNT_CLUSTERS_PROMPT = (
    "How many point groups are there in the scatterplot? Answer with a number in curly brackets, e.g., {9}."
)

Q2_COUNT_CLUSTERS_PROMPT = "Count the clusters in this scatterplot. Answer with a number in curly brackets, e.g., {9}."

Q3_COUNT_CLUSTERS_PROMPT = "Is this a scatterplot with 4 clusters? Answer in curly brackets, e.g., {Yes} or {No}."

# Source:
# - MiMo-VL Technical Report (https://arxiv.org/abs/2506.03569)
# - https://github.com/XiaomiMiMo/lmms-eval/blob/41fd2457f8fc989f064d2f62ca9154582ac8fa1e/lmms_eval/tasks/countbench/countbench.yaml
# - https://github.com/XiaomiMiMo/lmms-eval/tree/41fd2457f8fc989f064d2f62ca9154582ac8fa1e/lmms_eval/tasks/pixmo_count
# - https://huggingface.co/datasets/vikhyatk/CountBenchQA
COUNT_BENCH_COUNT_CLUSTERS_PROMPT = (
    "How many clusters are there in the scatterplot?\nPlease give the answer in a single number."
)

COUNT_BENCH_BOXED_COUNT_CLUSTERS_PROMPT = (
    "How many clusters are there in the scatterplot?\nPut your final answer in \\boxed{} using a single number."
)

COUNT_BENCH_BOXED_MIMO_COUNT_CLUSTERS_PROMPT = "How many clusters are there in the scatterplot?\nPlease count the objects by grounding. Put your answer in \\boxed{}."

# Source: EarthGPT: A Universal Multi-modal Large Language Model for Multi-sensor Image Comprehension in Remote Sensing Domain (https://arxiv.org/abs/2401.16822)
HBB_CLUSTERS_PROMPT = "Detect all clusters in this scatterplot image and describe using horizontal bounding boxes."

# Source: https://huggingface.co/AIDC-AI/Ovis2-1B/blob/b5c50bc2836fd46a6cd0feb39269eeb5968fac1d/README.md#usage
OVIS_ZERO_SHOT_COT_SUFFIX = (
    "Provide a step-by-step solution to the problem, and conclude with 'the answer is' followed by the final solution."
)

# Source:
# - Point, Detect, Count: Multi-Task Medical Image Understanding with Instruction-Tuned Vision-Language Models (https://arxiv.org/abs/2505.16647)
# - https://github.com/simula/PointDetectCount/blob/8c139ad9a3b22c974a57ca498f6623cc7aea931f/create_datasetJSON.py#L19
PDC_SUFFIX = "Please say 'This isn't in the image.' if it is not in the image."

# Source: https://pyimagesearch.com/2025/06/09/object-detection-and-visual-grounding-with-qwen-2-5/
PY_IMAGE_SEARCH_COUNT_CLUSTERS_PROMPT = (
    "Detect all clusters in the scatterplot and return their locations in the form of coordinates."
)
PY_IMAGE_SEARCH_SUFFIX = (
    "Return their locations in the form of coordinates in the format {'bbox_2d': [x1, y1, x2, y2]}."
)

# Source: https://huggingface.co/google/gemma-3n-E2B-it/blob/ec2b9d6e5490d284771a8262bf9b4564a1cb4868/README.md#running-the-model-on-a-single-gpu
DEBUG_PROMPT = "Describe this image in detail."

# Source: PUB: Plot Understanding Benchmark and Dataset for Evaluating Large Language Models on Synthetic Visual Data Interpretation (https://arxiv.org/abs/2409.02617)
PUB_HBB_CLUSTERS_PROMPT = """
Where is each cluster located? Please provide the coordinates of the areas occupied by distinct clusters. Give your answer based on points locations. Suggested clustering might be incorrect. As answer give an array [[x_lower, y_lower], [x_upper, y_upper]] with the coordinates of each cluster, following the format:

```json
{
    '<cluster index>': [[<lower bound x>, <lower bound y>], [<upper bound y>, <upper bound x>]],
}
```

Your response should contain only the json data.
""".strip()

PUB_CLUSTER_CENTERS_PROMPT = """
Where are located the centers of the clusters? As answer give an array [x, y] with the coordinates of for each cluster, following the format:

```json
{
    "<index>": [x, y],
}
```

Your response should contain only the json data.
""".strip()


# Source:
# - Good at captioning, bad at counting: Benchmarking GPT-4V on Earth observation data (https://arxiv.org/abs/2401.17600)
# - https://github.com/Earth-Intelligence-Lab/vleo-bench/blob/bc5ff1506aa973c4a5c434a52f35f67b0d43e185/src/datasets/dior_rsvg.py#L97-L102
def get_vleo_hbb_clusters_prompt(image: Path) -> str:
    with Image.open(image) as im:
        w, h = im.size

    return f"You are given an {w} x {h} scatterplot image. Identify the extent of each cluster in the format of [xmin, ymin, xmax, ymax], where the top-left coordinate is (x_min, y_min) and the bottom-right coordinate is (x_max, y_max). You should answer with a list of coordinates without further explanation."


# Source:
# - UIShift: Enhancing VLM-based GUI Agents through Self-supervised Reinforcement Learning (https://arxiv.org/abs/2505.12493)
# - InfiX-ai/InfiGUI-R1-3B (https://huggingface.co/InfiX-ai/InfiGUI-R1-3B)
def get_ui_shift_od_prompt(image: Path) -> str:
    with Image.open(image) as im:
        w, h = im.size

    return f'The scatterplot image resolution is {w}x{h}. Point to each outlier, output their coordinates using JSON format: [{{"point_2d": [x, y]}}]'


# Source: Customizing Visual-Language Foundation Models for Multi-modal Anomaly Detection and Reasoning (https://arxiv.org/abs/2403.11083)
CUSTOM_VLM_OD_PROMPT = (
    "Please determine whether the image contains anomalies or outliner points. If yes, give a specific reason."
)

# Source:
# - https://simedw.com/2025/07/10/gemini-bounding-boxes/
# - https://news.ycombinator.com/item?id=44520757
SIMEDW_HBB_CLUSTERS_PROMPT = """
Look carefully at this image and detect ALL visible clusters, including small ones.

IMPORTANT: Focus on finding as many clusters as possible, even if they are small, distant, or partially visible.
Make sure that the bounding box is as tight as possible.

For each detected cluster, provide:
- "confidence": how certain you are (0.0 to 1.0)
- "box_2d": bounding box [ymin, xmin, ymax, xmax] normalized 0-1000

Detect everything you can see that matches valid clusters. Don't be conservative - include clusters even if you're only moderately confident.

Return as JSON array:
[
  {
    "confidence": 0.95,
    "box_2d": [100, 200, 300, 400]
  },
  {
    "confidence": 0.80,
    "box_2d": [50, 150, 250, 350]
  }
]
""".strip()

SIMEDW_ALT_HBB_CLUSTERS_PROMPT = """
Look carefully at this image and detect ALL visible clusters, including small ones.

IMPORTANT: Focus on finding as many clusters as possible, even if they are small, distant, or partially visible.
Make sure that the bounding box is as tight as possible.

For each detected cluster, provide:
- "confidence": how certain you are (0.0 to 1.0)
- "box_2d": bounding box [xmin, ymin, xmax, ymax] normalized 0-1000

Detect everything you can see that matches valid clusters. Don't be conservative - include clusters even if you're only moderately confident.

Return as JSON array:
[
  {
    "confidence": 0.95,
    "box_2d": [200, 100, 400, 300]
  },
  {
    "confidence": 0.80,
    "box_2d": [150, 50, 350, 250]
  }
]
""".strip()

# Source:
# - How Well Does GPT-4o Understand Vision? Evaluating Multimodal Foundation Models on Standard Computer Vision Tasks (https://fm-vision-evals.epfl.ch/)
# - https://github.com/EPFL-VILAB/fm-vision-evals/blob/d278c670da4d7866b3df815bef06e7f351795d9f/taskit/tasks/object.py#L686-L796
VC_TASKS_NAIVE_1_HBB_CLUSTERS_PROMPT = """
Locate each cluster and represent each location of the region. Regions are represented by [x1, y1, x2, y2] coordinates. x1 x2 are the left-most and right-most points of the region, normalized into 0 to 1000, where 0 is the left and 1000 is the right. y1 y2 are the top-most and bottom-most points of the region, normalized into 0 to 1000, where 0 is the top and 1000 is the bottom. The output should be in JSON format, structured as follows: [{"coordinates": [x1, y1, x2, y2]}, ...]
""".strip()

VC_TASKS_NAIVE_3_HBB_CLUSTERS_PROMPT = """
Your task is to precisely locate the cluster(s) described below in the given scatterplot and represent their position(s) using normalized bounding box(es).

<object>cluster(s)</object>

### Bounding Box Representation:
Each bounding box should be represented as [x1, y1, x2, y2] where:
- **x1** is the left-most point of the cluster
- **x2** is the right-most point of the cluster
- **y1** is the top-most point of the cluster
- **y2** is the bottom-most point of the cluster
All coordinates must be normalized according to the following system:
- Horizontal coordinates (x1, x2) are scaled between 0 and 1000, where 0 represents the left edge of the plot and 1000 represents the right edge.
- Vertical coordinates (y1, y2) are scaled between 0 and 1000, where 0 represents the top edge of the plot and 1000 represents the bottom edge.

### Steps for Accurate Detection:
1. **Analyze Cluster Boundaries**: Start by identifying the full extent of each cluster, including any scattered points that belong to the cluster. Ensure that all data points belonging to each cluster are contained within their respective bounding boxes.
2. **Normalize Coordinates**: Carefully scale the coordinates of each bounding box to the 0 to 1000 range, ensuring consistency and precision.
3. **Refine for Accuracy**: Double-check to ensure each bounding box is correctly oriented and that all cluster points are fully contained.

### Output Format:
Once you have determined the bounding box(es), provide the result in the following JSON format:
{
    "reasoning_steps": [
        "Step 1: ...",
        "Step 2: ...",
        "Step 3: ..."
    ],
    "clusters": [
        {
            "cluster_id": 1,
            "coordinates": [x1, y1, x2, y2]
        },
        {
            "cluster_id": 2,
            "coordinates": [x1, y1, x2, y2]
        }
    ]
}

### Additional Considerations:
- Ensure that each bounding box fully captures all data points belonging to that cluster without leaving any points outside the box.
- Avoid unnecessary padding around clusters. Keep each bounding box as tight as possible while still enclosing all cluster points.
- If clusters overlap or have ambiguous boundaries, use your best judgment to separate them based on density and spacing patterns.
- The normalized coordinates should accurately represent each cluster's position within the scatterplot's full dimensions.
""".strip()

VC_TASKS_NAIVE_3_ALT_HBB_CLUSTERS_PROMPT = """
Your task is to precisely locate and identify all clusters in the given scatterplot and represent their positions using normalized bounding boxes.

<object>clusters in scatterplot</object>

### Cluster Detection Objective:
Identify distinct groups or clusters of data points in the scatterplot. A cluster is defined as a dense grouping of data points that are clearly separated from other groups by areas of lower point density.

### Bounding Box Representation:
Each cluster should be represented as [x1, y1, x2, y2] where:
- **x1** is the left-most point of the cluster
- **x2** is the right-most point of the cluster
- **y1** is the top-most point of the cluster
- **y2** is the bottom-most point of the cluster
All coordinates must be normalized according to the following system:
- Horizontal coordinates (x1, x2) are scaled between 0 and 1000, where 0 represents the left edge of the plot and 1000 represents the right edge.
- Vertical coordinates (y1, y2) are scaled between 0 and 1000, where 0 represents the top edge of the plot and 1000 represents the bottom edge.

### Steps for Accurate Cluster Detection:
1. **Identify Dense Regions**: Scan the scatterplot to identify areas with high concentrations of data points that form visually distinct groups.
2. **Assess Cluster Separation**: Determine clear boundaries between clusters by identifying gaps or areas of lower point density that separate distinct groups.
3. **Define Cluster Boundaries**: For each identified cluster, determine the extent that encompasses all points belonging to that cluster, including any outlying points that are clearly part of the group.
4. **Normalize Coordinates**: Carefully scale the coordinates of each bounding box to the 0 to 1000 range, ensuring consistency and precision.
5. **Validate Completeness**: Ensure all significant clusters have been identified and that overlapping clusters are properly distinguished.

### Output Format:
Once you have identified all clusters, provide the result in the following JSON format:
{
    "reasoning_steps": [
        "Step 1: Identified [number] distinct clusters based on point density and separation...",
        "Step 2: Defined boundaries for each cluster...",
        "Step 3: Normalized coordinates for all clusters...",
        "Step 4: Validated cluster completeness and separation..."
    ],
    "clusters": [
        {
            "cluster_id": 1,
            "coordinates": [x1, y1, x2, y2],
            "description": "Brief description of cluster characteristics"
        },
        {
            "cluster_id": 2,
            "coordinates": [x1, y1, x2, y2],
            "description": "Brief description of cluster characteristics"
        }
    ],
    "total_clusters": [number]
}

### Additional Considerations:
- **Minimum Cluster Size**: Only identify clusters that contain a meaningful number of data points.
- **Cluster Density**: Focus on areas where points are noticeably denser than the surrounding regions.
- **Boundary Tightness**: Keep bounding boxes as tight as possible while ensuring all cluster points are included.
- **Overlapping Clusters**: If clusters appear to overlap, use your best judgment to separate them based on the densest concentrations of points.
- **Noise vs. Clusters**: Distinguish between random scattered points (noise) and true clusters with clear grouping patterns.
- **Visual Separation**: Clusters should be visually distinct from one another with clear gaps or separation in the data distribution.
- **Single Points**: Individual isolated points should not be considered clusters unless they form a meaningful pattern with nearby points.

### Edge Cases:
- If no clear clusters are present (uniformly distributed points), return an empty clusters array.
- If only one cluster is present, still use the array format with a single cluster object.
- If clusters have irregular shapes, ensure the bounding box captures the full extent of all points in the cluster.
""".strip()

VC_TASKS_NAIVE_5_HBB_CLUSTERS_PROMPT = """
You are an AI assistant tasked with detecting objects in images. Your goal is to locate each of the specific objects within an image and represent their locations using normalized coordinates.

Your task is to locate each of the following objects in the image:
<object>cluster</object>

When representing the location of each object, use the following coordinate system:
- The coordinates should be in the format [y_min, x_min, y_max, x_max]
- **y_min** and **y_max** represent the top-most and bottom-most points of the region
- **x_min** and **x_max** represent the left-most and right-most points of the region

Your output should be in a JSON, structured as follows:
[
    {
        "coordinates": [y_min, x_min, y_max, x_max],
        ...
    }
]
""".strip()

# Source:
# - Chain of Draft: Thinking Faster by Writing Less (https://arxiv.org/abs/2502.18600)
# - https://learnprompting.org/docs/advanced/thought_generation/chain-of-draft
# - https://github.com/sileix/chain-of-draft/tree/a7dbf5dea808b1aa1e12f7a90ea573321581df78/configs
COT_COUNT_CLUSTERS_SYSTEM_PROMPT = "Think step by step to answer the following question.\nReturn the number of clusters, for example 9, at the end of the response after a separator ####."

COD_COUNT_CLUSTERS_SYSTEM_PROMPT = "Think step by step, but only keep minimum draft for each thinking step, with 5 words at most.\nReturn the number of clusters, for example 9, at the end of the response after a separator ####."

# Source:
# - https://huggingface.co/spaces/sergiopaniego/vlm_object_understanding
# - https://huggingface.co/spaces/sergiopaniego/vlm_object_understanding/blob/99a53c4d226a4b4d445cfbc5ec21be3d70ee7fde/app.py#L299
PANIEGO_HBB_CLUSTERS_PROMPT = "Detect all clusters in the scatterplot and return their locations."

CLUSTER_POINTS_PROMPT_TEXT_V1 = 'Detect all clusters in the scatterplot. For each detected cluster, provide its center point in normalized coordinates (x, y), where (0, 0) is the top-left corner and (1000, 1000) is the bottom-right corner of the image. Answer with a JSON object, e.g., {"cluster_centers": [[250, 40], [700, 500]]}.'

# Source:
# - https://ai.google.dev/gemini-api/docs/image-understanding#object-detection
# - https://developers.googleblog.com/en/conversational-image-segmentation-gemini-2-5/

GEMINI_HBB_CLUSTERS_PROMPT = "Detect the all of the prominent clusters in the scatterplot. The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000."

GEMINI_CLUSTER_SEGMENTATION_PROMPT = 'Give the segmentation masks for the clusters.\nOutput a JSON list of segmentation masks where each entry contains the 2D bounding box in the key "box_2d" and the segmentation mask in key "mask".'
