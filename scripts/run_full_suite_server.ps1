param(
    [string]$DataRoot = "data",
    [string]$OutputRoot = "outputs/server_2026_06",
    [int]$CocoMaxImages = 500,
    [string]$CocoSampling = "random",
    [int]$CocoSeed = 308,
    [int]$CocoTopK = 100,
    [double]$OwlScoreThreshold = 0.01,
    [double]$OwlNmsThreshold = -1.0,
    [double]$GroundingBoxThreshold = 0.20,
    [double]$GroundingTextThreshold = 0.20,
    [double]$YoloConfidence = 0.001,
    [double]$YoloIouThreshold = 0.7,
    [int]$ImageSize = 640,
    [string]$RefCocoSplit = "val",
    [int]$RefCocoMaxRows = 0,
    [string]$RefCocoExpressionMode = "all",
    [int]$SubsetWorkers = 8,
    [string]$OwlThresholds = "0.01,0.03,0.05,0.10,0.20",
    [string]$GroundingThresholds = "0.20,0.25,0.30,0.35,0.40",
    [string]$YoloThresholds = "0.001,0.01,0.05,0.10,0.25"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Invoke-Python {
    param(
        [string[]]$Arguments
    )

    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: python $($Arguments -join ' ')"
    }
}

$CocoDataDir = Join-Path $DataRoot "coco"
$RefCocoDataDir = Join-Path $DataRoot "refcoco"
$CocoTag = "{0}_{1}_seed{2}" -f $CocoMaxImages, $CocoSampling, $CocoSeed
$RefCocoTag = "{0}_{1}_{2}" -f $RefCocoSplit, ($(if ($RefCocoMaxRows -le 0) { "full" } else { "rows$RefCocoMaxRows" })), $RefCocoExpressionMode

$env:DATA_ROOT = $DataRoot
$env:OUTPUT_ROOT = $OutputRoot
$env:COCO_MAX_IMAGES = "$CocoMaxImages"
$env:COCO_SAMPLING = $CocoSampling
$env:COCO_SEED = "$CocoSeed"
$env:COCO_TOP_K = "$CocoTopK"
$env:OWL_SCORE_THRESHOLD = "$OwlScoreThreshold"
$env:OWL_NMS_THRESHOLD = "$OwlNmsThreshold"
$env:GROUNDING_BOX_THRESHOLD = "$GroundingBoxThreshold"
$env:GROUNDING_TEXT_THRESHOLD = "$GroundingTextThreshold"
$env:YOLO_CONFIDENCE = "$YoloConfidence"
$env:YOLO_IOU_THRESHOLD = "$YoloIouThreshold"
$env:IMAGE_SIZE = "$ImageSize"
$env:REFCOCO_SPLIT = $RefCocoSplit
$env:REFCOCO_MAX_ROWS = "$RefCocoMaxRows"
$env:REFCOCO_EXPRESSION_MODE = $RefCocoExpressionMode

Write-Host "Preparing COCO subset..."
Invoke-Python @(
    "scripts\prepare_coco_subset.py",
    "--data-dir", $CocoDataDir,
    "--max-images", "$CocoMaxImages",
    "--sampling", $CocoSampling,
    "--seed", "$CocoSeed",
    "--workers", "$SubsetWorkers"
)

Write-Host "Running OWL-ViT COCO evaluation..."
Invoke-Python @(
    "scripts\evaluate_coco_owlvit.py",
    "--data-dir", $CocoDataDir,
    "--output-dir", (Join-Path $OutputRoot ("coco_owlvit_eval_{0}" -f $CocoTag)),
    "--max-images", "$CocoMaxImages",
    "--sampling", $CocoSampling,
    "--seed", "$CocoSeed",
    "--score-threshold", "$OwlScoreThreshold",
    "--nms-threshold", "$OwlNmsThreshold",
    "--top-k", "$CocoTopK"
)

Write-Host "Running Grounding DINO COCO evaluation..."
Invoke-Python @(
    "scripts\evaluate_coco_grounding_dino.py",
    "--data-dir", $CocoDataDir,
    "--output-dir", (Join-Path $OutputRoot ("coco_grounding_dino_eval_{0}" -f $CocoTag)),
    "--max-images", "$CocoMaxImages",
    "--sampling", $CocoSampling,
    "--seed", "$CocoSeed",
    "--box-threshold", "$GroundingBoxThreshold",
    "--text-threshold", "$GroundingTextThreshold",
    "--top-k", "$CocoTopK"
)

Write-Host "Running YOLO-World COCO evaluation..."
Invoke-Python @(
    "scripts\evaluate_coco_yolo_world.py",
    "--data-dir", $CocoDataDir,
    "--output-dir", (Join-Path $OutputRoot ("coco_yolo_world_eval_{0}" -f $CocoTag)),
    "--max-images", "$CocoMaxImages",
    "--sampling", $CocoSampling,
    "--seed", "$CocoSeed",
    "--confidence", "$YoloConfidence",
    "--iou-threshold", "$YoloIouThreshold",
    "--top-k", "$CocoTopK",
    "--image-size", "$ImageSize"
)

Write-Host "Running RefCOCO evaluation..."
Invoke-Python @(
    "scripts\evaluate_refcoco.py",
    "--model-type", "owlvit",
    "--data-dir", $RefCocoDataDir,
    "--split", $RefCocoSplit,
    "--max-rows", "$RefCocoMaxRows",
    "--expression-mode", $RefCocoExpressionMode,
    "--refresh-manifest",
    "--output-dir", (Join-Path $OutputRoot ("refcoco_owlvit_eval_{0}" -f $RefCocoTag))
)
Invoke-Python @(
    "scripts\evaluate_refcoco.py",
    "--model-type", "grounding-dino",
    "--data-dir", $RefCocoDataDir,
    "--split", $RefCocoSplit,
    "--max-rows", "$RefCocoMaxRows",
    "--expression-mode", $RefCocoExpressionMode,
    "--output-dir", (Join-Path $OutputRoot ("refcoco_grounding_dino_eval_{0}" -f $RefCocoTag))
)
Invoke-Python @(
    "scripts\evaluate_refcoco.py",
    "--model-type", "yolo-world",
    "--data-dir", $RefCocoDataDir,
    "--split", $RefCocoSplit,
    "--max-rows", "$RefCocoMaxRows",
    "--expression-mode", $RefCocoExpressionMode,
    "--output-dir", (Join-Path $OutputRoot ("refcoco_yolo_world_eval_{0}" -f $RefCocoTag)),
    "--image-size", "$ImageSize"
)

Write-Host "Running threshold sensitivity analyses..."
Invoke-Python @(
    "scripts\evaluate_threshold_sensitivity.py",
    "--annotation-file", (Join-Path $CocoDataDir "annotations\instances_val2017.json"),
    "--predictions", (Join-Path $OutputRoot ("coco_owlvit_eval_{0}\coco_predictions.json" -f $CocoTag)),
    "--metrics", (Join-Path $OutputRoot ("coco_owlvit_eval_{0}\metrics.json" -f $CocoTag)),
    "--thresholds", $OwlThresholds,
    "--output", (Join-Path $OutputRoot "threshold_sensitivity\owlvit.json")
)
Invoke-Python @(
    "scripts\evaluate_threshold_sensitivity.py",
    "--annotation-file", (Join-Path $CocoDataDir "annotations\instances_val2017.json"),
    "--predictions", (Join-Path $OutputRoot ("coco_grounding_dino_eval_{0}\coco_predictions.json" -f $CocoTag)),
    "--metrics", (Join-Path $OutputRoot ("coco_grounding_dino_eval_{0}\metrics.json" -f $CocoTag)),
    "--thresholds", $GroundingThresholds,
    "--output", (Join-Path $OutputRoot "threshold_sensitivity\grounding_dino.json")
)
Invoke-Python @(
    "scripts\evaluate_threshold_sensitivity.py",
    "--annotation-file", (Join-Path $CocoDataDir "annotations\instances_val2017.json"),
    "--predictions", (Join-Path $OutputRoot ("coco_yolo_world_eval_{0}\coco_predictions.json" -f $CocoTag)),
    "--metrics", (Join-Path $OutputRoot ("coco_yolo_world_eval_{0}\metrics.json" -f $CocoTag)),
    "--thresholds", $YoloThresholds,
    "--output", (Join-Path $OutputRoot "threshold_sensitivity\yolo_world.json")
)

Write-Host "Running OWL-ViT NMS ablation on the 100-image diagnostic subset..."
Invoke-Python @(
    "scripts\evaluate_coco_owlvit.py",
    "--data-dir", $CocoDataDir,
    "--output-dir", (Join-Path $OutputRoot "coco_owlvit_eval_100_nms"),
    "--max-images", "100",
    "--sampling", $CocoSampling,
    "--seed", "$CocoSeed",
    "--score-threshold", "$OwlScoreThreshold",
    "--nms-threshold", "0.5",
    "--top-k", "$CocoTopK"
)

Write-Host "Regenerating report figures..."
Invoke-Python @(
    "scripts\generate_report_figures.py"
)

Write-Host "Compiling report..."
Push-Location report
try {
    & pdflatex example_paper.tex
    if ($LASTEXITCODE -ne 0) { throw "pdflatex failed" }
    & bibtex example_paper
    if ($LASTEXITCODE -ne 0) { throw "bibtex failed" }
    & pdflatex example_paper.tex
    if ($LASTEXITCODE -ne 0) { throw "pdflatex failed" }
    & pdflatex example_paper.tex
    if ($LASTEXITCODE -ne 0) { throw "pdflatex failed" }
}
finally {
    Pop-Location
}

Write-Host "Done."
