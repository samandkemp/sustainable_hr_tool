param(
    [string]$input = "data/raw",
    [string]$modeldir = "scratch_models",
    [switch]$clean
)

Write-Host "Running validation on: $input -> results saved to: $modeldir"
python -m src.train --input $input --model-dir $modeldir --validate-only

if ($clean) {
    if (Test-Path $modeldir) {
        Write-Host "Removing directory: $modeldir"
        Remove-Item -Recurse -Force $modeldir
    } else {
        Write-Host "No directory to remove: $modeldir"
    }
}

Write-Host "Done. The scratch script is disposable; remove the `scratch` folder when finished."
