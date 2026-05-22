Param()

$outDir = "outputs"
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

$notebookList = "notebooks_to_run.txt"
if (-not (Test-Path $notebookList)) { Write-Error "Notebook list $notebookList not found"; exit 1 }

Get-Content $notebookList | ForEach-Object {
    $nb = $_.Trim()
    if ([string]::IsNullOrWhiteSpace($nb)) { return }
    Write-Host "Running $nb"
    & papermill $nb "$outDir\$([IO.Path]::GetFileNameWithoutExtension($nb))-executed.ipynb"
}

Write-Host "All notebooks executed. Outputs in $outDir"
