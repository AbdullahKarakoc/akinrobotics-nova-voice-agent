# Runs scripts/evaluate.py for several LLM / knowledge-mode configurations.
# Each configuration restarts the server with different environment variables.
#   powershell -ExecutionPolicy Bypass -File scripts/eval_matrix.ps1 [-Configs "qwen2.5:3b|rag,qwen2.5:3b|full"]
param(
    [string]$Configs = "qwen2.5:3b|rag,qwen2.5:3b|full,qwen2.5:1.5b|rag,qwen2.5:1.5b|full,qwen3:1.7b|rag,qwen3:1.7b|full",
    [int]$Port = 8000,
    [string]$Suffix = ""   # appended to report names, e.g. "_gpu"
)
$env:PORT = "$Port"

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
New-Item -ItemType Directory -Force (Join-Path $root "reports") | Out-Null

foreach ($config in $Configs.Split(",")) {
    $model, $mode = $config.Split("|")
    $env:LLM_MODEL = $model
    $env:KNOWLEDGE_MODE = $mode
    $env:LLM_NUM_CTX = if ($mode -eq "full") { "6144" } else { "3072" }
    if ($model -like "qwen3*") { $env:LLM_THINK = "false" } else { Remove-Item Env:LLM_THINK -ErrorAction SilentlyContinue }
    $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

    $tag = (($model -replace "^hf\.co/[^/]+/", "") -replace "[:./]", "-") + "_" + $mode + $Suffix
    $log = Join-Path $root "reports\server_$tag.log"
    Write-Host "=== $model / $mode ==="
    $server = Start-Process -FilePath $python -ArgumentList "-m", "app" -WorkingDirectory $root `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err" -PassThru -WindowStyle Hidden

    $ready = $false
    for ($i = 0; $i -lt 120; $i++) {
        Start-Sleep -Seconds 5
        try { $h = Invoke-RestMethod "http://localhost:$Port/health" -TimeoutSec 3; if ($h.llm_ready) { $ready = $true; break } } catch {}
    }
    if ($ready) {
        & $python -m scripts.evaluate --url "http://localhost:$Port" --out (Join-Path $root "reports\eval_$tag.json")
    } else {
        Write-Host "server did not become ready"
    }
    Stop-Process -Id $server.Id -Force
    Start-Sleep -Seconds 3
}
