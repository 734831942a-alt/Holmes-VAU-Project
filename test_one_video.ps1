param(
    [Parameter(Mandatory = $true)]
    [string]$VideoPath,

    [string]$ModelPath = ".\ckpts\HolmesVAU_lora_debug",
    [string]$BaseModelPath = ".\ckpts\HolmesVAU-2B",
    [string]$CodeRoot = ".\internvl_chat\internvl",
    [string]$SamplerPath = ".\holmesvau\ATS\anomaly_scorer.pth",
    [string]$PythonExe = "C:\Users\ASUS\anaconda3\envs\holmesvau\python.exe",
    [string]$Prompt = "Please explain the traffic incident process, causes, and impact in detail.",
    [int]$SelectFrames = 12,
    [switch]$UseATS = $true
)

$ErrorActionPreference = "Stop"

function Resolve-AbsPath([string]$PathLike) {
    return (Resolve-Path -LiteralPath $PathLike).Path
}

Write-Host "[1/4] Checking input paths..."
$videoAbs = Resolve-AbsPath $VideoPath
$modelAbs = Resolve-AbsPath $ModelPath
$baseAbs = Resolve-AbsPath $BaseModelPath
$codeAbs = Resolve-AbsPath $CodeRoot
$samplerAbs = Resolve-AbsPath $SamplerPath
$pythonAbs = Resolve-AbsPath $PythonExe

Write-Host "Video   : $videoAbs"
Write-Host "Model   : $modelAbs"
Write-Host "Base    : $baseAbs"
Write-Host "Code    : $codeAbs"
Write-Host "Sampler : $samplerAbs"
Write-Host "Python  : $pythonAbs"

Write-Host "[2/4] Checking required model code files..."
$required = @(
    @{ Name = "configuration_internvl_chat.py"; Src = (Join-Path $codeAbs "model\internvl_chat\configuration_internvl_chat.py") },
    @{ Name = "modeling_internvl_chat.py";      Src = (Join-Path $codeAbs "model\internvl_chat\modeling_internvl_chat.py") },
    @{ Name = "configuration_intern_vit.py";    Src = (Join-Path $codeAbs "model\internvl_chat\configuration_intern_vit.py") },
    @{ Name = "modeling_intern_vit.py";         Src = (Join-Path $codeAbs "model\internvl_chat\modeling_intern_vit.py") },
    @{ Name = "configuration_internlm2.py";     Src = (Join-Path $codeAbs "model\internlm2\configuration_internlm2.py") },
    @{ Name = "modeling_internlm2.py";          Src = (Join-Path $codeAbs "model\internlm2\modeling_internlm2.py") },
    @{ Name = "tokenization_internlm2.py";      Src = (Join-Path $codeAbs "model\internlm2\tokenization_internlm2.py") },
    @{ Name = "tokenization_internlm2_fast.py"; Src = (Join-Path $codeAbs "model\internlm2\tokenization_internlm2_fast.py") },
    @{ Name = "conversation.py";                Src = (Join-Path $codeAbs "conversation.py") }
)

foreach ($item in $required) {
    $target = Join-Path $modelAbs $item.Name
    $src = $item.Src
    if (-not (Test-Path -LiteralPath $src)) {
        $fallback = Join-Path $baseAbs $item.Name
        if (Test-Path -LiteralPath $fallback) {
            $src = $fallback
        } else {
            throw "Missing source file: $src"
        }
    }
    Copy-Item -LiteralPath $src -Destination $target -Force
    Write-Host "Synced: $($item.Name)"
}

Write-Host "[3/4] Setting environment variables..."
$env:PYTHONPATH = "$env:PYTHONPATH;$PWD"

Write-Host "[4/4] Running inference..."
$useAtsBool = if ($UseATS) { "True" } else { "False" }

$pyCode = @"
import torch
from holmesvau.holmesvau_utils import load_model, generate

m = r'$modelAbs'
s = r'$samplerAbs'
video = r'$videoAbs'
prompt = r'''$Prompt'''
device = torch.device('cuda:0')

model, tokenizer, gcfg, sampler = load_model(m, s, device)
pred, _, idx, _ = generate(
    video, prompt, model, tokenizer, gcfg, sampler,
    select_frames=$SelectFrames, use_ATS=$useAtsBool
)
print('Sampled frames:', idx)
print('Model answer:')
print(pred)
"@

& $pythonAbs -c $pyCode

Write-Host "Done."
