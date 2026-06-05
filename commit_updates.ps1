# =====================================================================
#  commit_updates.ps1
#  Run this from the repo root (E:\Dessertation) in PowerShell,
#  AFTER closing VS Code (its Git integration holds the index lock).
#  Your working files are NOT touched — this only rebuilds .git\index.
# =====================================================================
$ErrorActionPreference = "Stop"
Set-Location "E:\Dessertation"

Write-Host "1) Clearing stale lock + corrupt index (working files are preserved)..."
if (Test-Path ".git\index.lock") { Remove-Item -Force ".git\index.lock" }
if (Test-Path ".git\index")      { Remove-Item -Force ".git\index" }

Write-Host "2) Rebuilding the index from the last commit..."
git reset            # mixed reset to HEAD: rebuilds index, keeps your changes

Write-Host "3) Staging updates..."
git add .gitignore
git add Code/project_*/*.ipynb
git add Code/VARIANTS_AND_BASELINES_EXPLAINED.md
git add Code/project_kaggle_opt_67b/opt_67b_comparison.pdf `
        Code/project_kaggle_opt_67b/opt_67b_variants_vs_baselines.html `
        Code/project_kaggle_opt_67b/opt_67b_variants_vs_baselines.tex
git add thesis_report

Write-Host "4) Review what will be committed:"
git status

Write-Host "5) Committing..."
git commit -m "Fix stale variant/baseline counts (12->16 variants, 4->6 wiki baselines); add MIND+Perplexity to Wikipedia held-out eval; add TinyLlama-1.1B (variant+baselines) and Qwen2.5-0.5B baselines notebooks; add dissertation report + variant/baseline reference doc"

Write-Host "6) Pushing to GitHub (will ask for username + Personal Access Token)..."
git push origin main

Write-Host "Done."
