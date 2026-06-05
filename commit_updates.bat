@echo off
REM ============================================================
REM  commit_updates.bat  --  double-click this file (it lives in
REM  the repo root E:\Dessertation). CLOSE VS CODE FIRST.
REM  It rebuilds the git index, stages everything, commits, pushes.
REM  Your working files are NOT touched.
REM ============================================================
cd /d "%~dp0"
echo.
echo ===  Make sure VS Code is CLOSED, then press a key  ===
pause

echo.
echo [1/5] Clearing stale lock + corrupt index (files are safe)...
if exist ".git\index.lock" del /f /q ".git\index.lock"
if exist ".git\index"      del /f /q ".git\index"

echo [2/5] Rebuilding index from last commit...
git reset

echo [3/5] Staging updates...
git add .gitignore Code/project_*/*.ipynb Code/VARIANTS_AND_BASELINES_EXPLAINED.md thesis_report commit_updates.ps1 commit_updates.bat
git add Code/project_kaggle_opt_67b/opt_67b_comparison.pdf Code/project_kaggle_opt_67b/opt_67b_variants_vs_baselines.html Code/project_kaggle_opt_67b/opt_67b_variants_vs_baselines.tex

echo.
echo [4/5] What will be committed:
git status
echo.

git commit -m "Fix variant/baseline counts and Falcon OOM; add MIND+Perplexity wiki eval and per-text timing benchmark; add TinyLlama + Qwen2.5-0.5B notebooks; add dissertation report and reference doc"

echo.
echo [5/5] Pushing to GitHub. When asked:
echo        Username = your GitHub username
echo        Password = a Personal Access Token (NOT your account password)
git push origin main

echo.
echo ============================================================
echo   DONE. Read the messages above. If you see an error,
echo   copy it and send it to me.
echo ============================================================
pause
