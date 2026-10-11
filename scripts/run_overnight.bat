@echo off
REM Visual track: train 130 -> 300 people, test 50 -> 130 people, then CV.
REM Safe to stop (Ctrl+C or shut down) and run again: finished work is skipped.
REM The test results are NOT opened here.
setlocal
cd /d "%~dp0.."
call ai_for_interview\Scripts\activate.bat
set GT=input\ground_truth

echo === 1/6 back up metadata ===
if not exist input\metadata\metadata_backup_before_visual300.json copy input\metadata\metadata.json input\metadata\metadata_backup_before_visual300.json

echo === 2/6 pick 170 more train people + the other 80 test people ===
if not exist %GT%\avi_train4_subset.csv python base\dataset\build_subset.py --split train --n 170 --name train4 --exclude-manifest %GT%\avi_train1_subset.csv %GT%\avi_train2_subset.csv %GT%\avi_train3_subset.csv --archive-others
if not exist %GT%\avi_eval2_subset.csv python base\dataset\build_subset.py --split test --n 80 --name eval2 --exclude-manifest %GT%\avi_eval_subset.csv

echo === 3/6 speech track for the test people (audio, transcript, embeddings) ===
python run_speech_pipeline.py --splits test

echo === 4/6 visual track, clip by clip (the long step) ===
python scripts\run_visual_batch.py --manifests eval2 train4
if errorlevel 1 goto :stopped

echo === 5/6 M5 ratings for the new clips (Ollama must be running) ===
python approaches\scoring_m5_local_llm\run.py

echo === 6/6 cross-validation on train+val (test stays locked) ===
python validation\cv_score_model.py --data full --text-emb minilm bge-small --llm qwen2.5:3b

echo.
echo ALL DONE. Send the last screen of output to Claude.
goto :eof

:stopped
echo Stopped. Run scripts\run_overnight.bat again to continue.
