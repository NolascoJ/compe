#!/bin/zsh

set -u
setopt NO_BG_NICE

ROOT=/Users/jnolascodecarles/ITBA/4_1Q/ml/compe
PYTHON="$ROOT/.venv/bin/python"
CONTROLLER_LOG="$ROOT/experiments/logs/cola_ridge_knn_bagging.log"
PID_FILE="$ROOT/experiments/logs/cola_ridge_knn_bagging.pid"

cd "$ROOT" || exit 1
print -r -- "$$" > "$PID_FILE"

stamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

run_stage() {
  local model="$1"
  local stage="$2"
  local model_log="$ROOT/experiments/logs/${model}.log"
  local checks=0

  print -r -- "$(stamp) START model=$model stage=$stage" >> "$CONTROLLER_LOG"
  "$PYTHON" -m compe.run "$model" --etapa "$stage" >> "$model_log" 2>&1 &
  local child_pid=$!
  print -r -- "$(stamp) PID model=$model stage=$stage pid=$child_pid log=$model_log" >> "$CONTROLLER_LOG"

  while kill -0 "$child_pid" 2>/dev/null; do
    sleep 15
    checks=$((checks + 1))
    if (( checks % 20 == 0 )); then
      print -r -- "$(stamp) CHECK model=$model stage=$stage pid=$child_pid status=running" >> "$CONTROLLER_LOG"
      tail -n 5 "$model_log" >> "$CONTROLLER_LOG" 2>&1
    fi
  done

  wait "$child_pid"
  local exit_code=$?
  if (( exit_code != 0 )); then
    print -r -- "$(stamp) FAILED model=$model stage=$stage pid=$child_pid exit=$exit_code; queue stopped" >> "$CONTROLLER_LOG"
    tail -n 40 "$model_log" >> "$CONTROLLER_LOG" 2>&1
    exit "$exit_code"
  fi
  print -r -- "$(stamp) DONE model=$model stage=$stage pid=$child_pid exit=0" >> "$CONTROLLER_LOG"
}

print -r -- "$(stamp) QUEUE_START controller_pid=$$ order=ridge,knn,bagging" >> "$CONTROLLER_LOG"

# Skip fully completed models when the queue is resumed.
if [[ -f "$ROOT/experiments/ridge/metricas.json" ]]; then
  print -r -- "$(stamp) SKIP model=ridge reason=already_complete" >> "$CONTROLLER_LOG"
else
  # Ridge's base, curve and exhaustive broad search already exist.
  run_stage ridge refinar
  run_stage ridge confirmar
  run_stage ridge final
fi

# KNN's base and curve exist; its broad search resumes from the saved rows.
if [[ -f "$ROOT/experiments/knn/metricas.json" ]]; then
  print -r -- "$(stamp) SKIP model=knn reason=already_complete" >> "$CONTROLLER_LOG"
else
  run_stage knn busqueda
  run_stage knn refinar
  run_stage knn confirmar
  run_stage knn final
fi

# Bagging has no completed stages.
if [[ -f "$ROOT/experiments/bagging/metricas.json" ]]; then
  print -r -- "$(stamp) SKIP model=bagging reason=already_complete" >> "$CONTROLLER_LOG"
else
  run_stage bagging base
  run_stage bagging curva
  run_stage bagging busqueda
  run_stage bagging refinar
  run_stage bagging confirmar
  run_stage bagging final
fi

print -r -- "$(stamp) QUEUE_DONE controller_pid=$$ exit=0" >> "$CONTROLLER_LOG"
