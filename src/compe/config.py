"""Rutas y constantes del proyecto. Unica fuente de verdad para nombres de columnas."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
FIGURES = ROOT / "reports" / "figures"
MODELS = ROOT / "models"
EXPERIMENTS = ROOT / "experiments"

SEED = 42
TARGET = "canceled_job"

# Describen el resultado: nunca entran al modelo.
LEAK_COLS = ["execution_status", "execution_status_date"]

# Sumas exactas de otras columnas.
REDUNDANT_COLS = ["total_processes", "total_gpu_hours"]

CAT_COLS = [
    "service_tier",
    "client_region",
    "client_type",
    "request_channel",
    "reserved_gpu_type",
    "assigned_gpu_type",
    "reservation_commitment",
    "workload_profile",
]

# Identificadores guardados como float: son categorias, no magnitudes.
ID_COLS = ["orchestration_agent_id", "organization_id"]

DATE_COLS = ["execution_year", "execution_month", "execution_week", "execution_day"]

NUM_COLS = [
    "hours_before_execution",
    "peak_gpu_hours",
    "offpeak_gpu_hours",
    "num_primary_processes",
    "num_auxiliary_processes",
    "num_background_processes",
    "previous_job_cancellations",
    "successful_previous_jobs",
    "resource_modifications",
    "queue_wait_time",
    "estimated_cost_per_hour",
    "special_resource_requests",
]

BIN_COLS = ["returning_client", "reserved_storage_units", "high_value_client"]

# Todo lo que puede ver el modelo (sin target ni fugas).
FEATURE_COLS = DATE_COLS + NUM_COLS + BIN_COLS + CAT_COLS + ID_COLS + REDUNDANT_COLS
