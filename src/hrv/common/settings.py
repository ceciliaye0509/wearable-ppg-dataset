from pathlib import Path
DEVICES=("Earring","Ring","Watch")
CHANNELS=("ppg_green","ppg_ir")
def normalize_participant_id(value):
    value=str(value).strip().upper().removeprefix("P")
    return f"P{int(value)}"
