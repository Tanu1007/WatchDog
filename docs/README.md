# WatchDog docs

| File | What's in it |
| --- | --- |
| [WORKLOG.md](WORKLOG.md) | Everything done so far: timestamps, durations, results, decisions, open items |
| [SETUP.md](SETUP.md) | Step-by-step guide for teammates to reproduce the environment and datasets |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Mermaid diagrams: system, alert FSM, code structure, data pipeline |
| [requirements-lock.txt](requirements-lock.txt) | Exact package versions from the working venv (`pip freeze`) |
| [logs/](logs/) | Raw output from downloads, extraction, conversion, manifest and tests, named `YYYY-MM-DD_<task>.log` |

Update `WORKLOG.md` as each task finishes: add a row with start, end and duration, and put the command output in `logs/`.
