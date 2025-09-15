# Behavior Acquisition App

![License](https://img.shields.io/github/license/caraido/Behavior_Acquisition)
[![Website](https://img.shields.io/badge/Check-Paper-blue)](https://doi.org/10.1016/j.cub.2024.11.041)

## Overview

Behavior Acquisition App is a lightweight (and extensible) software for orchestrating behavioral data acquisition devices/services and exposing their state over a real‑time socket server for a graphical user interface (GUI) or other controlling/monitoring clients.

---

## Quick Summary

The application:

1. Builds an Acquisition Group (AG) and a shared Status object via `setup()`.
2. Registers event and data callbacks via `initCallbacks(ag, status)`.
3. Starts a socket-based server (likely a Flask/Socket.IO style app) via `initServer(ag, status)`.
4. Serves the live acquisition control & telemetry interface on port **5001**.
5. Supports a "mock" mode for development/testing: `python main.py mock`.

---

## Core Concepts

| Concept | Description |
|--------|-------------|
| Acquisition Group (`ag`) | Central orchestrator aggregating devices / streams (e.g., cameras, microphone, etc.) and providing coordinated start/stop & data dispatch. |
| Status | Shared, mutable state object (health metrics, device statuses, session state, timestamps, counters) consumed by callbacks and the server layer. |
| Callbacks (`initCallbacks`) | Register functions reacting to acquisition events (new frame, device error, state change) to update Status and emit socket events. |
| Server (`initServer`) | Creates the web/socket interface—providing real‑time subscription channels and API endpoints for the GUI or automation clients. |
| Mock Mode | Swaps real `setup` / `callbacks` with `mockSetup` / `mockCallbacks` to simulate devices and deterministic status updates. |

## Features

- **Multi-modal Data Collection**: Capture behavioral data from various sources (video, microphone, optogenetic probe, manual inputs)
- **Automated Processing Pipeline**: Convert raw data into structured formats suitable for analysis
- **Customizable Analysis Tools**: Apply various analytical methods to extracted behavioral metrics
- **Visualization Components**: Generate insightful visualizations of behavioral patterns
- **Real-time telemetry & control via sockets**: supports low-latency updates from acquisition layer to GUI.

## Architecture Overview

```
+---------------------+
|  Configuration Load |
+----------+----------+
           |
           v
+---------------------+        +------------------+
|  Device Manager     | <----> |  Time Sync / CLK |
+----------+----------+        +------------------+
           |
           v
+---------------------+        +--------------------+
| Acquisition Loop    | -----> |  Processing Stages |
| (async / threads)   |        |  (e.g. decoding,   |
+----------+----------+        |   preprocessing)   |
           |                   +---------+----------+
           v                             |
+---------------------+                  v
|  Data Buffer / MQ   |          +------------------+
+----------+----------+          |  Storage Writer  |
           |                     |  (raw + meta)    |
           v                     +--------+---------+
+---------------------+                   |
|   Live Monitor / UI | <-----------------+
+---------------------+
```



## Installation

```bash
# Clone the repository
git clone https://github.com/caraido/Behavior_Acquisition.git

# Navigate to the project directory
cd Behavior_Acquisition

# Install dependencies
pip install -r requirements.txt

# Run setup script
python setup.py install
```
or contact tianhaolei2019@u.northwestern.edu
