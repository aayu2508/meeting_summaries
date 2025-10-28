# meeting_summaries Project

## Overview
The `meeting_summaries` project is designed to process audio recordings of meetings, extract key ideas, and manage participation data. It utilizes automatic speech recognition (ASR) and diarization techniques to segment and analyze audio content.

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd meeting_summaries
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy `.env.example` to `.env` and fill in the required values such as `HF_TOKEN` and `ASR_MODEL`.

## Usage Guidelines
- Place original media files in the `data/raw/` directory.
- Run the main script to start the ASR diarization pipeline:
  ```
  bash run.sh
  ```
- Output segments will be saved in `data/outputs/m1/segments.jsonl`.

## Directory Structure
- `data/`: Contains raw media files and output segments.
- `src/`: Source code for processing and analysis.
- `scripts/`: Additional scripts for project-related tasks.
- `run.sh`: Convenience script to execute the main processes.

For more detailed information on each component, refer to the respective files in the `src/` directory.