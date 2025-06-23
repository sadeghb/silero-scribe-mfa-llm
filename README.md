# AI-Powered Audio Disfluency Editing Pipeline

## 1. Overview

This project implements a sophisticated, multi-stage pipeline designed to automatically process long-form audio, identify speech disfluencies (such as filler words and repetitions), and generate a high-quality dataset of edited audio clips. It leverages a combination of Voice Activity Detection (VAD), AI-driven transcription and analysis, and phoneme-level forced alignment to produce a variety of "natural" and "unnatural" sounding audio edits.

The primary goal is to create a rich dataset suitable for training machine learning models to perform automated audio editing tasks.

## 2. Features

- **Modular Architecture**: Built on an Orchestrator/Services model for flexibility and easy maintenance.
- **Multi-Stage Caching**: Caches the output of each major stage (VAD, Scribe, MFA, LLM) to prevent redundant processing and minimize expensive API calls.
- **Voice Activity Detection**: Uses Silero VAD for accurate detection of speech segments, forming the basis for all subsequent chunking.
- **AI-Powered Transcription & Analysis**:
    - Utilizes the ElevenLabs Scribe API for highly accurate transcription.
    - Employs a Large Language Model (Google Gemini) to intelligently identify and tag disfluencies in the transcript.
- **Phoneme-Level Precision**: Integrates the Montreal Forced Aligner (MFA) to generate hyper-accurate, phoneme-level timestamps for every word.
- **Advanced Audio Editing**:
    - Generates four distinct audio clips for each identified disfluency.
    - Creates "natural" sounding cuts by slicing audio during silent periods.
    - Creates "unnatural" cuts using a configurable "phoneme invasion" technique for dataset diversity.
- **Rich Metadata**: Produces detailed JSON metadata for every generated clip, documenting the edits performed.

## 3. Architecture

The system is designed around a central **Pipeline Orchestrator** (`src/pipeline_orchestrator.py`) that manages the end-to-end workflow. It coordinates a suite of **Specialist Services**, where each service is a single-responsibility class that handles a specific technical job (e.g., `VADService`, `ScribeService`, `AudioEditorService`). This modular design allows for components to be easily tested, swapped, or modified.

## 4. Pipeline Workflow

The orchestrator executes the following sequence of operations for each input audio file:

### Stage A: Transcription and Disfluency Identification

1.  **VAD**: The `VADService` processes the audio to find all speech timestamps.
2.  **Split Point Generation**: The `SplitPointService` analyzes the VAD output to determine all acoustically safe timestamps where the audio can be split.
3.  **Chunking & Transcription**: The audio is split into manageable chunks, which are sent to the Scribe API via the `ScribeService`. Raw results for each chunk are cached.
4.  **Transcript Normalization**: The `ScribeNormalizerService` combines the chunked transcripts, adjusts all timestamps to be absolute to the start of the original file, and cleans up transcription artifacts. The full transcript object is cached.
5.  **LLM Disfluency Tagging**: The clean transcript text is sent to the Gemini LLM via the `LLMCutSelectorService`. The LLM returns the same text with disfluencies wrapped in `<cut>` tags. This marked-up text is cached.

### Stage B: Alignment and Dataset Generation

6.  **MFA Alignment**: In parallel, a separate workflow generates phoneme-level alignments.
    - The `MfaChunkerService` creates optimal audio chunks for the aligner.
    - The `MfaAlignerService` runs the `mfa align` command.
    - The `MfaNormalizerService` parses the output TextGrids into a final, hyper-accurate JSON transcript with phoneme timings. This is cached.
7.  **Parsing Cuts**: The `CutParserService` reads the LLM's marked-up text and compares it against the full Scribe transcript to create a definitive list of word IDs to be cut.
8.  **Audio Editing**: For each cut in the list, the `AudioEditorService` performs the main editing logic:
    - It defines a "container chunk" by expanding from the target words to the nearest VAD-based silent points, ensuring a one-word context.
    - It generates four audio clips: `original`, `natural_cut`, `unnatural_backward` (backward phoneme invasion), and `unnatural_forward` (forward phoneme invasion).
    - All precise cut timestamps are derived from the MFA data and adjusted to the nearest zero-crossing.
9.  **Dataset Finalization**: The `DatasetGeneratorService` saves the four audio clips and a detailed `metadata.json` file for each cut into the `output_dataset` directory.

## 5. Key Technologies

- **Python 3.10+**
- **VAD**: Silero VAD
- **Audio**: Pydub, Librosa
- **Transcription**: ElevenLabs Scribe API
- **LLM**: Google Gemini API
- **Alignment**: Montreal Forced Aligner (MFA)
- **Orchestration**: Custom Python classes
- **Configuration**: YAML

## 6. Project Structure

-   **silero_scribe_mfa/**
    -   **audio_inputs/**: Place your source audio files here.
    -   **cache/**: Caches intermediate results from each pipeline stage.
        -   **vad/**
        -   **split_points/**
        -   **audio_chunks/**
        -   **scribe/**
        -   **mfa/**
        -   **llm/**
    -   **output_dataset/**: The final generated dataset is stored here.
    -   **src/**: All Python source code.
        -   **services/**: Contains all the modular, single-responsibility services.
            -   `vad_service.py`
            -   `split_point_service.py`
            -   `transcription_chunker_service.py`
            -   `audio_splitter_service.py`
            -   `scribe_service.py`
            -   `scribe_normalizer_service.py`
            -   `llm_cut_selector_service.py`
            -   `mfa_chunker_service.py`
            -   `mfa_aligner_service.py`
            -   `mfa_normalizer_service.py`
            -   `cut_parser_service.py`
            -   `audio_editor_service.py`
            -   `dataset_generator_service.py`
        -   **utils/**: Contains helper utilities.
            -   `config_loader.py`
            -   `mfa_text_normalizer.py`
        -   `model_loader.py`: Loads the Silero VAD model.
        -   `pipeline_orchestrator.py`: The central class that manages the pipeline workflow.
    -   `main.py`: The main entry point to run the application.
    -   `config.yaml`: The central configuration file for all settings and API keys.
    -   `requirements.txt`: Python package dependencies.

## 7. Setup and Installation

1.  **Clone Repository**:
    ```bash
    git clone <repository_url>
    cd silero_scribe_mfa
    ```

2.  **Install Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install Montreal Forced Aligner**:
    Follow the official MFA installation instructions for your operating system. Ensure the `mfa` command is available in your system's PATH.

4.  **Download MFA Models**:
    You will need to download a pre-trained acoustic model and dictionary. For English, you can use:
    ```bash
    mfa model download acoustic english_us_arpa
    mfa model download dictionary english_us_arpa
    ```

5.  **Configure API Keys**:
    Open the `config.yaml` file and enter your API keys for `elevenlabs` (for Scribe) and `llm` (for Google Gemini).

## 8. How to Run

1.  Place your source audio files (e.g., `.wav`, `.mp3`) into the `audio_inputs/` directory.
2.  Run the main script from your terminal:
    ```bash
    python main.py
    ```
3.  The pipeline will process each file and display a progress bar. All intermediate files will be stored in the `cache/` directory, and the final dataset will be generated in the `output_dataset/` directory.

## 9. Output Structure

The final output is organized by the source audio file and then by each cut performed on that file.

-   **output_dataset/**
    -   **<source_audio_name>/**
        -   **cut_1/**
            -   `original.wav`: The small, unedited chunk of audio containing the disfluency and its context.
            -   `natural_cut.wav`: The "natural" sounding edit of the original chunk.
            -   `unnatural_backward.wav`: The unnatural cut with backward phoneme invasion.
            -   `unnatural_forward.wav`: The unnatural cut with forward phoneme invasion.
            -   `metadata.json`: A file containing detailed metadata about this specific cut.
        -   **cut_2/**
            -   *(...same structure as cut_1)*