# src/services/split_point_service.py
import pandas as pd
from typing import Dict, Any

class SplitPointService:
    """
    Analyzes VAD timestamps to generate a list of all points where the audio
    can be safely split, including the silence extent around each point.
    """
    def __init__(self):
        print("SplitPointService initialized.")

    def run(self, vad_timestamps_df: pd.DataFrame, total_duration_ms: int) -> pd.DataFrame:
        """
        Generates a DataFrame of eligible split points.

        Args:
            vad_timestamps_df (pd.DataFrame): DataFrame with 'start_ms' and 'end_ms'.
            total_duration_ms (int): The total duration of the source audio file.

        Returns:
            pd.DataFrame: A DataFrame with 'split_point_ms', 'silence_start_ms', 
                          and 'silence_end_ms' columns.
        """
        if vad_timestamps_df.empty:
            return pd.DataFrame(columns=['split_point_ms', 'silence_start_ms', 'silence_end_ms'])

        split_points = []

        # 1. First split point is always at the beginning
        split_points.append({
            'split_point_ms': 0,
            'silence_start_ms': 0,
            'silence_end_ms': vad_timestamps_df.iloc[0]['start_ms']
        })

        # 2. Intermediate split points are in the middle of silences
        for i in range(len(vad_timestamps_df) - 1):
            silence_start = vad_timestamps_df.iloc[i]['end_ms']
            silence_end = vad_timestamps_df.iloc[i+1]['start_ms']
            mid_point = silence_start + (silence_end - silence_start) / 2
            
            split_points.append({
                'split_point_ms': int(mid_point),
                'silence_start_ms': silence_start,
                'silence_end_ms': silence_end
            })

        # 3. Last split point is always at the very end
        split_points.append({
            'split_point_ms': total_duration_ms,
            'silence_start_ms': vad_timestamps_df.iloc[-1]['end_ms'],
            'silence_end_ms': total_duration_ms
        })

        return pd.DataFrame(split_points)
    