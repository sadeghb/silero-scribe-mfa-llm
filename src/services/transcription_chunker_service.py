# src/services/transcription_chunker_service.py
import pandas as pd

class TranscriptionChunkerService:
    """
    Uses a list of eligible split points to create chunks for transcription
    that are as long as possible without exceeding a maximum duration.
    """
    def __init__(self, max_duration_ms: int = 475000):
        print("TranscriptionChunkerService initialized.")
        self.max_duration_ms = max_duration_ms

    def run(self, split_points_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates chunk timestamps based on eligible split points.

        Args:
            split_points_df (pd.DataFrame): DataFrame from SplitPointService.

        Returns:
            pd.DataFrame: DataFrame with 'chunk_start_ms' and 'chunk_end_ms'.
        """
        if split_points_df.empty:
            return pd.DataFrame(columns=['chunk_start_ms', 'chunk_end_ms'])

        chunks = []
        split_points = split_points_df['split_point_ms'].tolist()
        
        current_chunk_start_index = 0
        while current_chunk_start_index < len(split_points) - 1:
            start_time = split_points[current_chunk_start_index]
            end_index = current_chunk_start_index
            
            # Find the furthest split point that is within the max duration
            for i in range(current_chunk_start_index + 1, len(split_points)):
                if split_points[i] - start_time <= self.max_duration_ms:
                    end_index = i
                else:
                    break # Stop as soon as we exceed the max duration

            # If no progress was made (e.g., a very long VAD segment), force at least one step
            if end_index == current_chunk_start_index:
                end_index += 1

            end_time = split_points[end_index]
            chunks.append({
                'chunk_start_ms': start_time,
                'chunk_end_ms': end_time
            })
            
            # The next chunk starts where the last one ended
            current_chunk_start_index = end_index

        return pd.DataFrame(chunks)
    