import numpy as np

class Evaluator:
    @staticmethod
    def sigmoid_to_true_duration(y: np.ndarray, fsamp=250) -> np.ndarray:
        """
        Convert sigmoided variant to true duration.
        
        Args:
            y: Sigmoided duration value from 0 to 1
            fsamp: Sampling frequency of the signal
            
        Returns:
            True duration in samples
        """
        return y * 2 * fsamp
    
    @staticmethod
    def true_duration_to_sigmoid(duration: np.ndarray, fsamp=250) -> np.ndarray:
        """
        Convert true duration to sigmoided variant.
        
        Args:
            duration: Duration in samples
            fsamp: Sampling frequency of the signal
            
        Returns:
            Sigmoided duration from 0 to 1
        """
        return duration / (2 * fsamp)
    
    @staticmethod
    def detections_to_segmentation(detections: np.ndarray, seq_len: int, confidence_threshold=1e-6) -> np.ndarray:
        """
        Convert detections to segmentation.
        
        Args:
            detections: Detection array [30, 3] where 3 = confidence, center offset, sigmoided duration
            seq_len: Length of the sequence
            confidence_threshold: Threshold for confidence values
            
        Returns:
            Segmentation array [seq_len, 1] with 0s and 1s
        """
        output = np.zeros((seq_len, 1), dtype=np.float32)
        denominator = np.zeros((seq_len, 1), dtype=np.float32)
        intervals = Evaluator.detections_to_intervals(detections, seq_len, confidence_threshold)
        intervals = Evaluator.intervals_nms(intervals)
        
        for start, end, confidence in intervals:
            output[int(start):int(end)+1, 0] += confidence
            denominator[int(start):int(end)+1, 0] += 1
        
        # Divide by denominator to get the average confidence
        output = np.nan_to_num(output / denominator)
            
        return output
    
    @staticmethod
    def detections_to_intervals(detections: np.ndarray, seq_len: int, confidence_threshold=1e-6) -> np.ndarray:
        """
        Convert detections to intervals.
        
        Args:
            detections: Detection array [30, 3] where 3 = confidence, center offset, sigmoided duration
            seq_len: Length of the sequence
            confidence_threshold: Threshold for confidence values
            
        Returns:
            Intervals array [N, 3] where 3 = start, end, confidence
        """
        output = np.zeros_like(detections)
        
        num_segments = detections.shape[0]
        segment_duration = seq_len / num_segments
        j = 0
        for i in range(num_segments):
            confidence, center_offset, sigmoided_duration = detections[i]
            if confidence < confidence_threshold:
                continue
            
            true_center = (i + center_offset) * segment_duration
            true_duration = Evaluator.sigmoid_to_true_duration(sigmoided_duration)
            start = true_center - true_duration / 2
            end = true_center + true_duration / 2
            
            # Clip start/end to [0, seq_len]
            start = np.clip(start, 0, seq_len)
            end = np.clip(end, 0, seq_len)
            
            output[j] = [start, end, confidence]
            j += 1
        
        return output[:j]  # Return only the filled part
    
    @staticmethod
    def segmentation_to_detections(segmentation: np.ndarray) -> np.ndarray:
        """
        Convert segmentation to detections.
        
        Args:
            segmentation: Segmentation array [seq_len, 1] with 0s and 1s
            
        Returns:
            Detections array [30, 3] where 3 = confidence, center offset, sigmoided duration
        """
        assert len(segmentation.shape) == 2, f"Expected segmentation to be 2D, but got {segmentation.shape}"
        assert segmentation.shape[1] == 1, f"Expected segmentation to have 1 channel, but got {segmentation.shape[1]}"
        
        seq_len = segmentation.shape[0]
        num_segments = 30
        segment_length = seq_len / num_segments
        
        # Initialize the output array
        detections = np.zeros((num_segments, 3), dtype=np.float32)
        
        # Find the start and end of each spindle
        starts = np.where(np.diff(segmentation[:,0]) == 1)[0]
        ends = np.where(np.diff(segmentation[:, 0]) == -1)[0]
        
        if segmentation[0, 0] == 1:
            starts = np.concatenate([[0], starts])
        if segmentation[-1, 0] == 1:
            ends = np.concatenate([ends, [seq_len - 1]])
            
        assert len(starts) == len(ends), f"Number of starts and ends do not match. Starts: {len(starts)}, Ends: {len(ends)}"
    
        # Iterate over each spindle
        for start, end in zip(starts, ends):
            center = (start + end) // 2
            segment_id = int(center / segment_length)
            
            # Mark spindle
            detections[segment_id, 0] = 1
            # Calculate center offset
            offset = (center % segment_length) / segment_length
            detections[segment_id, 1] = offset
            # Calculate duration
            true_duration = end - start
            detections[segment_id, 2] = Evaluator.true_duration_to_sigmoid(true_duration)
        
        return detections
    
    @staticmethod
    def intervals_nms(intervals: np.ndarray, iou_threshold=1.0) -> np.ndarray:
        """
        Perform non-maximum suppression on intervals.
        
        Args:
            intervals: Intervals array [N, 3] where 3 = start, end, confidence
            iou_threshold: IoU threshold for suppression
            
        Returns:
            Filtered intervals array [M, 3] where M <= N
        """
        if len(intervals) == 0:
            return np.array([])
        
        # Drop all intervals which are zeroes (padding)
        mask = intervals[:, 0] != 0
        if not np.any(mask):
            return np.array([])
            
        intervals = intervals[mask]

        # Sort intervals by confidence score in descending order
        intervals = intervals[intervals[:, 2].argsort()[::-1]]

        selected_intervals = []

        while len(intervals) > 0:
            # Select the interval with the highest confidence
            current_interval = intervals[0]
            selected_intervals.append(current_interval)

            if len(intervals) == 1:
                break

            # Compute IoU (Intersection over Union) between the selected interval and the rest
            start_max = np.maximum(current_interval[0], intervals[1:, 0])
            end_min = np.minimum(current_interval[1], intervals[1:, 1])
            intersection = np.maximum(0, end_min - start_max)
            union = (current_interval[1] - current_interval[0]) + (intervals[1:, 1] - intervals[1:, 0]) - intersection
            iou = intersection / union

            # Keep intervals with IoU less than the threshold
            intervals = intervals[1:][iou < iou_threshold]

        return np.array(selected_intervals)
