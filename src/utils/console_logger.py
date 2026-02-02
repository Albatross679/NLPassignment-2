# src/utils/console_logger.py
"""Console output logging - captures stdout/stderr to log files."""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, TextIO, Dict, Any


class TeeStream:
    """
    A stream wrapper that writes to both a file and the original stream.

    This enables "tee" behavior where output goes to both the console
    and a log file simultaneously.
    """

    def __init__(
        self,
        file_handle: TextIO,
        original_stream: TextIO,
        tee_to_console: bool = True,
        line_timestamps: bool = False,
        timestamp_format: str = "%H:%M:%S",
        flush_frequency: int = 1
    ):
        """
        Initialize TeeStream.

        Args:
            file_handle: File to write to
            original_stream: Original sys.stdout or sys.stderr
            tee_to_console: If True, also write to original stream
            line_timestamps: If True, prefix each line with timestamp
            timestamp_format: Format string for timestamps
            flush_frequency: Flush file after this many lines (1 = every line)
        """
        self.file_handle = file_handle
        self.original_stream = original_stream
        self.tee_to_console = tee_to_console
        self.line_timestamps = line_timestamps
        self.timestamp_format = timestamp_format
        self.flush_frequency = flush_frequency
        self._line_count = 0
        self._line_buffer = ""

    def write(self, text: str) -> int:
        """Write text to file and optionally to console."""
        if not text:
            return 0

        # Handle line timestamps
        if self.line_timestamps:
            output_text = self._add_timestamps(text)
        else:
            output_text = text

        # Write to file
        self.file_handle.write(output_text)

        # Track lines for flush frequency
        self._line_count += text.count('\n')
        if self._line_count >= self.flush_frequency:
            self.file_handle.flush()
            self._line_count = 0

        # Write to original stream if tee is enabled
        if self.tee_to_console and self.original_stream:
            self.original_stream.write(text)
            self.original_stream.flush()

        return len(text)

    def _add_timestamps(self, text: str) -> str:
        """Add timestamps to the beginning of each line."""
        if not text:
            return text

        lines = text.split('\n')
        timestamped_lines = []

        for i, line in enumerate(lines):
            # Don't timestamp empty trailing newline
            if i == len(lines) - 1 and line == '':
                timestamped_lines.append('')
            elif line or i < len(lines) - 1:
                timestamp = datetime.now().strftime(self.timestamp_format)
                if self._line_buffer:
                    # Continue previous incomplete line
                    timestamped_lines.append(line)
                    self._line_buffer = ""
                else:
                    timestamped_lines.append(f"[{timestamp}] {line}")

        # If text doesn't end with newline, buffer for next write
        if text and not text.endswith('\n'):
            self._line_buffer = lines[-1] if lines else ""

        return '\n'.join(timestamped_lines)

    def flush(self) -> None:
        """Flush both file and original stream."""
        self.file_handle.flush()
        if self.tee_to_console and self.original_stream:
            self.original_stream.flush()

    def fileno(self) -> int:
        """Return file descriptor for compatibility."""
        if self.original_stream:
            return self.original_stream.fileno()
        return self.file_handle.fileno()

    def isatty(self) -> bool:
        """Check if stream is a TTY."""
        if self.tee_to_console and self.original_stream:
            return self.original_stream.isatty()
        return False

    @property
    def encoding(self) -> str:
        """Return encoding for compatibility."""
        if self.original_stream:
            return self.original_stream.encoding
        return 'utf-8'


class ConsoleLogger:
    """
    Console logger that captures stdout/stderr to log files.

    Supports:
    - Combined or separate stream logging
    - Tee behavior (output to both file and console)
    - Optional per-line timestamps
    - Context manager protocol

    Usage:
        console_logger = ConsoleLogger(config=cfg_dict.get('console', {}), run_dir=run_dir)
        # ... training code with print statements ...
        console_logger.close()

    Or as context manager:
        with ConsoleLogger(config=cfg_dict.get('console', {}), run_dir=run_dir):
            # ... training code ...
    """

    def __init__(self, config: Dict[str, Any], run_dir: Path):
        """
        Initialize ConsoleLogger and start capturing output.

        Args:
            config: Console configuration dictionary with keys:
                - enabled: bool (default True)
                - filename: str (default "console.log")
                - separate_streams: bool (default False)
                - stdout_filename: str (default "stdout.log")
                - stderr_filename: str (default "stderr.log")
                - tee_to_console: bool (default True)
                - line_timestamps: bool (default False)
                - timestamp_format: str (default "%H:%M:%S")
                - flush_frequency: int (default 1)
            run_dir: Directory to write log files to
        """
        self.config = config
        self.run_dir = Path(run_dir)
        self.enabled = config.get('enabled', True)

        # Store original streams
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

        # File handles
        self._combined_file: Optional[TextIO] = None
        self._stdout_file: Optional[TextIO] = None
        self._stderr_file: Optional[TextIO] = None

        # TeeStream wrappers
        self._stdout_tee: Optional[TeeStream] = None
        self._stderr_tee: Optional[TeeStream] = None

        self._is_active = False

        if self.enabled:
            self._start_capture()

    def _start_capture(self) -> None:
        """Start capturing stdout/stderr."""
        if self._is_active:
            return

        # Configuration
        separate_streams = self.config.get('separate_streams', False)
        tee_to_console = self.config.get('tee_to_console', True)
        line_timestamps = self.config.get('line_timestamps', False)
        timestamp_format = self.config.get('timestamp_format', '%H:%M:%S')
        flush_frequency = self.config.get('flush_frequency', 1)

        # Ensure run_dir exists
        self.run_dir.mkdir(parents=True, exist_ok=True)

        if separate_streams:
            # Separate files for stdout and stderr
            stdout_filename = self.config.get('stdout_filename', 'stdout.log')
            stderr_filename = self.config.get('stderr_filename', 'stderr.log')

            self._stdout_file = open(self.run_dir / stdout_filename, 'w', encoding='utf-8')
            self._stderr_file = open(self.run_dir / stderr_filename, 'w', encoding='utf-8')

            self._stdout_tee = TeeStream(
                file_handle=self._stdout_file,
                original_stream=self._original_stdout,
                tee_to_console=tee_to_console,
                line_timestamps=line_timestamps,
                timestamp_format=timestamp_format,
                flush_frequency=flush_frequency
            )

            self._stderr_tee = TeeStream(
                file_handle=self._stderr_file,
                original_stream=self._original_stderr,
                tee_to_console=tee_to_console,
                line_timestamps=line_timestamps,
                timestamp_format=timestamp_format,
                flush_frequency=flush_frequency
            )
        else:
            # Combined file for both streams
            filename = self.config.get('filename', 'console.log')
            self._combined_file = open(self.run_dir / filename, 'w', encoding='utf-8')

            self._stdout_tee = TeeStream(
                file_handle=self._combined_file,
                original_stream=self._original_stdout,
                tee_to_console=tee_to_console,
                line_timestamps=line_timestamps,
                timestamp_format=timestamp_format,
                flush_frequency=flush_frequency
            )

            self._stderr_tee = TeeStream(
                file_handle=self._combined_file,
                original_stream=self._original_stderr,
                tee_to_console=tee_to_console,
                line_timestamps=line_timestamps,
                timestamp_format=timestamp_format,
                flush_frequency=flush_frequency
            )

        # Redirect streams
        sys.stdout = self._stdout_tee
        sys.stderr = self._stderr_tee

        self._is_active = True

    def close(self) -> None:
        """Stop capturing and restore original streams."""
        if not self._is_active:
            return

        # Flush any remaining output
        if self._stdout_tee:
            self._stdout_tee.flush()
        if self._stderr_tee:
            self._stderr_tee.flush()

        # Restore original streams
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

        # Close file handles
        if self._combined_file:
            self._combined_file.close()
            self._combined_file = None
        if self._stdout_file:
            self._stdout_file.close()
            self._stdout_file = None
        if self._stderr_file:
            self._stderr_file.close()
            self._stderr_file = None

        self._stdout_tee = None
        self._stderr_tee = None
        self._is_active = False

    def __enter__(self) -> 'ConsoleLogger':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensures streams are restored."""
        self.close()

    def __del__(self) -> None:
        """Destructor - ensure cleanup on garbage collection."""
        self.close()
