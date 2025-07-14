import cProfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


@contextmanager
def profile_to_file(
    filename: str | None = None,
    output_dir: str = "profiles",
    create_dir: bool = True,
    enabled: bool = False,
) -> Iterator[Path | None]:
    """Context manager to profile a block of code and save statistics.

    Saves the profiling data to a binary `.prof` file, which can be analyzed
    with tools like SnakeViz. If disabled, the context manager has near-zero
    overhead.

    Args:
        filename (str | None, optional): The name for the output file (without
            extension). If None, a timestamped filename is generated.
            Defaults to None.
        output_dir (str, optional): The directory where the profile file will
            be saved. Defaults to "profiles".
        create_dir (bool, optional): If True, creates the output directory if
            it does not exist. Defaults to True.
        enabled (bool, optional): If True, profiling is activated. If False,
            the context manager does nothing. Defaults to False.

    Yields:
        Iterator[Path | None]: The path to the generated `.prof` file if
            profiling is enabled, otherwise None.

    """
    if not enabled:
        yield None
        return

    if create_dir:
        Path(output_dir).mkdir(exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"profile_{timestamp}"

    # Ensure the file has a .prof extension
    filepath = Path(output_dir) / (Path(filename).stem + ".prof")

    profiler = cProfile.Profile()
    profiler.enable()

    try:
        yield filepath  # Restituisce il path per riferimento
    finally:
        profiler.disable()
        # Save the raw profiling statistics to the file
        profiler.dump_stats(filepath)
        print(f"Profiler data saved to: {filepath.resolve()}")
