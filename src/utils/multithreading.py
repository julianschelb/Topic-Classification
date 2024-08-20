# ===========================================================================
#                      Helpers for Multithreaded Operations
# ===========================================================================

import threading


def startThreads(num_threads, datasets, devices, func):
    """Starts a number of threads, and returns the results."""

    threads = []  # List to hold all the thread objects
    results = []  # List to hold the returned datasets
    # Lock to ensure thread safety when updating results
    results_lock = threading.Lock()

    def threadTarget(process_id, dataset, device):
        """Target function for each thread."""
        result_dataset = func(process_id, dataset, device)
        with results_lock:
            # Append the result to results in a thread-safe manner
            results.append(result_dataset)

    # Create threads
    for i in range(num_threads):
        thread = threading.Thread(
            target=threadTarget,
            args=(i, datasets[i], devices[i])
        )
        threads.append(thread)

    # Start threads
    for thread in threads:
        thread.start()

    # Wait for threads to finish
    for thread in threads:
        thread.join()

    return results  # Return the collected datasets
