"""Bounded retrieval concurrency with durable batch identity and local progress."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, CancelledError
from contextlib import contextmanager
from contextvars import ContextVar
import threading

from .jobs import bind_job, check_job_cancelled
from .qa_recovery import bind_receipt_branch, has_serial_model_receipts


_progress = ContextVar('retrieval_progress', default=None)


@contextmanager
def retrieval_progress(callback):
    token = _progress.set(callback)
    try:
        yield
    finally:
        _progress.reset(token)


def report_retrieval(message):
    callback = _progress.get()
    if callback:
        callback(message)


def map_rerank_batches(batches, invoke, workers=3):
    """Yield completed batches; never enqueue the entire candidate set at once."""
    if has_serial_model_receipts():
        for index, (_, batch) in enumerate(batches):
            check_job_cancelled()
            yield index, invoke(batch, check_job_cancelled)
        return
    stopped = threading.Event()
    def check():
        check_job_cancelled()
        if stopped.is_set():
            raise CancelledError()
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='wechat-rerank')
    pending, position = {}, 0
    def submit(index):
        name, batch = batches[index]
        def run():
            check()
            return invoke(batch, check)
        future = pool.submit(bind_job(bind_receipt_branch('rerank:' + name, run)))
        pending[future] = index
    try:
        while position < min(workers, len(batches)):
            submit(position)
            position += 1
        while pending:
            check()
            done, _ = wait(pending, timeout=.2, return_when=FIRST_COMPLETED)
            for future in sorted(done, key=lambda f: pending[f]):
                index = pending.pop(future)
                yield index, future.result()
            for _ in done:
                if position < len(batches):
                    submit(position)
                    position += 1
    finally:
        stopped.set()
        for future in pending:
            future.cancel()
        # In-flight requests finish and persist receipts before their DB closes.
        pool.shutdown(wait=True, cancel_futures=True)


def parallel_recall(branches):
    """Independent read-only channels; stable receipt scopes survive recovery.

    Each branch owns its database connection. Join all workers before returning
    or raising so a paid embedding receipt never outlives its checkpoint DB.
    """
    with ThreadPoolExecutor(max_workers=min(4,len(branches)),thread_name_prefix='wechat-recall') as pool:
        futures = {name:pool.submit(bind_job(bind_receipt_branch('recall:'+name,invoke)))
                   for name,invoke in branches}
        return {name:future.result() for name,future in futures.items()}
