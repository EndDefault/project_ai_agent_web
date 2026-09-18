"""여러 Tool에서 공통으로 사용하는 실행 시간 측정 데코레이터."""

from functools import wraps
from time import perf_counter


def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        started = perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = perf_counter() - started
            print(f"[Tool: {func.__name__}] {elapsed:.2f}초", flush=True)

    return wrapper
