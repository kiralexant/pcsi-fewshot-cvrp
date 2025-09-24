import numpy as np
from typing import Iterable, Iterator, Optional

class NumpyFixedQueue:
    """
    Fixed-size FIFO queue implemented on top of a NumPy array.

    - Appending to a full queue discards the oldest element automatically.
    - O(1) append/pop using a ring buffer.
    """

    __slots__ = ("_buf", "_maxlen", "_head", "_tail", "_count")

    def __init__(self, maxlen: int, dtype=float) -> None:
        if maxlen <= 0:
            raise ValueError("maxlen must be a positive integer")
        self._buf = np.empty(maxlen, dtype=dtype)
        self._maxlen = maxlen
        self._head = 0  # index of the oldest element
        self._tail = 0  # index where next append goes
        self._count = 0

    # --- core ops --------------------------------------------------------

    def append(self, x) -> None:
        """Append x; drop oldest if full."""
        self._buf[self._tail] = x
        self._tail = (self._tail + 1) % self._maxlen
        if self._count == self._maxlen:
            # overwrite: advance head as well (drop oldest)
            self._head = (self._head + 1) % self._maxlen
        else:
            self._count += 1

    def pop(self):
        """Pop and return the oldest element."""
        if self._count == 0:
            raise IndexError("pop from empty queue")
        x = self._buf[self._head]
        self._head = (self._head + 1) % self._maxlen
        self._count -= 1
        return x

    # --- utilities -------------------------------------------------------

    def extend(self, xs: Iterable) -> None:
        """Append many items (keeps the most recent maxlen items)."""
        for x in xs:
            self.append(x)

    def peek(self):
        """Return the oldest element without removing it."""
        if self._count == 0:
            raise IndexError("peek from empty queue")
        return self._buf[self._head]

    def clear(self) -> None:
        self._head = self._tail = 0
        self._count = 0

    def empty(self) -> bool:
        return self._count == 0

    def full(self) -> bool:
        return self._count == self._maxlen

    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Iterator:
        for i in range(self._count):
            yield self._buf[(self._head + i) % self._maxlen]

    def to_numpy(self, copy: bool = True) -> np.ndarray:
        """
        Return items from oldest->newest as a NumPy array.
        If the buffer hasn't wrapped, returns a view when copy=False.
        If wrapped, returns a contiguous copy.
        """
        if self._count == 0:
            return self._buf[:0].copy() if copy else self._buf[:0]
        h, t = self._head, self._tail
        if h < t:
            view = self._buf[h:t]
            return view.copy() if copy else view
        # wrapped: concatenate two slices to be contiguous
        return np.concatenate((self._buf[h:], self._buf[:t]))

    def __repr__(self) -> str:
        return f"NumpyFixedQueue(maxlen={self._maxlen}, data={self.to_numpy()!r})"
