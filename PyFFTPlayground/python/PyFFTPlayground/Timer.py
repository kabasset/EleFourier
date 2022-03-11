#
# Copyright (C) 2012-2020 Euclid Science Ground Segment
#
# This library is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3.0 of the License, or (at your option)
# any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
#

"""
:file: python/PyFFTPlayground/Timer.py

:date: 12/15/21
:author: https://realpython.com/python-timer/

"""
import time
from dataclasses import dataclass, field


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class
       Taken from https://realpython.com/python-timer/
    """


@dataclass
class Timer:
    """Store each increment and the total elapsed time (elapsed_time should be computed from incs)"""

    _start_time: float = None
    incs: list[float] = field(default_factory=list)
    elapsed_time: float = 0

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        # Get increments in milli-seconds
        inc = (time.perf_counter() - self._start_time) * 1000
        self.elapsed_time += inc
        self.incs.append(inc)
        self._start_time = None

        return inc
