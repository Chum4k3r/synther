#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from synther import Synther, Harmonica


def main() -> None:
    synther = Synther(
        samplerate=48000,
        buffer_size=128,
        num_buffers=4,
        instrument=Harmonica()
    )

    synther.run()
    return


if __name__ == '__main__':
    main()
