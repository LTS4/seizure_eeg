"""Download data from TUH server with rsync"""
import sys
from pathlib import Path
from time import sleep

import pexpect


def download(source: Path, target: Path, password: str):
    """Download data from `source` to `target`"""

    success = False
    while not success:
        # spawn rsync
        child = pexpect.spawn(
            f"rsync -auxvL nedc@www.isip.piconepress.com:data/{source} {target}", encoding="utf-8"
        )
        child.logfile_read = sys.stdout

        case1 = child.expect_exact(
            ["nedc@www.isip.piconepress.com's password: ", pexpect.EOF],
            timeout=None,
        )
        if case1 == 0:
            child.sendline(password)
        else:
            sleep(10)
            continue

        case2 = child.expect_exact(
            [
                "receiving file list",
                "Permission denied, please try again.",
                pexpect.EOF,
            ],
            timeout=None,
        )
        if case2 == 0:
            child.expect(pexpect.EOF, timeout=None)
            child.close()

            if child.exitstatus == 0:
                success = True
        elif case2 == 1:
            raise ValueError("Invalid password, verify your .env file or NEDC_PASSWORD variable")
