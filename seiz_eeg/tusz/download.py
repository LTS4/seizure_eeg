"""Download data from TUH server with rsync"""
import sys

import pexpect


def authenticate(child: pexpect.spawn, password: str):
    """Authenticate to nedc@www.isip.piconepress.com in child precess

    Args:
        child (pexpect.spawn): pexpect process
        password (str): Password for nedc@www.isip.piconepress.com

    Raises:
        ValueError: If password is invalid
    """
    case = child.expect_exact(
        [
            "nedc@www.isip.piconepress.com's password: ",
            "ECDSA key fingerprint is SHA256:J+iAVuYB8jswRPDMSet9qGWVL5xrPJ4RDe7w9LSKRyY",
        ],
        timeout=None,
    )
    if case == 0:
        child.sendline(password)

        scase = child.expect_exact(
            [
                "receiving",
                "Permission denied, please try again.",
            ],
            timeout=None,
        )
        if scase == 0:
            return
        if scase == 1:
            raise ValueError("Invalid password, verify your .env file or NEDC_PASSWORD variable")

    elif case == 1:
        child.expect_exact("Are you sure you want to continue connecting (yes/no)? ")
        child.sendline("yes")

        authenticate(child, password)


def download(version: str, target: str, password: str):
    """Download TUH seizure data

    Args:
        version (str): Seizure corpus version (e.g. "v1.5.2")
        target (str): Folder where to download data
        password (str): nedc password
    """
    while True:
        # spawn rsync
        child = pexpect.spawn(
            (
                "rsync -auxvL "
                f"nedc@www.isip.piconepress.com:data/eeg/tuh_eeg_seizure/{version}/ "
                f"{target}"
            ),
            encoding="utf-8",
        )
        child.logfile_read = sys.stdout

        authenticate(child, password)

        child.expect(pexpect.EOF, timeout=None)
        child.close()

        if child.exitstatus == 0:
            return
