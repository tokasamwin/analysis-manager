# -*- coding: utf-8 -*-
"""Shared pytest setup.

Must set ANALYSISMANAGER_DATADIR *before* analysisManager is imported
(importing the package initialises its configuration), so that the test
suite writes its data into a temporary directory instead of the user's
home directory, and never hits the interactive first-run prompt.
"""

import os
import tempfile

TEST_DATA_DIR = os.path.join(tempfile.gettempdir(), 'analysis_manager_test_data')
os.environ.setdefault('ANALYSISMANAGER_DATADIR', TEST_DATA_DIR)
