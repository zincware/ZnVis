"""
ZnVis: A Zincwarecode package.
License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0
Copyright Contributors to the Zincwarecode Project.
Contact Information
-------------------
email: zincwarecode@gmail.com
github: https://github.com/zincware
web: https://zincwarecode.com/
Citation
--------
If you use this module please cite us with:

Summary
-------
Module for ZnVis processes using in testing.
"""
import multiprocessing
import traceback


class Process(multiprocessing.Process):
    """
    Process class for use in ZnVis testing.
    """

    def __init__(self, *args, **kwargs):
        """
        Multiprocessing class constructor.
        """
        multiprocessing.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = multiprocessing.Pipe()
        self._exception = None

    def run(self):
        """
        Run the process and catch exceptions.
        """
        try:
            multiprocessing.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))

    @property
    def exception(self):
        """
        Exception property to be stored by the process.
        """
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception
