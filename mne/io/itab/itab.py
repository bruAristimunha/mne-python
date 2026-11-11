# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

from ..._fiff.utils import _read_segments_file
from ...utils import _check_fname, _verbose_control, fill_doc_static, verbose_static
from ..base import BaseRaw
from .constants import ITAB
from .info import _mhd2info
from .mhd import _read_mhd


@fill_doc_static("preload", "verbose")
class RawITAB(BaseRaw):
    """Raw object from an ITAB MEG file.

    Parameters
    ----------
    fname : path-like
        Path to the ITAB data file (``.raw``). The header file with the same name
        plus ``.mhd`` (e.g., ``run01.raw.mhd``) must be in the same directory.
    preload : bool | str
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires
        large amount of memory). If preload is a string, it is the name of a
        freshly created memory-mapped file used to store the data on the hard
        drive (slower, requires less memory). An existing file is overwritten.
        The caller owns the file and is responsible for removing it after the
        Raw object is no longer in use. For supported Raw readers, the exact string
        ``"auto"`` instead reuses decoded data below the directory configured by
        :func:`mne.set_cache_dir`. Entries persist without a size limit and are mapped
        copy-on-write. Use ``Path("auto")`` for a literal filename.

        .. versionchanged:: 1.13
           Support for the ``"auto"`` decoded-data cache was added.
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods.
    """

    @_verbose_control
    def __init__(self, fname, preload=False, *, verbose=None):
        fname = _check_fname(fname, "read", True, "fname")
        fname_mhd = fname.with_name(fname.name + ".mhd")
        _check_fname(fname_mhd, "read", True, "ITAB header file")
        mhd = _read_mhd(fname_mhd)
        info, annotations = _mhd2info(mhd)
        little_endian = mhd["data_type"] == ITAB.DATA_TYPE_LITTLE_ENDIAN
        raw_extras = dict(
            dtype="<i4" if little_endian else ">i4",
            n_channels=mhd["nchan"],
            offset=mhd["start_data"],
        )
        super().__init__(
            info,
            preload,
            last_samps=[mhd["ntpdata"] - 1],
            filenames=[fname],
            raw_extras=[raw_extras],
            orig_units={ch["label"]: ch["unit"] for ch in mhd["ch"]},
            orig_format="int",
            verbose=verbose,
        )
        self.set_annotations(annotations)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        extras = self._raw_extras[fi]
        _read_segments_file(
            self,
            data,
            idx,
            fi,
            start,
            stop,
            cals,
            mult,
            dtype=extras["dtype"],
            n_channels=extras["n_channels"],
            offset=extras["offset"],
        )


@verbose_static("preload")
def read_raw_itab(
    fname: Path | str,
    preload: bool | str = False,
    *,
    verbose: bool | str | int | None = None,
) -> RawITAB:
    """Reader for an ITAB MEG file.

    Parameters
    ----------
    fname : path-like
        Path to the ITAB data file (``.raw``). The header file with the same name
        plus ``.mhd`` (e.g., ``run01.raw.mhd``) must be in the same directory.
    preload : bool | str
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires
        large amount of memory). If preload is a string, it is the name of a
        freshly created memory-mapped file used to store the data on the hard
        drive (slower, requires less memory). An existing file is overwritten.
        The caller owns the file and is responsible for removing it after the
        Raw object is no longer in use. For supported Raw readers, the exact string
        ``"auto"`` instead reuses decoded data below the directory configured by
        :func:`mne.set_cache_dir`. Entries persist without a size limit and are mapped
        copy-on-write. Use ``Path("auto")`` for a literal filename.

        .. versionchanged:: 1.13
           Support for the ``"auto"`` decoded-data cache was added.
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

    Returns
    -------
    raw : instance of RawITAB
        A Raw object containing ITAB data.
        See :class:`mne.io.Raw` for documentation of attributes and methods.

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods of RawITAB.

    Notes
    -----
    The ``.mhd`` header stores the sensor positions and the head markers relative
    to the subject's head, so ``info["dev_head_t"]`` is the identity. The first three
    markers are read as the nasion, right and left preauricular points, the fourth
    as an extra head point, and any others as HPI coils.

    .. versionadded:: 1.14
    """
    return RawITAB(fname, preload=preload, verbose=verbose)
