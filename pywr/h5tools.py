import tables
import os


class H5Store(object):
    def __init__(
        self,
        filename,
        filter_kwds=None,
        mode="r",
        title="",
        metadata=None,
        create_directories=False,
    ):
        self._opened = False
        if isinstance(filename, (str, os.PathLike)):
            # filename is a path to open
            self.filename = filename
            # Note sure how else to deal with str / unicode requirements in pytables
            # See this issue: https://github.com/PyTables/PyTables/issues/522
            import sys

            if filter_kwds:
                if sys.version_info[0] == 2 and "complib" in filter_kwds:
                    filter_kwds["complib"] = filter_kwds["complib"].encode()
                filters = tables.Filters(**filter_kwds)
            else:
                filters = None

            # Create directories for the filename if required
            if create_directories:
                try:
                    os.makedirs(os.path.dirname(filename))
                except OSError as exception:
                    import errno

                    if exception.errno != errno.EEXIST:
                        raise

            self.file = tables.open_file(
                filename, mode=mode, filters=filters, title=title
            )
            self._opened = True
        elif isinstance(filename, tables.File):
            # filename is a pytables file
            self.file = filename
            assert self.file.isopen
            self.filename = self.file.filename
            self._opened = False
        else:
            raise TypeError(
                f"{self.__class__.__name__} must be initialised with a filename to open or "
                f"an open tables.File"
            )

        # now update metadata if given
        if metadata is not None and self.file.mode != "r":
            for k, v in metadata.items():
                setattr(self.file.root._v_attrs, k, v)

    def __del__(self):
        if self._opened and self.file.isopen:
            self.file.close()
