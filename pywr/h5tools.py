import tables
import sys
from past.builtins import basestring

class H5Store(object):
    def __init__(self, filename, filter_kwds=None, mode="r", title=''):
        filter_kwds = filter_kwds
        mode = mode
        self._opened = False
        if isinstance(filename, basestring):
            # filename is a path to open
            self.filename = filename
            # Note sure how else to deal with str / unicode requirements in pytables
            # See this issue: https://github.com/PyTables/PyTables/issues/522
            import sys
            if filter_kwds:
                if sys.version_info[0] == 2 and 'complib' in filter_kwds:
                    filter_kwds['complib'] = filter_kwds['complib'].encode()
                filters = tables.Filters(**filter_kwds)
            else:
                filters = None
            self.file = tables.open_file(filename, mode=mode, filters=filters, title=title)
            self._opened = True
        elif isinstance(filename, tables.File):
            # filename is a pytables file
            self.file = filename
            assert(self.file.isopen)
            self.filename = self.file.filename
            self._opened = False
        else:
            raise TypeError("{} must be initalised with a filename to open or an open tables.File".format(self.__class__.__name__))

    def __del__(self):
        if self._opened and self.file.isopen:
            self.file.close()
