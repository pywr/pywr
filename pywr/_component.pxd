

cdef class Component:
    cdef object _name
    cdef object _model
    cdef public basestring comment
    cdef readonly object parents
    cdef readonly object children
    cpdef setup(self)
    cpdef reset(self)
    cpdef before(self)
    cpdef after(self)
    cpdef finish(self)