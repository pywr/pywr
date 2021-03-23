

cdef class Component:
    cdef object _name
    cdef readonly object model
    cdef public str comment
    cdef public dict tags
    cdef readonly object parents
    cdef readonly object children
    cpdef setup(self)
    cpdef reset(self)
    cpdef before(self)
    cpdef after(self)
    cpdef finish(self)
