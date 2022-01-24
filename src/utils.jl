"""
    @fieldequal Supertype

Define a method of `==` for all subtypes of `Supertype`
such that `==` returns true if each pair of the field values
from two instances are equal by `==`.
"""
macro fieldequal(Supertype)
    return esc(quote
        function ==(x::T, y::T) where T <: $Supertype
            f = fieldnames(T)
            getfield.(Ref(x),f) == getfield.(Ref(y),f)
        end
    end)
end
