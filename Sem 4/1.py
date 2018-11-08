"""
import time, threading

def foo():
    print(time.ctime())
    threading.Timer(5, foo).cancel()

foo()
"""
def logger(foo):
    def kek(*args, **kwargs):
        print('Called {name} function'.format(name=foo.__name__))
        return foo(*args, **kwargs)
    return kek

@logger
def add(a):
    return a + 1

def class_decor(decor):
    def decorate(cls):
        for attr in cls.__dict__:
            if callable(getattr(cls, attr)):        
                setattr(cls, attr, decor(getattr(cls, attr)))
        print(cls, cls.__subclasses__())
        for sub in cls.__subclasses__():
            print(sub.__name__)
            decorate(sub)
        return cls
    return decorate

@class_decor(logger)
class VseLoggiryetsya:
    def __init__(self, val):
        self.val = val
    
    def mul(self, x):
        return self.val * x
    
    def add(self, x):
        return self.val + x

class Descendant(VseLoggiryetsya):
    def div(self, x):
        return self.val / x


class MyDescr:
    def __init__(self, val):
        self.val = val
        
    def __get__(self, obj, objtype):
        print('GETTTTTT', obj)
        return self.val + 1
    
    def __set__(self, obj, val):
        print('SETTTTTT', obj)
        self.val = val - 1

class Kek:
    kek = MyDescr(2)
    def __init__(self, val):
        self.kek = val

def wrapper(mult):
    def mul(foo):
        def bar(*args):
            return mult * foo(*args)
        return bar
    return mul

@wrapper(5)
def baz(a, b):
    return a + b


def main():
    lol = VseLoggiryetsya(4)
    print(lol.add(2))
    print(lol.mul(4))
    d = Descendant(4)
    print(d.div(2))

if __name__ == '__main__':
    main()