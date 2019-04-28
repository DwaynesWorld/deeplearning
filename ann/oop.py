class SimpleClass():
    def __init__(self, name):
        print("hello " + name)

    def yell(self):
        print("yelling")


x = SimpleClass("kt")
x.yell()


class ExtendedClass(SimpleClass):
    def __init__(self):
        super().__init__("kyle")
        print("Extend")


y = ExtendedClass()
y.yell()
