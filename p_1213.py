# 23. 12. 13

import math

class Circle:

    def __init__(self, radius):
        self.radius = radius

    # def __init__(self, radius):
    #    self.set_radius(radius)

    def get_circumference(self):
        return 2 * math.pi * self.__radius

    def get_area(self):
        return math.pi * (self.__radius ** 2)

    #
    #
    #

    def get_radius(self): # getter
        return self.__radius
    def set_radius(self, radius): # setter
        if radius <= 0:
            raise TypeError('Radius must be greater than 0.')
        self.__radius = radius

    @property
    def radius(self):
        return self.__radius
    # def radius(self):
    # return 3

    @radius.setter
    def radius(self, radius):
        self.__radius = radius

circle = Circle(20)

print(f'circumference : {circle.get_circumference()}')
print(f'area : {circle.get_area()}')

# print(circle.__radius) # AttributeError : 'Circle' object has no attribute '__radius'.

print(circle.radius) #
circle.radius = -5 #
print(circle.radius)










#
#
#

class Animal:

    def __init__(self):
        self.legs = 4

    def walking(self):
        print(f'It moves around on {self.legs} legs.')

class Human(Animal):
    def __init__(self):
        super().__init__() #
        self.legs = 2

    def thinking(self):
        print('I think...')

    def walking(self): #
        super().walking() #
        print('Actually, it walks on two feet.')

me = Human()
me.thinking() #
me.walking() #

#

class Stack:

    def __init__(self):
        self.list = []

    def push(self, val):
        self.list.append(val)
    def pop(self):
        val = self.list[-1]
        del(self.list[-1])
        return val

stack = Stack()
stack.push(10)
stack.push(20)
stack.push(30)
print(stack.list)
stack.pop()
print(stack.list)
stack.pop()
print(stack.list)
stack.pop()







#
# gpt
import math

class Circle:
    def __init__(self, radius):
        self.radius = radius  #
    @property
    def radius(self):
        return self.__radius

    @radius.setter
    def radius(self, radius):
        if radius <= 0:
            raise ValueError('Radius must be greater than 0.')
        self.__radius = radius

    def get_circumference(self):
        return 2 * math.pi * self.radius  #

    def get_area(self):
        return math.pi * (self.radius ** 2)  #

circle = Circle(20)

print(f'Circumference: {circle.get_circumference()}')
print(f'Area: {circle.get_area()}')

print(circle.radius)  # print : 20
circle.radius = 10
print(circle.radius)  # print : 10

