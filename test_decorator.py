# 装饰器函数
def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

# 使用装饰器
@my_decorator
def say_hello():
    print("Hello!")

# 调用被装饰后的函数
say_hello()
