import logging
import sys

# 模拟分布式计算：只有根节点打印日志
CURRENT_RANK = 0

# 配置日志系统

logging.basicConfig(
    # 日志的级别
    level=logging.INFO,
    # 设置日志的输出格式
    # 使用%作为占位符
    # logging提供的格式字段：
    # %(asctime)s - 时间
    # %(levelname)s - 日志级别
    # %(lineno)d - 行号
    # %(message)s - 日志信息
    format="%(asctime)s - [%(levelname)s] - Line:(%(lineno)d) - %(message)s",
    # 设置时间格式
    datefmt="%H:%M:%S",
    # 日志输出位置
    stream=sys.stdout
    # 使用filename='log.txt'可以将日志输出到文件

)

# 常见的日志打印封装函数
# msg:要打印的日志信息
# *args, **kwargs: 可变参数，传递给logging.info等函数
# *args:将所有匿名的参数打包为元组传入，logging中的message是可以使用%占位符的，*args就是将占位符的参数传入
# **kwargs:将所有命名参数打包作为字典传入，一般用于传入exc_info等命名参数，比如exc_info=TRUE表示打印异常信息
def print_rank_0(msg, *args, **kwargs):
    # 设置默认的stacklevel为2，含义是：默认打印的行号是调用该函数的语句的行号，方便定位
    # 如果不指定，输出的行号一直都是logging.info的行号
    kwargs.setdefault('stacklevel', 2),
    # 分布式计算中，一般尽在根节点打印日志
    if CURRENT_RANK == 0:
        logging.info(msg, *args, **kwargs)

def demo():
    print("---使用args来传入参数---")
    epoch = 5
    loss = 0.0234
    # 这里传入的msg是带有两个占位符的，后面的epoch和loss会被打包成元组传入，作为占位符的参数
    print_rank_0("epoch: %d, loss: %.4f", epoch, loss)

    print("---使用kwargs传入命名参数---")
    # 模拟异常情况，使用exc_info=True来打印异常信息
    try:
        x = 1/0
    except ZeroDivisionError:
        print_rank_0("除0错误,错误信息", exc_info=True)


if __name__ == '__main__':
    demo()

"""
---使用args来传入参数---
21:31:42 - [INFO] - Line:(46) - epoch: 5, loss: 0.0234
---使用kwargs传入命名参数---
21:31:42 - [INFO] - Line:(53) - 除0错误,错误信息
Traceback (most recent call last):
  File "/home/hhr/Nova/learning/task2/test_logging/test_logging.py", line 51, in demo
    x = 1/0
        ~^~
ZeroDivisionError: division by zero
"""