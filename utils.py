'''
Some useful functions.
'''


def ltime():
    tm = time.localtime()
    tmstr = str(tm.tm_hour)+':'+str(tm.tm_min)+':'+str(tm.tm_sec)
    return tmstr
