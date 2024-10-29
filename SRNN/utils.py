import numpy as np


def convert_to_preferred_format(sec):
    day= int(sec/3600/24)
    sec = sec % (24 * 3600)
    hour = sec // 3600
    sec %= 3600
    min = sec // 60
    sec %= 60
    return "%02dd%02dh%02dm%02ds" % (day, hour, min, sec)


def compute_time(start_time,end_time,epochs,epoch):
    each_time=(end_time-start_time)/epoch
    rest_time=(epochs-epoch)*each_time
    return print("Training Still Needs :-",convert_to_preferred_format(rest_time), " (Estimated)." )
    
    