import pycuda.driver as drv
import logging


def query_available_gpus():
    drv.init()
    gpu_count = drv.Device.count()

    if gpu_count > 0:
        logging.info("%d GPU(s) found:" % drv.Device.count())

        for ordinal in range(drv.Device.count()):
            dev = drv.Device(ordinal)
            logging.info('#' + str(ordinal) + ' ' +  dev.name())
    else:
        logging.info("No GPUs found.")

    return gpu_count