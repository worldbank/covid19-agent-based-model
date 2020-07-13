import os

def get_data_dir(*args):
    dname = os.path.dirname(os.path.abspath(__file__))  # covid19_abm
    dname = os.path.dirname(dname)  # src
    dname = os.path.dirname(dname)  # covid19-agent-based-model

    return os.path.join(dname, 'data', *args)
