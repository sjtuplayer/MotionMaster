# scripts/resource.py

config = None

def set_config(new_config):
    """修改全局 config"""
    global config
    config = new_config

def get_config():
    """获取当前的全局 config"""
    return config